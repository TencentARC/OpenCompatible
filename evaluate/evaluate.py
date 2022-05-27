import os
import time

import faiss
import numpy as np
import torch
import torch.distributed as dist

from utils.util import AverageMeter
from .roxford_rparis_metrics import calculate_mAP_roxford_rparis


# import gc


def evaluate_func(model, query_loader, gallery_loader, query_gts, logger, args, test_type='', model2=None,
                  dataset_name='gldv2'):
    model.eval()
    if model2 is not None:
        model2.eval()
    test_time = time.time()
    logger.info("=> begin eval")
    # query feat
    if test_type == '' or test_type == 'bct':
        extract_features(model, query_loader, 'q', logger, args)
    else:
        raise ValueError(f"unrecognized test_type: {test_type}")

    # gallery feat
    if test_type == 'bct':
        extract_features(model2, gallery_loader, 'db', logger, args)  # use old_model to extract
    else:
        raise ValueError(f"unrecognized test_type: {test_type}")

    dist.barrier()
    # torch.cuda.empty_cache()  # empty gpu cache if using faiss gpu index

    if args.rank == 0:
        logger.info("=> concat feat and label file")
        query_feats = concat_file(args, "feat_q", final_size=(len(query_loader.dataset), args.embedding_size))
        query_labels = concat_file(args, "label_q", final_size=(len(query_loader.dataset),))
        query_labels = query_labels.astype(np.int32)

        gallery_feats = concat_file(args, "feat_db", final_size=(len(gallery_loader.dataset), args.embedding_size))
        gallery_labels = concat_file(args, "label_db", final_size=(len(gallery_loader.dataset),))
        gallery_labels = gallery_labels.astype(np.int32)

        logger.info("=> calculate rank")
        if dataset_name == 'gldv2':
            ranked_gallery_indices = calculate_rank(logger, query_feats, gallery_feats, topk=100)
            logger.info("=> calculate mAP")
            mAP = calculate_mAP_gldv2(ranked_gallery_indices, query_gts[2], topk=100)
        elif dataset_name == 'roxford' or dataset_name == 'rparis':
            ranked_gallery_indices = calculate_rank(logger, query_feats, gallery_feats, topk=gallery_feats.shape[0])
            logger.info("=> calculate mAP")
            mAP = calculate_mAP_roxford_rparis(logger, ranked_gallery_indices.transpose(), query_gts)
        else:
            raise ValueError
        logger.info(f"mAP: {mAP:.4f}")
    dist.barrier()
    # torch.cuda.empty_cache()
    test_time = time.time() - test_time
    logger.info(f"Testing takes {test_time / 60.0:.2f} minutes")
    return mAP if args.rank == 0 else 0.0


@torch.no_grad()
def extract_features(model, data_loader, tag, logger, args):
    batch_time = AverageMeter('Process Time', ':6.3f')
    data_time = AverageMeter('Test Data Time', ':6.3f')

    labels_all = np.empty(len(data_loader.sampler), dtype=np.float32)
    features_all = np.empty((len(data_loader.sampler), args.embedding_size), dtype=np.float32)
    pointer = 0

    end = time.time()
    for i, (images, labels) in enumerate(data_loader):
        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            feat, _ = model(images.cuda())
        batchsize = labels.size(0)
        features_all[pointer:pointer + batchsize] = feat.cpu().numpy()
        labels_all[pointer:pointer + batchsize] = labels.numpy()
        pointer += batchsize

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Extract Features: [{}/{}]\t'
                        'Time {:.3f} ({:.3f})\t'
                        'Data {:.3f} ({:.3f})\t'
                        .format(i, len(data_loader),
                                batch_time.val, batch_time.avg,
                                data_time.val, data_time.avg))

    np.save(os.path.join(args.output, f'feat_{tag}_rank{dist.get_rank()}.npy'), features_all)
    np.save(os.path.join(args.output, f'label_{tag}_rank{dist.get_rank()}.npy'), labels_all)


def concat_file(args, file_name, final_size):
    concat_ret = np.empty(final_size, dtype=np.float32)
    pointer = 0
    for rank in range(dist.get_world_size()):
        file_path = os.path.join(args.output, f"{file_name}_rank{rank}.npy")
        data = np.load(file_path)
        data_size = data.shape[0]
        concat_ret[pointer:pointer + data_size] = data
        pointer += data_size
        os.remove(file_path)
    save_path = os.path.join(args.output, f"{file_name}.npy")
    np.save(save_path, concat_ret)
    return concat_ret


def calculate_rank(logger, query_feats, gallery_feats, topk):
    logger.info(f"query_feats shape: {query_feats.shape}")
    logger.info(f"gallery_feats shape: {gallery_feats.shape}")
    num_q, feat_dim = query_feats.shape

    logger.info("=> build faiss index")
    gallery_feats = gallery_feats / np.linalg.norm(gallery_feats, axis=1)[:, np.newaxis]
    query_feats = query_feats / np.linalg.norm(query_feats, axis=1)[:, np.newaxis]
    faiss_index = faiss.IndexFlatIP(feat_dim)
    # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
    faiss_index.add(gallery_feats)
    logger.info("=> begin faiss search")
    _, ranked_gallery_indices = faiss_index.search(query_feats, topk)
    return ranked_gallery_indices


def calculate_mAP_gldv2(ranked_gallery_indices, query_gts, topk):
    num_q = ranked_gallery_indices.shape[0]
    average_precision = np.zeros(num_q, dtype=float)
    for i in range(num_q):
        retrieved_indices = np.where(np.in1d(ranked_gallery_indices[i], np.array(query_gts[i])))[0]
        if retrieved_indices.shape[0] > 0:
            retrieved_indices = np.sort(retrieved_indices)
            gts_all_count = min(len(query_gts[i]), topk)
            for j, index in enumerate(retrieved_indices):
                average_precision[i] += (j + 1) * 1.0 / (index + 1)
            average_precision[i] /= gts_all_count
    return np.mean(average_precision)
