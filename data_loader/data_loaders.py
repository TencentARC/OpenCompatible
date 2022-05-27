import os

from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader.GLDv2 import GLDv2_train_dataloader, GLDv2_test_dataloader, ROxford_test_dataloader

cls_num_dic = {'roxford': 81313, 'imagenet': 1000, 'places365': 365, 'market': 1502}
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class GLDv2TrainDataset(DataLoader):
    """
    Google Landmark v2 dataset
    """

    def __init__(self, args):
        train_trans = transforms.Compose([
            transforms.RandomResizedCrop(args.dataset.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        self.data_dir = args.dataset.data_dir
        self.train_loader = GLDv2_train_dataloader(args.dataset.data_dir, args.dataset.img_list, train_trans,
                                                   args.distributed, batch_size=args.trainer.batch_size,
                                                   num_workers=args.trainer.workers,
                                                   use_pos_sampler=args.use_pos_sampler)
        self.class_num = cls_num_dic['gldv2']


class GLDv2EvalDataset(DataLoader):
    """
        Google Landmark v2 dataset
    """

    def __init__(self, args):
        val_trans = transforms.Compose([
            transforms.Resize([args.test_dataset.img_size, args.test_dataset.img_size]),
            transforms.ToTensor(),
            normalize,
        ])

        query_dir, gallery_dir = args.test_dataset.data_dir, args.test_dataset.data_dir
        query_img_list = os.path.join(args.trainer.img_list_dir, "gldv2_public_query_list.txt")
        gallery_img_list = os.path.join(args.trainer.img_list_dir, "gldv2_gallery_list.txt")
        query_gts_list = os.path.join(args.trainer.img_list_dir, "gldv2_public_query_gt.txt")
        self.query_loader, self.gallery_loader, self.query_gts = GLDv2_test_dataloader(query_dir, query_img_list, \
                                                                                       gallery_dir, \
                                                                                       gallery_img_list, query_gts_list, \
                                                                                       transform=val_trans, \
                                                                                       batch_size=args.trainer.batch_size, \
                                                                                       num_workers=args.trainer.workers, \
                                                                                       distributed=args.distributed)


class GLDv2TestDataset(DataLoader):
    """
        Google Landmark v2 dataset
    """

    def __init__(self, args):
        val_trans = transforms.Compose([
            transforms.Resize([args.test_dataset.img_size, args.test_dataset.img_size]),
            transforms.ToTensor(),
            normalize,
        ])

        query_dir, gallery_dir = args.test_dataset.data_dir, args.test_dataset.data_dir
        query_img_list = os.path.join(args.trainer.img_list_dir, "gldv2_private_query_list.txt")
        gallery_img_list = os.path.join(args.trainer.img_list_dir, "gldv2_gallery_list.txt")
        query_gts_list = os.path.join(args.trainer.img_list_dir, "gldv2_private_query_gt.txt")
        self.query_loader, self.gallery_loader, self.query_gts = GLDv2_test_dataloader(query_dir, query_img_list, \
                                                                                       gallery_dir, \
                                                                                       gallery_img_list, query_gts_list, \
                                                                                       transform=val_trans, \
                                                                                       batch_size=args.trainer.batch_size, \
                                                                                       num_workers=args.trainer.workers, \
                                                                                       distributed=args.distributed)


class ROxfordTestDataLoader(DataLoader):
    """
        Revisited Oxford/Paris test dataset
    """

    def __init__(self, args):
        val_trans = transforms.Compose([
            transforms.Resize([args.img_test_size, args.img_test_size]),
            transforms.ToTensor(),
            normalize,
        ])
        test_dataset_name = args.test_dataset.name
        pkl_path = os.path.join(args.test_dataset.data_dir, f"{test_dataset_name}.pkl")

        query_dir, gallery_dir = args.test_dataset.data_dir, args.test_dataset.data_dir
        self.query_loader, self.gallery_loader, self.query_gts = ROxford_test_dataloader(
                                                                 pkl_path, query_dir, \
                                                                 gallery_dir, \
                                                                 transform=val_trans, \
                                                                 batch_size=args.trainer.batch_size, \
                                                                 num_workers=args.trainer.workers, \
                                                                 distributed=args.distributed)


class MS1Mv3TrainDataLoader(DataLoader):
    """
        MS1Mv3 training dataset
    """

    def __init__(self, args):
        pass


class IJBCTestDataLoader(DataLoader):
    """
        IJB-C test dataset
    """

    def __init__(self, args):
        pass
