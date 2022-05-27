"""
    Landmark Retrieval dataset
"""
import os
import pickle

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data
import torch.utils.data.distributed
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .sampler import DistributedClassSampler, SubsetRandomSampler


# ImageFile.LOAD_TRUNCATED_IMAGES = True


def default_loader(path):
    return Image.open(path).convert('RGB')


# def warning_loader(path):
#     try:
#         return Image.open(path).convert('RGB')   # There are some corrupted images in RParis dataset
#     except(OSError, NameError):
#         print('OSError, Path:', path)
#         return None


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            # impath, imlabel = line.strip().replace("\n","").split(" ")[:2]
            impath, imlabel = line.strip().split()[:2]
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(Dataset):
    def __init__(self, root, flist, transform=None, flist_reader=default_flist_reader,
                 loader=default_loader, bbxs=None):
        self.root = root
        if type(flist) is str:
            self.imlist = flist_reader(flist)
        else:
            self.imlist = flist
        self.transform = transform
        self.loader = loader
        self.bbxs = bbxs  # for roxford and rparis query img

    def __getitem__(self, index):
        # impath, target = self.imlist[index]['impath'], self.imlist[index]['class']
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        # while img is None:
        #     # index = random.randint(0, self.__len__()-1)
        #     index += 1
        #     impath, target = self.imlist[index]
        #     img = self.loader(os.path.join(self.root, impath))
        # img1 = self.transform(img)
        # return img1, target
        # img2 = self.transform(img)
        # return img1, img2, target
        if self.bbxs is not None:
            img = img.crop(self.bbxs[index])
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imlist)


def GLDv2_train_dataloader(traindir, img_list, train_transform, distributed=False,
                           batch_size=64, num_workers=32, pin_memory=True, use_pos_sampler=False):
    train_set = ImageFilelist(traindir, img_list, train_transform, loader=default_loader)

    if distributed:
        if use_pos_sampler:
            train_sampler = DistributedClassSampler(dataset=train_set, num_instances=2)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                        pin_memory=pin_memory, num_workers=num_workers, sampler=train_sampler,
                        drop_last=True
                        )

    return loader


def val_img_list_dataloader(dir, img_list, transform,
                            batch_size=64, num_workers=32):
    val_set = ImageFilelist(dir, img_list, transform=transform)

    data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             pin_memory=True, num_workers=num_workers)

    return data_loader


def GLDv2_test_dataloader(query_dir, query_img_list, gallery_dir, gallery_img_list, query_gts_list, transform,
                          batch_size, num_workers, distributed):
    query_set = ImageFilelist(query_dir, query_img_list, transform)
    gallery_set = ImageFilelist(gallery_dir, gallery_img_list, transform)

    if distributed:
        indices_query = np.array_split(np.arange(len(query_set)), dist.get_world_size())[dist.get_rank()]
        sampler_query = SubsetRandomSampler(indices_query)
        indices_gallery = np.array_split(np.arange(len(gallery_set)), dist.get_world_size())[dist.get_rank()]
        sampler_gallery = SubsetRandomSampler(indices_gallery)
    else:
        sampler_query, sampler_gallery = None, None

    query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
                              sampler=sampler_query)
    gallery_loader = DataLoader(gallery_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                                num_workers=num_workers, sampler=sampler_gallery)

    query_gts_sets = [[], [], []]  # [img_name: str, img_index: int, gts: int list]
    with open(query_gts_list, 'r') as f:
        for line in f.readlines():
            img_name, img_index, tmp_gts = line.split(" ")
            gts = [int(i) for i in tmp_gts.split(",")]
            query_gts_sets[0].append(img_name)
            query_gts_sets[1].append(int(img_index))
            query_gts_sets[2].append(gts)
    return query_loader, gallery_loader, query_gts_sets


def ROxford_test_dataloader(pkl_path, query_dir, gallery_dir, transform, batch_size, num_workers, distributed):
    with open(pkl_path, 'rb') as f:
        pkl_file = pickle.load(f)

    query_imlist, gallery_imlist, bbx_list = [], [], []
    query_img_names = pkl_file['qimlist']
    for i, query in enumerate(query_img_names):
        query_imlist.append([query + '.jpg', i])

    gallery_img_names = pkl_file['imlist']
    for i, gallery in enumerate(gallery_img_names):
        gallery_imlist.append([gallery + '.jpg', i])

    for i in range(len(pkl_file['gnd'])):
        bbx_list.append(pkl_file['gnd'][i]['bbx'])

    query_set = ImageFilelist(query_dir, query_imlist, transform, bbxs=bbx_list)  # Attention: use bbxs option here
    gallery_set = ImageFilelist(gallery_dir, gallery_imlist, transform)

    if distributed:
        indices_query = np.array_split(np.arange(len(query_set)), dist.get_world_size())[dist.get_rank()]
        sampler_query = SubsetRandomSampler(indices_query)
        indices_gallery = np.array_split(np.arange(len(gallery_set)), dist.get_world_size())[dist.get_rank()]
        sampler_gallery = SubsetRandomSampler(indices_gallery)
    else:
        sampler_query, sampler_gallery = None, None

    query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
                              sampler=sampler_query)
    gallery_loader = DataLoader(gallery_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                                num_workers=num_workers, sampler=sampler_gallery)

    return query_loader, gallery_loader, pkl_file['gnd']
