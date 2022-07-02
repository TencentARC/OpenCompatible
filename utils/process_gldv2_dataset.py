import os
from pathlib import Path
import csv
import argparse
import random
import numpy as np
random.seed(666)
np.random.seed(666)

def generate_gallery_list(source_file, saved_file):
    saved_file = open(saved_file, 'w')
    csv_reader = csv.reader(open(source_file, encoding='utf-8'))
    csv_reader.__next__()
    count = 0
    for row in csv_reader:
        saved_file.write("index/%s/%s/%s/%s.jpg %d\n"%(row[0][0],row[0][1],row[0][2],row[0],count) )
        count += 1
    saved_file.close()
    csv_reader.close()
    print("Gallery indices are built.")

def generate_query_list(source_file, ref_file, saved_file_root, type='private'):
    query_list_file = open(os.path.join(saved_file_root, f'gldv2_{type}_query_list.txt'), 'w')
    query_gts_file = open(os.path.join(saved_file_root, f'gldv2_{type}_query_gt.txt'), 'w')

    gallery_dict = {}
    with open(ref_file, 'r') as f:
        for line in f.readlines():
            key = line.split(" ")[0].split("/")[-1][:-4]
            value = line.split(" ")[1].replace("\n","")
            gallery_dict[key] = value

    print("Gallery dict is built.")

    csv_reader = csv.reader(open(source_file, encoding='utf-8'))
    csv_reader.__next__()
    count = 0
    for row in csv_reader:
        if row[2].lower() == type:
            query_list_file.write("test/%s/%s/%s/%s.jpg %d\n" % (row[0][0], row[0][1], row[0][2], row[0], count))
            gts = []
            for gt in row[1].split(" "):
                gts.append(gallery_dict[gt])
            gts_str = ','.join(gts).replace('\n','')
            query_gts_file.write("test/%s/%s/%s/%s.jpg %d %s\n" % (row[0][0], row[0][1], row[0][2], row[0], count, gts_str))
            count += 1
        else:
            continue
    query_list_file.close()
    query_gts_file.close()
    csv_reader.close()
    print("Query indices are built.")

def split_gldv2_by_class(input_file, classes, ratio=0.3, dataset='gldv2'):
    input_file = Path(input_file)
    bucket = [[] for _ in range(classes)]
    random.seed(666)
    np.random.seed(666)
    all_class_list = np.arange(0, classes)
    np.random.shuffle(all_class_list)
    old_class_list = all_class_list[:int(classes * ratio)]
    old_class_bool_index = np.array([False for _ in range(classes)])
    old_class_bool_index[old_class_list] = True

    with open(input_file, 'r') as f:
        for line in f.readlines():
            bucket[int(line.split(" ")[-1])].append(line)

    old_class_file = open(
        input_file.parent / f"{dataset}_train_old_{int(ratio * 100)}percent_class.txt", "w")
    new_class_file = open(
        input_file.parent / f"{dataset}_train_new_{int((1-ratio) * 100)}percent_class.txt", "w")

    old_class_num, old_img_num = 0, 0
    new_class_num, new_img_num = 0, 0

    for i in range(classes):
        if old_class_bool_index[i]:
            for j in bucket[i]:
                old_class_file.write(j)
                old_img_num += 1
            old_class_num += 1
        else:
            for j in bucket[i]:
                new_class_file.write(j)
                new_img_num += 1
            new_class_num += 1
    old_class_file.close()
    new_class_file.close()
    print("Old class count: %d, img count: %d" % (old_class_num, old_img_num))
    print("New class count: %d, img count: %d" % (new_class_num, new_img_num))
    print("Done.")


def split_gldv2_by_data(input_file, classes, ratio=0.3, dataset='gldv2'):
    input_file = Path(input_file)
    bucket = np.zeros([classes], dtype=int)
    paths = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            line_splits = line.split(" ")
            current_label = int(line_splits[-1]) if line_splits[-1] != '\n' else int(line_splits[-2])
            bucket[current_label] += 1
            paths.append(line)

    old_training_data_file = open(
        input_file.parent/f"{dataset}_train_old_{int(ratio * 100)}percent.txt" , "w")

    new_training_data_file = open(
        input_file.parent / f"{dataset}_train_new_{int((1-ratio) * 100)}percent.txt", "w")

    start_indexes = np.zeros([classes], dtype=int)
    old_num = 0
    new_num = 0
    for i in range(classes):
        if bucket[i] == 0:
            continue
        if i > 0:
            start_indexes[i] = start_indexes[i - 1] + bucket[i - 1]
        curr_count = len(bucket[i])
        if curr_count == 1:
            old_training_data_file.write(bucket[i][0])
            new_training_data_file.write(bucket[i][0])
            old_num += 1
            new_num += 1
        elif curr_count>1 and curr_count < 4:
            old_training_data_file.write(bucket[i][0])
            old_num += 1
            for j in range(1, curr_count):
                new_training_data_file.write(bucket[i][j])
                new_num += 1
        else:
            random.seed(666 + i)
            np.random.seed(666 + i)
            all_list = np.arange(0, curr_count)
            np.random.shuffle(all_list)
            old_list = all_list[:int(curr_count * ratio)]
            # old_list = np.array(sorted(random.sample(range(0, current_count), int(current_count * ratio))), dtype=int)
            old_bool_index = np.array([False for _ in range(curr_count)])
            old_bool_index[old_list] = True
            for j, element in enumerate(bucket[i]):
                if old_bool_index[j]:
                    old_training_data_file.write(element)
                    old_num += 1
                else:
                    new_training_data_file.write(element)
                    new_num += 1

    old_training_data_file.close()
    new_training_data_file.close()
    print("Old data count: %d" % old_num)
    print("New data count: %d" % new_num)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--root_path', default='../annotation/gldv2', type=str, help='')
    parser.add_argument('--gene_gallery', action='store_true')
    parser.add_argument('--gene_query', action='store_true')
    parser.add_argument('--split_file', default='../annotation/gldv2/label_81313.txt', type=str, help='')
    parser.add_argument('--split_gldv2_by_data', action='store_true')
    parser.add_argument('--split_gldv2_by_class', action='store_true')
    parser.add_argument('--split_ratio', default=0.3, type=float, help='')
    args = parser.parse_args()

    root_path = args.root_path
    Path(root_path).mkdir(parents=True, exist_ok=True)

    if args.gene_gallery:
        generate_gallery_list(os.path.join(root_path, 'index.csv'),
                              os.path.join(root_path, 'gldv2_gallery_list.txt'))

    if args.gene_query:
        assert os.path.isfile(os.path.join(root_path, 'retrieval_solution_v2.1.csv')), \
            "Please generate gallery indices first"
        generate_query_list(os.path.join(root_path, 'retrieval_solution_v2.1.csv'),
                            os.path.join(root_path, 'gldv2_gallery_list.txt'),
                            root_path, type='private')

        generate_query_list(os.path.join(root_path, 'retrieval_solution_v2.1.csv'),
                            os.path.join(root_path, 'gldv2_gallery_list.txt'),
                            root_path, type='public')

    cls_num_dic = {'gldv2': 81313, 'imagenet': 1000, 'places365': 365, 'market': 1502}
    if args.split_gldv2_by_data:
        whole_training_file = os.path.join(root_path, 'label_81313.txt')
        assert os.path.isfile(whole_training_file), \
            "Please download label_81313.txt first."
        split_gldv2_by_data(whole_training_file, cls_num_dic['gldv2'], args.split_ratio, dataset='gldv2')

    if args.split_gldv2_by_class:
        whole_training_file = os.path.join(root_path, 'label_81313.txt')
        assert os.path.isfile(whole_training_file), \
            "Please download label_81313.txt first."
        split_gldv2_by_class(whole_training_file, cls_num_dic['gldv2'], args.split_ratio, dataset='gldv2')


