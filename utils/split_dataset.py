import os
import random

import matplotlib.pyplot as plt
import numpy as np

random.seed(666)


def split_datasets_class(input_file, classes, is_plot=False):
    bucket = np.zeros([classes], dtype=int)
    paths = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            bucket[int(line.split(" ")[-1])] += 1
            paths.append(line)

    old_class_list = np.array(sorted(random.sample(range(0, classes), classes // 2)), dtype=int)
    class_bool_index = np.array([False for _ in range(classes)])
    class_bool_index[old_class_list] = True
    new_class_list = []
    for i in range(classes):
        if not class_bool_index[i]:
            new_class_list.append(i)
    new_class_list = np.array(new_class_list)

    old_class_file = open("old.txt", "w")
    new_class_file = open("new.txt", "w")

    writen_lines = 0
    for index, is_old in enumerate(class_bool_index):

        if is_old:
            for i in range(bucket[index]):
                old_class_file.write(paths[writen_lines + i])
        else:
            for i in range(bucket[index]):
                new_class_file.write(paths[writen_lines + i])
        writen_lines += bucket[index]
    old_class_file.close()
    new_class_file.close()

    if is_plot:
        plt.bar(old_class_list, bucket[old_class_list], label='Old class')
        plt.bar(new_class_list, bucket[new_class_list], label='New class')

        plt.xticks()
        plt.legend(loc="best")
        plt.ylabel('Count')
        plt.xlabel('class no.')
        plt.savefig("./" + "stats" + ".pdf", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show(bbox_inches='tight', pad_inches=0)

    old_all_images = np.sum(bucket[old_class_list])
    new_all_images = np.sum(bucket[new_class_list])
    print("All old images: ", old_all_images)
    print("All new imamges: ", new_all_images)


def split_datasets_data(input_file, classes, percent=0.5, train_or_test='train', dataset='imagenet'):
    bucket = np.zeros([classes], dtype=int)
    paths = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            line_splits = line.split(" ")
            current_label = int(line_splits[-1]) if line_splits[-1] != '\n' else int(line_splits[-2])
            bucket[current_label] += 1
            paths.append(line)

    if not os.path.exists("../data/%s" % dataset):
        os.mkdir("../data/%s" % dataset)

    old_training_data_file = open(
        "../data/%s/%s_%s_old_%dpercent.txt" % (dataset, dataset, train_or_test, percent * 100), "w")

    new_training_data_file = open(
        "../data/%s/%s_%s_new_%dpercent.txt" % (dataset, dataset, train_or_test, (1 - percent) * 100), "w")

    start_indexes = np.zeros([classes], dtype=int)
    for i in range(classes):
        if bucket[i] == 0:
            continue
        if i > 0:
            start_indexes[i] = start_indexes[i - 1] + bucket[i - 1]
        picked_indexes = np.array(sorted(random.sample(range(0, bucket[i]), int(bucket[i] * percent))), dtype=int)
        bool_indexes = np.array([False for _ in range(bucket[i])])
        bool_indexes[picked_indexes] = True

        for j in range(bucket[i]):
            if bool_indexes[j]:
                old_training_data_file.write(paths[j + start_indexes[i]])
            else:
                new_training_data_file.write(paths[j + start_indexes[i]])

    old_training_data_file.close()
    new_training_data_file.close()


def split_roxford_rparis_data(input_file, classes, percent=0.5, train_or_test='test', dataset='roxford'):
    flag_class = "#"
    start_indexes = np.zeros([classes + 1], dtype=int)
    current_index = 0
    count = 0
    paths = []
    with open(input_file, "r") as f:
        for line in f.readlines():
            line_splits = line.split(" ")
            current_class = line_splits[0].split("_")[0] if dataset == 'roxford' else line_splits[0].split("_")[1]
            # Initialization
            if flag_class == "#": flag_class = current_class
            if current_class != flag_class:
                current_index += 1
                start_indexes[current_index] = count
                flag_class = current_class

            count += 1
            paths.append(line)
    start_indexes[-1] = count

    if not os.path.exists("../data/%s" % dataset):
        os.mkdir("../data/%s" % dataset)

    old_training_data_file = open(
        "../data/%s/%s_%s_old_%dpercent.txt" % (dataset, dataset, train_or_test, percent * 100), "w")
    new_training_data_file = open(
        "../data/%s/%s_%s_new_%dpercent.txt" % (dataset, dataset, train_or_test, (1 - percent) * 100), "w")

    for i in range(1, classes + 1):
        current_count = start_indexes[i] - start_indexes[i - 1]
        picked_indexes = np.array(sorted(random.sample(range(0, current_count), int(current_count * percent))),
                                  dtype=int)
        bool_indexes = np.array([False for _ in range(current_count)])
        bool_indexes[picked_indexes] = True

        for j in range(current_count):
            if bool_indexes[j]:
                old_training_data_file.write(paths[j + start_indexes[i - 1]])
            else:
                new_training_data_file.write(paths[j + start_indexes[i - 1]])

    old_training_data_file.close()
    new_training_data_file.close()


def refine_txt(input_file):
    with open(input_file, "r") as f:
        refined_file = open('./refined_market.txt', 'w')
        for line in f.readlines():
            line_splits = line.split(" ")
            current_label = (line_splits[-1]) if line_splits[-1] != '\n' else (line_splits[-2])
            refined_file.write(line_splits[1] + "/" + line_splits[0] + " " + line_splits[1] + '\n')
        refined_file.close()


if __name__ == '__main__':
    # input_file = '../data/imgnet_train_img_list_for_new.txt'
    # split_datasets_data(input_file, classes=1000, percent=0.5, train_or_test = 'train', dataset='imagenet')

    # input_file = '../data/places/val_list.txt'
    # split_datasets_data(input_file, classes=365, percent=0.5, train_or_test='test', dataset='places')

    # input_file = '../data/market1501/market1501_train_old_30percent.txt'
    # # split_datasets_data(input_file, classes=1502, percent=0.3, train_or_test='train', dataset='market1501')
    # refine_txt(input_file)

    # input_file = './data/roxford_img_list.txt'
    # split_roxford_rparis_data(input_file = input_file, classes=17, percent=0.5, train_or_test = 'test', dataset='roxford')

    input_file = './data/rparis_img_list.txt'
    split_roxford_rparis_data(input_file=input_file, classes=12, percent=0.5, train_or_test='test', dataset='rparis')
