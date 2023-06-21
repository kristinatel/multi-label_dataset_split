import os
import shutil
import csv
import numpy as np
import pandas as pd
import collections
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import argparse


# Initialize variables
class_list_file = './classes.txt'
csv_path = './dataset.csv'
image_names_column = []
kfold_train = pd.DataFrame()
kfold_test = pd.DataFrame()


def multilabel_txt_to_csv(images_path, labels_path, class_list_file):
    """This function converts a yolo format dataset into a csv file containing the names of images
    along with the list of classes present in each image"""

    labels_column = []

    # Get the names of all images in a list
    for r, d, f in os.walk(images_path):
        for _file in f:
            image_names_column.append(_file)
    # Get the labels from the txt files and make a label column list of all image labels (without duplicates)
    for file in os.listdir(labels_path):
        with open(os.path.join(labels_path, file)) as txt:
            labels = []
            for line in txt:
                words = line.split()
                for i in range(len(class_list)):
                    if int(words[0]) is i:
                        if class_list[i] not in labels:
                            labels.append(class_list[i])
            labels_column.append(labels)

    # Create csv data
    header = ['images', 'labels']
    csv_rows = zip(image_names_column, labels_column)

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in csv_rows:
            writer.writerow(row)
    return image_names_column, class_list


def stratified_split(csv_path, class_list, n_folds, val_size):
    """This function performs a shuffled stratified k-fold split of a multi-label dataset,
    and creates the new datasets in the 'split_datasets' directory"""
    df = pd.read_csv(csv_path)
    print(df.head())
    print(f'...{len(df)} images and labels.')
    df['labels'] = df['labels'].apply(lambda x: x[1:-1].replace("'", "").replace(',', ''))
    labels = df['labels']
    labels_list = labels.to_list()
    c = collections.Counter(labels_list)
    print(c)

    sep_labels = []
    for label in labels_list:
        sep_labels.extend(label.split(' '))
    cs = collections.Counter(sep_labels)
    # print(cs)

    key = {label: i for i, label in enumerate(cs.keys())}
    print(f' Classes: {key}')

    text_to_category = {label: [] for label in cs.keys()}
    for idx, item in df.iterrows():
        for label in text_to_category:
            if label in item['labels']:
                text_to_category[label].append(1)
            else:
                text_to_category[label].append(0)

    for label in text_to_category:
        df[label] = text_to_category[label]
    # print(df.head())

    X, Y = labels.to_numpy(), df[class_list].to_numpy(dtype=np.float32)
    msss = MultilabelStratifiedShuffleSplit(n_splits=n_folds, test_size=val_size, random_state=0)

    n_folds = 1
    for train_index, test_index in msss.split(X, Y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        train_list = train_index.tolist()
        test_list = test_index.tolist()
        for i in range(len(train_list)):
            train_list[i] = image_names_column[train_list[i]]

        for i in range(len(test_list)):
            test_list[i] = image_names_column[test_list[i]]

        kfold_train[f'fold_{n_folds}'] = train_list
        kfold_test[f'fold_{n_folds}'] = test_list
        n_folds += 1

    # Check if split_datasets folder exists in directory
    if os.path.exists('./split_datasets'):
        shutil.rmtree('./split_datasets')

    # Make directories
    for i in range(kfold_train.shape[1]):
        os.makedirs(f'./split_datasets/fold_{i+1}/train/images')
        os.makedirs(f'./split_datasets/fold_{i+1}/train/labels')
        for file in kfold_train[f'fold_{i+1}']:
            og_path = os.path.join(args.images_path, file)
            target_path = os.path.join(f'./split_datasets/fold_{i+1}/train/images', file)
            shutil.copyfile(og_path, target_path)

            og_txt_path = os.path.join(args.labels_path, file.replace('.jpg', '.txt'))
            target_txt_path = os.path.join(f'./split_datasets/fold_{i + 1}/train/labels', file.replace('.jpg', '.txt'))
            shutil.copyfile(og_txt_path, target_txt_path)

        os.makedirs(f'./split_datasets/fold_{i + 1}/val/images')
        os.makedirs(f'./split_datasets/fold_{i + 1}/val/labels')
        for file in kfold_test[f'fold_{i+1}']:
            og_path = os.path.join(args.images_path, file)
            target_path = os.path.join(f'./split_datasets/fold_{i + 1}/val/images', file)
            shutil.copyfile(og_path, target_path)

            og_txt_path = os.path.join(args.labels_path, file.replace('.jpg', '.txt'))
            target_txt_path = os.path.join(f'./split_datasets/fold_{i + 1}/val/labels', file.replace('.jpg', '.txt'))
            shutil.copyfile(og_txt_path, target_txt_path)

    print('Done!\n'"New dataset(s) in folder 'split_datasets'!")

    # Delete dataset.csv file
    os.remove(csv_path)


if __name__ == '__main__':
    # Create command-line arguments parser
    parser = argparse.ArgumentParser(description='Script for converting YOLO format dataset into CSV and performing stratified split.')
    parser.add_argument('--images_path', type=str, help='Path to the images directory', required=True)
    parser.add_argument('--labels_path', type=str, help='Path to the labels directory', required=True)
    parser.add_argument('--n_folds', type=int, default=1, help='Number of folds for stratified split (default: 1)')
    parser.add_argument('--val_size', type=float, default=0.2, help='Fraction of the dataset to be used for validation (default: 0.2)')
    args = parser.parse_args()

    # Read class list from file
    with open(class_list_file, 'r') as f:
        class_list = [line.strip() for line in f.readlines()]

    # Convert YOLO format dataset into CSV
    multilabel_txt_to_csv(args.images_path, args.labels_path, class_list)

    # Perform stratified split of the CSV dataset
    stratified_split(csv_path, class_list, args.n_folds, args.val_size)

