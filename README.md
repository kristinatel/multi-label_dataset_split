# multi-label_dataset_split

This is a Python script that performs a shuffled stratified k-fold split on a multi-label dataset in YOLO format. It splits the dataset into training and validation sets while ensuring that the distribution of labels is preserved.

## Requirements

To use this tool, you need to have the following dependencies installed:

- Python (version 3.6 or higher)
- [pandas](https://github.com/pandas-dev/pandas)(licensed under the [BSD 3-Clause License](https://github.com/pandas-dev/pandas/blob/master/LICENSE))
- [numpy](https://github.com/numpy/numpy)(licensed under the [BSD 3-Clause License](https://github.com/numpy/numpy/blob/main/LICENSE.txt))
- [iterstrat](https://github.com/trent-b/iterative-stratification) ((licensed under the [BSD 3-Clause License](https://github.com/trent-b/iterative-stratification/blob/master/LICENSE))

You can install the required packages by running the following command:

`pip install -r requirements.txt`

## Usage

1. Ensure that your dataset is in YOLO format, with an images folder and a corresponding labels folder.

2. Edit the `classes.txt` file to contain all the classes in your dataset, with each class on one line.

3. Open a terminal or command prompt, navigate to the directory containing the `split.py` script, and run the following command:

`python split.py --images_path <images_path> --labels_path <labels_path> --n_folds <number_of_folds> --val_size <val_size_fraction>`

- `--images_path`: Path to the directory containing the image files.
- `--labels_path`: Path to the directory containing the label files.
- `--n_folds` (optional): Number of dataset folds to create. Defaults to 1 if not specified.
- `--val_size` (optional): Fraction of the dataset to be used for testing. Defaults to 0.2 if not specified.

4. The script will create a directory named `split_datasets` that will contain the shuffled stratified splits of the dataset.

- The `split_datasets` directory will have subdirectories for each fold (e.g., `fold_1`, `fold_2`, etc.).  Where each fold is a different split of the dataset.
- Each fold directory will have `train` and `val` subdirectories.
- The `train` directory will contain the images and label files for training, while the `val` directory will contain the images and label files for validation.

## License

This project is licensed under the [MIT License](LICENSE).
