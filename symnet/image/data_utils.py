"""
Image data utility functions.

A lot of this code is from the Keras repo:
https://github.com/keras-team/autokeras/blob/master/autokeras/image/image_supervised.py
https://github.com/keras-team/autokeras/blob/master/autokeras/utils.py
"""

from multiprocessing import Pool, cpu_count
import os
import numpy as np
import pandas as pd
import imageio
from scipy.ndimage import zoom


def _image_to_array(img_path):
    """Read the image from the path and return it as an numpy.ndarray.

    Load the image file as an array

    Args:
        img_path: a string whose value is the image file name
    """
    if os.path.exists(img_path):
        img = read_image(img_path)
        if len(img.shape) < 3:
            img = img[..., np.newaxis]
        return img
    else:
        raise ValueError("%s image does not exist" % img_path)


def read_images(img_file_names, images_dir_path, parallel=True):
    """Read the images from the path and return their numpy.ndarray instances.
    Args:
        img_file_names: List of strings representing image file names.
        images_dir_path: Path to the directory containing images.
        parallel: (Default: True) Run _image_to_array will use multiprocessing.

    Returns:
        x_train: a list of numpy.ndarrays containing the loaded images.
    """
    img_paths = [os.path.join(images_dir_path, img_file)
                 for img_file in img_file_names]

    if os.path.isdir(images_dir_path):
        if parallel:
            pool = Pool(processes=cpu_count())
            x_train = pool.map(_image_to_array, img_paths)
            pool.close()
            pool.join()
        else:
            x_train = [_image_to_array(img_path) for img_path in img_paths]
    else:
        raise ValueError("Directory containing images does not exist")
    return np.asanyarray(x_train)


def load_image_dataset(csv_file_path, images_path, parallel=True):
    """Load images from their files and load their labels from a csv file.
    Assumes the dataset is a set of images and the labels are in a CSV file.
    The CSV file should contain two columns whose names are 'File Name' and 'Label'.
    The file names in the first column should match the file names of the images with extensions,
    e.g., .jpg, .png.
    The path to the CSV file should be passed through the `csv_file_path`.
    The path to the directory containing all the images should be passed through `image_path`.
    Args:
        csv_file_path: a string of the path to the CSV file
        images_path: a string of the path containing the directory of the images
        parallel: (Default: True) Load dataset using multiprocessing.
    Returns:
        x: Four dimensional numpy.ndarray. The channel dimension is the last dimension.
        y: a numpy.ndarray of the labels for the images
    """
    img_file_names, y = read_csv_file(csv_file_path)
    x = read_images(img_file_names, images_path, parallel)
    return np.array(x), np.array(y)


def read_csv_file(path: str):
    """
    Reads a CSV file and returns the values in its two columns. This is meant to be used
    to read file names and their corresponding labels.
    :param path: str. Path to CSV file
    :return: (filenames, labels)
    """
    df = pd.read_csv(path, header=True)
    return df[df.columns[0]], df[df.columns[1]]


def read_image(path: str):
    """
    Read an image file
    :param path: str. Path to image
    :return: The image
    """
    return imageio.imread(path)


def compute_median_dimensions(images: np.ndarray):
    """
    Compute the median of each dimension of each image in the array
    :param images: array-like. List of images
    :return: median shape
    """
    if images is None or len(images.shape) == 0:
        return []

    median_shape = np.median([x.shape for x in images], axis=0)
    return median_shape.astype(int)


def resize_images(images: np.ndarray, size=None):
    """
    Resizes all images to a fixed size.
    :param images: array-like. List of images.
    :param size: array-like. Size to resize images to
    :return: resized images
    """
    if images is None or len(images.shape) == 0:
        return images

    if size is None:
        size = compute_median_dimensions(images)

    return np.array([zoom(input=x, zoom=np.divide(size, x.shape)) for x in images])
