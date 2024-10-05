"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import cv2 as cv


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def rescale2gray(target):
    # rescale original to 8 bit values [0,255]
    x0 = np.min(target)
    x1 = np.max(target)
    y0 = 0
    y1 = 255.0
    i8 = ((target - x0) * ((y1 - y0) / (x1 - x0))) + y0

    # # create new array with rescaled values and unsigned 8 bit data type
    o8 = i8.astype(np.uint8)

    # calculate porosity
    img = cv.medianBlur(o8, 5)
    return img

def targetFindingCalPorosity(target, percent, isDisplay = False):
    '''
        target = fake image
    '''
    if type(percent) == torch.Tensor:
        percent = percent.clone().detach().cpu().numpy() 

    if type(target) == torch.Tensor:
        target = target.clone().detach().permute(1,2,0).cpu().numpy()
    # img process for calculate
    img= rescale2gray(target)

    # area finding
    # Threshold the image to create a binary image
    ret, thresh = cv.threshold(img, 100, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, 2, 1)

    cnt = contours
    big_contour = []
    max = 0
    for j in cnt:
        area = cv.contourArea(j)  #--- find the contour having biggest area ---
        if (area > max):
            max = area
            big_contour = j
    if len(big_contour) != 0:
        # create binary mask
        height = img.shape[0]
        width = img.shape[1]
        # Prepare a black canvas:
        canvas = np.zeros((height, width))

        # Draw the outer circle:
        color = (255, 255, 255)
        cv.drawContours(canvas, big_contour, -1, color, cv.FILLED)

        solid_percent = percent[0]
        hole_percent_list = 1-solid_percent[canvas != 0]
        porosity = hole_percent_list.mean()
        if isDisplay:
            print(f"Porosity >>>> {porosity}")
    else:
        porosity = 0
        if isDisplay:
            print("Empty")

    return porosity
        