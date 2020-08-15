## Bismillahhirahmannirahim

import logging
import os

import cv2
import numpy as np

import Config


def printbanner(type):
    bannerdirpath = os.path.join(os.getcwd(), 'Banner')

    if type == 'lx2':
        fn = 'Lx2.txt'
    elif type == 'qx2':
        fn = 'Qx2.txt'
    elif type == 'ux2':
        fn = 'Ux2.txt'
    else:
        raise ValueError('Unknown banner type')

    bannerpath = os.path.join(bannerdirpath, fn)
    if os.path.isfile(bannerpath):
        with open(bannerpath, 'r') as bannerfile:
            banner = bannerfile.readlines()
            for line in banner:
                print(line.strip('\n'))


def givebalak(logfilename):
    # Logging
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    slog = logging.StreamHandler()
    slog.setLevel(logging.INFO)
    slog.setFormatter(logging.Formatter('=> %(message)s'))
    log.addHandler(slog)

    logdirpath = os.path.join(os.getcwd(), 'Log')
    logfilepath = os.path.join(logdirpath, logfilename)
    flog = logging.FileHandler(logfilepath)
    flog.setLevel(logging.DEBUG)
    flog.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s ==> %(message)s'))
    log.addHandler(flog)

    return log


def giveimgls(imgdir, grayscale = True, resize = None):

    imgfilenamels = os.listdir(imgdir)
    imgfilepathls = [os.path.join(imgdir, filename) for filename in imgfilenamels]
    if grayscale is True:
        imgls = [cv2.imread(imgfilepath, cv2.IMREAD_GRAYSCALE) for imgfilepath in imgfilepathls]
    else:
        imgls = [cv2.imread(imgfilepath) for imgfilepath in imgfilepathls]

    if resize is not None:
        imgls = [cv2.resize(img, resize) for img in imgls]

    return np.array(imgls), imgfilepathls


def fillborder(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imgmask = cv2.inRange(hsv, (50, 100, 100), (150, 255, 255))
    h, w = imgmask.shape

    for y in range(h):
        if 255 in imgmask[y]:
            hmin = y
            break
    for y in reversed(range(h)):
        if 255 in imgmask[y]:
            hmax = y
            break
    for x in range(w):
        if 255 in imgmask[:, x]:
            wmin = x
            break
    for x in reversed(range(w)):
        if 255 in imgmask[:, x]:
            wmax = x
            break

    gtmask = np.zeros((h + 2, w + 2), np.uint8)
    cx = int((wmin + wmax) / 2)
    cy = int((hmin + hmax) / 2)

    cv2.floodFill(imgmask, gtmask, (cx, cy), 255);

    resizeimgval = Config.SegImgResolutions[:-1]
    imgmask = cv2.resize(imgmask, resizeimgval)
    gtmask = cv2.resize(gtmask, resizeimgval)

    return imgmask, gtmask
