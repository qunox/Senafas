# Bismillahhirahmmannirahim

import os
import cv2
import numpy as np

import Config
from Src.Util import fillborder


class AnomEngine():

    def __init__(self, blk, enginedict):
        self.blk = blk
        self.enginedict = enginedict
        self.probmatwidth = self.enginedict['config']['imgwidth']
        self.probmathight = self.enginedict['config']['imgheight']
        if Config.QxeAutoSegmentations is True:
            self.segmodel = enginedict['segmenmodal']
            self.segmodelhight = enginedict['segmenmodal'].input_shape[1]
            self.segmodelwidth = enginedict['segmenmodal'].input_shape[2]
            self.segmodelresize = (-1,  self.segmodelwidth, self.segmodelhight, 1)
        self.binresol = self.enginedict['config']['binresol']

    def analyze(self, inputimgpath):

        inputimg = cv2.imread(inputimgpath, cv2.IMREAD_GRAYSCALE)
        rinputimg = cv2.resize(inputimg, (self.probmatwidth, self.probmathight))
        imgfn = inputimgpath.split('/')[-1:]

        if Config.QxeAutoSegmentations is True:
            seginputimg = cv2.resize(inputimg, (self.segmodelhight, self.segmodelwidth))
            seginputimg = np.reshape(seginputimg, self.segmodelresize)
            predictgtmask = self.segmodel.predict(seginputimg)
            predictgtmask = np.reshape(predictgtmask, (self.segmodelhight, self.segmodelwidth))
            gtmask = predictgtmask.astype('float')
            gtmask = cv2.resize(gtmask, (self.probmathight, self.probmatwidth))
            gtmask = np.where(gtmask > Config.QxeSegmenCutVal, 1 , 0)

        else:
            if not imgfn in os.listdir(self.enginedict['dir']['border']):
                txt = 'Couldnot fine border file for: %s' % imgfn
                self.blk.error(txt)
                raise IOError

            bdrimgpath = os.path.join(self.enginedict['dir']['border'], imgfn)
            bdrimg = cv2.imread(bdrimgpath)
            bdrimg = cv2.resize(bdrimg, (self.probmathight, self.probmatwidth))
            self.blk.debug('Done loading the border img, filling the border')
            imgmask, gtmask = fillborder(bdrimg)
            self.blk.debug('Done filling the border')

        # Determining the cluster label of the input image
        self.blk.debug('Predicting the input img cluster lbl')
        clsimg = cv2.resize(inputimg, Config.ClsImgResolutions)
        imgclslbl = self.enginedict['clustermodal'].predict(clsimg.reshape(1, -1))[0]
        self.blk.debug('Input image cls lbl: %s' % imgclslbl)

        self.blk.debug('Loading the ProbMat')
        if self.enginedict['preload'] is True:
            clsprobmat = self.enginedict['probmat'][imgclslbl]
        else:
            clsprobmatpath = self.enginedict['probmatpath'][imgclslbl]
            clsprobmat = np.load(clsprobmatpath)
        self.blk.debug('Done loading the ProbMat')

        anammat = np.zeros((self.probmathight, self.probmatwidth))
        for y in range(self.probmathight):
            for x in range(self.probmatwidth):
                pxval = rinputimg[y][x]
                binnum = np.digitize(pxval, self.binresol)
                anammat[y][x] = clsprobmat[y][x][binnum]

        self.blk.info('Done constructing the anomaly img: %s' % imgfn)
        return inputimg, anammat, gtmask