# Bismillahhirahmannirahim

import os
import cv2
import joblib
import numpy as np
import matplotlib as mpl
from sklearn.externals import joblib

import Config

mpl.use('Agg')
from matplotlib import pyplot as plt

class AnomPlot():

    def __init__(self, blk, dirdict):
        self.blk = blk
        self.dirdict = dirdict

    def plot(self, inputimg, anammat, gtmask, fn):

        imgheight, imgwidth = inputimg.shape
        ammath, ammatw = anammat.shape

        if ammath != imgheight or ammatw != imgwidth:
            self.blk.debug('Resizing the anammat to : %s' % str((imgheight, imgwidth)))
            anammat = cv2.resize(anammat, (imgwidth, imgheight))

        gth, gtw = gtmask.shape
        if gth != imgheight or gtw != imgwidth:
            self.blk.debug('Resizing the gtmask to : %s' % str((imgheight, imgwidth)))
            gtmask = gtmask.astype('float32')
            gtmask = cv2.resize(gtmask, (imgwidth, imgheight))

        maskimgprobmat = np.zeros((imgheight, imgwidth))
        for y in range(imgheight):
            for x in range(imgwidth):
                if gtmask[y][x] > 0:
                    maskimgprobmat[y][x] = anammat[y][x]
                else:
                    maskimgprobmat[y][x] = None

        cutimgprobmat_75 = np.zeros((imgheight, imgwidth))
        for y in range(imgheight):
            for x in range(imgwidth):
                if gtmask[y][x] > 0:
                    probval = anammat[y][x]
                    if probval >= 0.75:
                        cutimgprobmat_75[y][x] = probval
                    else:
                        cutimgprobmat_75[y][x] = None
                else:
                    cutimgprobmat_75[y][x] = None

        cutimgprobmat_25 = np.zeros((imgheight, imgwidth))
        for y in range(imgheight):
            for x in range(imgwidth):
                if gtmask[y][x] > 0:
                    probval = anammat[y][x]
                    if probval >= 0.25:
                        cutimgprobmat_25[y][x] = probval
                    else:
                        cutimgprobmat_25[y][x] = None
                else:
                    cutimgprobmat_25[y][x] = None

        if Config.ApltSaveMeta is True:

            outputdic = {'img': inputimg,
                         'anammat': anammat,
                         'gtmask': gtmask}

            imgname, fnsuffix = os.path.splitext(fn)
            outputpath = os.path.join(self.dirdict['result-meta'], imgname + '.pkl')
            self.blk.info('Save meta-result to : %s' % outputpath)
            joblib.dump(outputdic, outputpath)
            self.blk.debug('Done saving meta-result to : %s' % outputpath)

        self.blk.info('Plotting Image: %s' % fn)
        outputimgpath = os.path.join(self.dirdict['result'], fn)

        title = 'Img: %s ' % fn
        fig, axls = plt.subplots(3, 2, figsize=(Config.ApltSubPltHeight, Config.ApltSubPltWidth),
                                 dpi=Config.ApltFigDpi)
        fig.suptitle(title, fontsize=Config.ApltFontSize)

        ax = axls[0, 0]
        ax.imshow(inputimg, cmap='gray')
        ax.title.set_text('Raw')

        ax = axls[0, 1]
        ax.imshow(gtmask, cmap='gray')
        ax.title.set_text('Mask')

        ax = axls[1, 0]
        clb = ax.imshow(anammat, cmap='Reds')
        plt.colorbar(clb, ax=ax)
        ax.title.set_text('ProbMat')

        ax = axls[1, 1]
        ax.imshow(gtmask, cmap='gray')
        ax.imshow(maskimgprobmat, cmap='Reds')
        ax.title.set_text('Masked_ProbMat')

        ax = axls[2, 0]
        ax.imshow(inputimg, cmap='gray')
        clb = ax.imshow(cutimgprobmat_25, cmap='Reds', vmin=0, vmax=1)
        plt.colorbar(clb, ax=ax)
        ax.title.set_text('Cut-lvl:0.25')

        ax = axls[2, 1]
        ax.imshow(inputimg, cmap='gray')
        clb = ax.imshow(cutimgprobmat_75, cmap='Reds', vmin=0, vmax=1)
        plt.colorbar(clb, ax=ax)
        ax.title.set_text('Cut-lvl:0.75')

        fig.tight_layout()
        fig.subplots_adjust(top=0.93)

        self.blk.debug('Save figure to: %s' % outputimgpath)
        plt.savefig(outputimgpath)
        plt.close()
        self.blk.debug('Done save figure to: %s' % outputimgpath)

        return outputimgpath