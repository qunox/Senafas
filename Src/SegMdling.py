# Bismillahhhirahmmannirahim

import os
from multiprocessing import Pool

import cv2
import matplotlib as mpl
import numpy as np
from sklearn.externals import joblib

import Config
from Src import SegMdlArch
from Src.Util import fillborder

mpl.use('Agg')
from matplotlib import pyplot as plt
from keras.models import load_model, save_model
from sklearn.model_selection import train_test_split

# ================================================ Global Cluster Config ===============================================
GlobalCongfig = {
    'epoch': 25,
    'validationsplit': 0.2
}

class SegmenMdl():

    def __init__(self, blk):
        self.blk = blk
        self.outputpath = os.path.join(Config.LxpExportLxProDDir, 'SegmentationOutput')
        self.primeimgdic = {}
        if not os.path.exists(self.outputpath):
            self.blk.debug('Creating dir: %s' % self.outputpath)
            os.mkdir(self.outputpath)
        if Config.SegLoadModel is True:
            self.model = load_model(Config.SegLoadModel)
        else:
            self.model = SegMdlArch.SegMdlArch(blk).model
        blk.debug(self.model.summary())

    def trainmodel(self):

        self.blk.info('Loading and pre-processing data')
        # checking the contain of the normal and border img
        rawfnls = os.listdir(Config.SegRawImgDir)
        bdrfnls = os.listdir(Config.SegBdrImgDir)

        if len(rawfnls) != len(bdrfnls):
            txt = 'Number file contain in Config.SegNrmImgDir is not equal to Config.SegBdrImgDir'
            self.blk.error(txt)
            raise ValueError(txt)

        rawfnls.sort()
        bdrfnls.sort()
        acceptimgfnls = []
        for nfn, bfn in zip(rawfnls, bdrfnls):
            if nfn != bfn:
                txt = 'File mismatch in Config.SegNrmImgDir and Config.SegBdrImgDir, file: %s, %s' % (nfn, bfn)
                self.blk.error(txt)
                raise ValueError(txt)
            else:
                _, ext = os.path.splitext(nfn)
                if ext in Config.OpsAcceptedImgFormat:
                    acceptimgfnls.append(nfn)
                else:
                    self.blk.debug('File %s is not rejected' % nfn)

        self.blk.info("Number of img accepted for segmentation: %s" % len(acceptimgfnls))
        self.primeimgdic['imgfilenamels'] = acceptimgfnls

        # Creating the normal img matrix
        rawimgls = []
        resizeimgval = Config.SegImgResolutions[:-1]
        for fl in acceptimgfnls:
            filepath = os.path.join(Config.SegRawImgDir, fl)
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, resizeimgval)
            rawimgls.append(img)
            self.blk.debug('Raw img accepted: %s' % filepath)
        self.primeimgdic['rawimgls'] = rawimgls

        # Creating border images
        bdrimgls = []
        for fl in acceptimgfnls:
            filepath = os.path.join(Config.SegBdrImgDir, fl)
            img = cv2.imread(filepath)
            bdrimgls.append(img)
            self.blk.debug('Border img accepted: %s' % filepath)
        self.primeimgdic['borderimgls'] = bdrimgls

        self.blk.debug('Creating process pool')
        pool = Pool(Config.SegProcessNum)
        self.blk.debug('Pool mapping')
        resls = pool.map(fillborder, bdrimgls)
        self.blk.debug('Done pool mapping')
        pool.close()

        imgmaskls = []
        gtmaskls = []

        for imgmask, gtmask in resls:
            imgmaskls.append(imgmask)
            gtmaskls.append(gtmask)
        self.primeimgdic['gtimgls'] = gtmaskls
        self.primeimgdic['imgmaskls'] = imgmaskls
        self.blk.debug('Done creating ground truth imgs')

        trainimgls, testimgls, trainmaskls, testmaskls = train_test_split(self.primeimgdic['rawimgls'],
                                                                          self.primeimgdic['imgmaskls'],
                                                                          test_size=GlobalCongfig['validationsplit'])
        datals = [trainimgls, testimgls, trainmaskls, testmaskls]
        lblls = ['train-img', 'test-img', 'train-mask', 'test-mask']
        rdatals = []
        inputtuple = (-1,) + Config.SegImgResolutions
        for ls, lbl in zip(datals, lblls):
            als = np.reshape(ls, inputtuple)
            self.blk.debug(str(als.shape) + ' ' + lbl)
            rdatals.append(als)
        trainimgls, testimgls, trainmaskls, testmaskl = rdatals

        self.blk.info('Training the model')
        self.model.fit(trainimgls,
                       trainmaskls,
                       epochs=GlobalCongfig['epoch'],
                       validation_data=(testimgls, testmaskl))
        self.blk.info('Done training the model')

        if Config.SegTestModel is True:
            testimgls = np.reshape(self.primeimgdic['rawimgls'], inputtuple)
            resimgls = self.model.predict(testimgls)
            resimgls = np.reshape(resimgls, inputtuple[:-1])
            self.primeimgdic['testgtimgls'] = resimgls

    def saveeverything(self):

        self.blk.info('Saving the segmentation model')
        savepath = os.path.join(Config.LxpExportLxProDDir, 'segmodel.h5')
        self.blk.debug('Saving to: %s' % savepath)
        save_model(self.model, savepath)
        self.blk.info('Saving the prime image dict')
        self.primeimgdic['config'] = GlobalCongfig
        savepath = os.path.join(self.outputpath, 'primeimgdict.pkl')
        self.blk.debug('Saving to: %s' % savepath)
        joblib.dump(self.primeimgdic, savepath)

        self.blk.info('Printing test images')
        if Config.SegPrintTestImg is True:
            printdir = os.path.join(self.outputpath, 'testimgresult')
            os.mkdir(printdir)
            for imgindex in range(len(self.primeimgdic['imgfilenamels'])):

                imgname = self.primeimgdic['imgfilenamels'][imgindex]
                rawimg = self.primeimgdic['rawimgls'][imgindex]
                bdrimg = self.primeimgdic['borderimgls'][imgindex]
                gtimg = self.primeimgdic['imgmaskls'][imgindex]
                testgtimg = self.primeimgdic['testgtimgls'][imgindex]

                imgls = [rawimg, bdrimg, gtimg, testgtimg]
                ptindex = 1
                fig = plt.figure(figsize=(10,10), dpi=80)
                for img in imgls:
                    ax = fig.add_subplot(2,2,ptindex)
                    ptindex += 1
                    ax.imshow(img, cmap='gray')
                    ax.set_axis_off()
                fig.suptitle(imgname)
                figpath = os.path.join(printdir, imgname)
                fig.savefig(figpath)
                self.blk.debug('Figure saved: %s' % figpath)
                plt.close()
        self.blk.info('Done printing test images')