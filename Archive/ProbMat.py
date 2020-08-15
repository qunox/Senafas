# Bismillahhirahmannirahim

import os
from multiprocessing import Pool

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

import Config
from Src import Util

# ================================================ Global Cluster Config ===============================================

probmatdic = {'config' : {},
              'salgovmat': {},
              'stack': {},
              'samplecount': {}}

def __givesavgolls__(packet):
    x, y, clslbl = packet

    maximgwidth = probmatdic['config']['maximgwidth']
    maximghights = probmatdic['config']['maximghights']
    binsedge = probmatdic['config']['binsedge']
    windowval = probmatdic['config']['windowval']
    polyoder = probmatdic['config']['polyoder']
    ndist = probmatdic['config']['ndist']
    binresolution = probmatdic['config']['binresol']
    clsimgstack = probmatdic['stack'][clslbl]
    effbincout = probmatdic['config']['effectivebincount']

    xls = range(x - ndist, x + ndist)
    yls = range(y - ndist, y + ndist)

    stack = []

    for yi in yls:
        for xi in xls:
            if xi < 0 or xi > maximgwidth or yi < 0 or yi > maximghights:
                continue
            nstack = clsimgstack[yi][xi].tolist()
            stack.extend(nstack)
    hiscountls, _ = np.histogram(stack, binsedge)
    pifunc = CubicSpline(binsedge[:effbincout], hiscountls)
    pxprobls = pifunc(binresolution)
    pxprobls = np.array([val if val > 0 else 0 for val in pxprobls])
    pxprobls = MinMaxScaler().fit_transform(pxprobls.reshape(-1, 1)).reshape(-1)
    pxprobls = savgol_filter(pxprobls, windowval, polyoder)
    pxprobls = [1 - val for val in pxprobls]
    return x, y, pxprobls

class ProbMat:

    def __init__(self, blk):
        self.blk = blk

    def giveProbMatDic(self, clusterlabells):
        nrmimgls, _ = Util.giveimgls(Config.ClsNormalImagesDirPath, resize=Config.ProbImgResolutions)
        imgwidth, imgheight = Config.ProbImgResolutions

        probmatdic['config']['imgwidth'] = imgwidth
        probmatdic['config']['imgheight'] = imgheight
        probmatdic['config']['maximgwidth'] = imgwidth - 1
        probmatdic['config']['maximghights'] = imgheight - 1
        probmatdic['config']['bincount'] = Config.ProbBinCount
        probmatdic['config']['effectivebincount'] = Config.ProbBinCount - 1
        probmatdic['config']['binsedge'] = np.linspace(0, 260, Config.ProbBinCount)
        probmatdic['config']['binresol'] = np.linspace(0, 260, Config.ProbBinResolution)
        probmatdic['config']['windowval'] = Config.ProbWindowLength
        probmatdic['config']['polyoder'] = Config.ProbPolyOrder
        probmatdic['config']['ndist'] = Config.ProbNeighbourDist

        clusterlblset = set(clusterlabells)
        stackimgdict = {}
        for clslbl in clusterlblset:
            stackimgdict[clslbl] = []

        for img, imgclslbl in zip(nrmimgls, clusterlabells):
            stackimgdict[imgclslbl].append(img)

        for clslbl in clusterlblset:
            probmatdic['stack'][clslbl] = np.dstack(stackimgdict[clslbl])
            self.blk.debug('Stack subcluster-image shape: %s' % str(probmatdic['stack'][clslbl].shape))
            probmatdic['samplecount'][clslbl] = len(stackimgdict[clslbl])

        for clslbl in clusterlblset:
            posls = []
            for x in range(imgwidth):
                for y in range(imgheight):
                    posls.append((x, y,clslbl))
            np.random.shuffle(posls)

            pool = Pool(Config.ProbProcessNumber)
            self.blk.info('Begin creating ProbMat for cluster: %s' % clslbl)
            resls = pool.map(__givesavgolls__, posls)
            pool.close()

            savgoldict = {}
            for y in range(imgheight):
                savgoldict[y] = {}
                for x in range(imgwidth):
                    savgoldict[y][x] = {}

            for res in resls:
                x, y, savgolls = res
                savgoldict[y][x] = savgolls

            probmatdic['salgovmat'][clslbl] = savgoldict
            probmatdic['samplecount'][clslbl] = len(probmatdic['stack'][clslbl])
            self.blk.info('Done creating ProbMat for cluster: %s' % clslbl)

        return probmatdic

    def saveprobmatdic(self):

        self.blk.info('Saving the Probability Matrix')
        savepath = os.path.join(Config.LxpExportLxProDDir, 'ProbMatDic.pkl')
        self.blk.debug('Save path: %s' % savepath)
        joblib.dump(probmatdic, savepath)
        self.blk.info('Done saving the Probability Matrix')
