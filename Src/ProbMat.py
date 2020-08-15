# Bismillahhirahmannirahim

import os
from multiprocessing import Pool

import joblib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler

import Config
from Src import Util

def __workerjob__(packet):

    # Unpacking:
    blk = Util.givebalak('Lx2.log')
    clsnrmimgstack = packet['clsnrmimgstack']
    binsedge = packet['binsedge']
    windowval = packet['windowval']
    polyoder = packet['polyoder']
    ndist = packet['ndist']
    binresolution = packet['binresol']
    clslbl = packet['clslbl']
    effbincout = packet['effbincout']
    noisy = packet['noisy']
    save = packet['save']
    probmatdumppath = packet['probmatdumppath']
    fluxmatdumppath = packet['fluxmatdumppath']
    returnprobmat = packet['returnprobmat']
    scaledval = packet['scaleval']
    pid = os.getpid()

    if noisy:
        blk.info('Finish initiliazing worker, pid: %s clslbl: %s' % (pid, clslbl))

    blk.debug('Creating probmat for clslbl: %s' % clslbl)
    imghieghts, imgwidth, samplenum = clsnrmimgstack.shape
    maximghieghts, maximgwidth = (imghieghts - 1, imgwidth -1)
    scaler = MinMaxScaler()

    def givepxnbrstack(y,x):
        xls = range(x - ndist, x + ndist)
        yls = range(y - ndist, y + ndist)
        pxstack = []
        for yi in yls:
            for xi in xls:
                if xi < 0 or xi > maximgwidth or yi < 0 or yi > maximghieghts:
                    continue
                nstack = clsnrmimgstack[yi][xi].tolist()
                pxstack.extend(nstack)
        return np.array(pxstack)

    def expratioscore(val):
        return np.exp(-val / samplenum)

    def givestackprobcurve(pxstack):

        hiscountls, _ = np.histogram(pxstack, binsedge)
        pifunc = CubicSpline(binsedge[:effbincout], hiscountls)
        pxprobls = pifunc(binresolution)
        pxprobls = savgol_filter(pxprobls, windowval, polyoder)
        pxprobls = np.array([val if val > 0 else 0 for val in pxprobls])
        scorels = expratioscore(pxprobls)

        medianval = (max(scorels) - min(scorels)) / 2 + min(scorels)
        varls = [medianval - val for val in scorels]
        varstd = np.log(np.std(varls))

        if scaledval is True:
            scorels = scaler.fit_transform(scorels.reshape(-1, 1)).reshape(-1)

        return scorels, varstd

    clsprobmatdict = {}
    clsprobmatdict['clslabel'] = clslbl
    probmat = np.zeros((imghieghts, imgwidth, len(binresolution)))
    varmat = np.zeros((imghieghts, imgwidth))
    for y in range(imghieghts):
        for x in range(imgwidth):
            npxstack = givepxnbrstack(y,x)
            scorels , varstd = givestackprobcurve(npxstack)
            probmat[y][x] = scorels
            varmat[y][x] = varstd
        if noisy:
            blk.info('Pid: %s Clslbl: %s Done Y=%s' % (pid, clslbl, y))

    blk.debug('Finish creating probmat for clslbl: %s' % clslbl)

    if save is True:
        blk.debug('Saving clslbl: %s probmat' % clslbl)
        np.save(probmatdumppath, probmat)
        blk.debug('Finish saving clslbl: %s probmat' % clslbl)
        blk.debug('Saving clslbl: %s fluxmat' % clslbl)
        np.save(fluxmatdumppath, varmat)
        blk.debug('Finish saving clslbl: %s fluxmat' % clslbl)

    if returnprobmat is True:
        return clsprobmatdict
    else:
        return None


class ProbMat():

    def __init__(self, blk):
        self.blk = blk

    def create(self, clusterlabells, save=True, returnprobmat = False):

        self.blk.debug('Reading imgs from %s' % Config.ClsNormalImagesDirPath)
        nrmimgls, _ = Util.giveimgls(Config.ClsNormalImagesDirPath, resize=Config.ProbImgResolutions)
        self.blk.debug('Done reading imgs')

        clusterlblset = set(clusterlabells)
        nrmimgdict = {}
        for clslbl in clusterlblset:
            nrmimgdict[clslbl] = []

        for img, imgclslbl in zip(nrmimgls, clusterlabells):
            nrmimgdict[imgclslbl].append(img)

        # measuring the sample cout for each cluster
        clssamplecountls = [len(nrmimgdict[clslbl]) for clslbl in clusterlblset]

        binsedge = np.linspace(0, 256, Config.ProbBinCount)
        binresol = np.linspace(0, 256, Config.ProbBinResolution)
        if save is True:
            probmatdumpdir = os.path.join(Config.LxpExportLxProDDir, 'ProbMat')
            if not os.path.exists(probmatdumpdir):
                self.blk.debug('Creating new directory: %s' % probmatdumpdir)
                os.mkdir(probmatdumpdir)

            fluxmatdumpdir = os.path.join(Config.LxpExportLxProDDir, 'FluxMat')
            if not os.path.exists(fluxmatdumpdir):
                self.blk.debug('Creating new directory: %s' % probmatdumpdir)
                os.mkdir(fluxmatdumpdir)

        self.blk.debug('Creating the packet list')
        packetdictls = []
        clsfnamels = []
        for clslbl in clusterlblset:
            self.blk.debug('Clslbl: %s Number of images: %s' % (clslbl, len(nrmimgdict[clslbl])))

            packetdict = {
                'clsnrmimgstack' : np.dstack(nrmimgdict[clslbl]),
                'binsedge' : binsedge,
                'binresol' : binresol,
                'effbincout': Config.ProbBinCount - 1,
                'windowval' : Config.ProbWindowLength,
                'polyoder' : Config.ProbPolyOrder,
                'ndist' : Config.ProbNeighbourDist,
                'clslbl': clslbl,
                'noisy': Config.ProbNoisyWorker,
                'save' : save,
                'scaleval' : Config.ProbScaledProb,
                'returnprobmat': returnprobmat
            }

            if save is True:
                clsdumpfname = '%s.npy' % clslbl
                packetdict['probmatdumppath'] = os.path.join(probmatdumpdir, clsdumpfname)
                clsfnamels.append(clsdumpfname)

                clsdumpfname = '%s.npy' % clslbl
                packetdict['fluxmatdumppath'] = os.path.join(fluxmatdumpdir, clsdumpfname)
                clsfnamels.append(clsdumpfname)

            packetdictls.append(packetdict)

        self.blk.debug('Finish creating the packet list')

        probmatconfigdic = {
            'imgwidth' : Config.ProbImgResolutions[1],
            'imgheight' : Config.ProbImgResolutions[0],
            'binsedge' : binsedge,
            'binresol' : binresol,
            'clslblls' : clusterlblset,
            'clsfnamels' : clsfnamels,
            'clssamplecount': clssamplecountls,
            'scaledprobval' : Config.ProbScaledProb,
            'scoretype' : Config.ProbScoringSys
        }
        self.blk.debug('Dumping the probmat config file')
        joblib.dump(probmatconfigdic, os.path.join(probmatdumpdir, 'config.pkl'))
        self.blk.debug('Finish dumping the probmat config file')

        self.blk.debug('Creating the process pool, num = %s' % Config.ProbProcessNumber)
        pool = Pool(processes=Config.ProbProcessNumber)
        self.blk.info('Begin ProbMat creation')
        clsprobmatdictls = pool.map(__workerjob__, packetdictls)
        pool.close()
        donedict = {'Done':True}
        donepath = os.path.join(probmatdumpdir, 'done.pkl')
        joblib.dump(donedict, donepath)

        self.blk.info('Finish ProbMat creation')

        if returnprobmat is True:
            probmatdict = {}
            probmatdict['configure'] = probmatconfigdic
            probmatdict['probmat'] = {}
            for clsprobmatdict in clsprobmatdictls:
                clslbl = clsprobmatdict['clslabel']
                probmatdict['probmat'][clslbl] = clsprobmatdict

            return probmatdict

        else:
            return None
