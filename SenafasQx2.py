# Bismillahhirahmannirahim

import os
import sys
import time
import shutil
import joblib
import ntpath
import numpy as np

import Config
from Src import Util
from Src.AnomEngine import AnomEngine
from Src.AnomPlot import AnomPlot

blk = Util.givebalak('Qx2.log')


# =================================================== FUNC =============================================================


def movefile(src, dest):
    blk.info('Transferring file from %s to %s' % (src, dest))
    shutil.move(src, dest)


def checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        blk.info('New directory has been created: %s' % path)
    else:
        blk.info('WARNING: directory %s already exist, old file could be overwrite' % path)


# =================================================== MAIN =============================================================

# Welcoming remarks and banner printing
blk.info('\n\nBismillahhirahmannirahim')
blk.info('Starting SenafasQx2 Engine')

blk.debug('Print banner')
Util.printbanner(type='qx2')

currentpath = os.getcwd()
lxproddir = os.path.join(currentpath, 'QxOutput')
Config.QxeOutputDir = os.path.join(lxproddir, Config.QxeOutputDirName)
blk.info('Checking output directory at: %s' % Config.QxeOutputDir)

checkdir(Config.QxeOutputDir)
if Config.QxeSeaveAnomMatrix is True:
    anommatoutputpath = os.path.join(Config.QxeOutputDir, 'AnomMat')
    checkdir(anommatoutputpath)

# Pre-Loading the engine to check memory space
enginedict = {}
blk.info('Constructing the engine')

blk.debug('Loading the cluster model')
clustermodelpath = os.path.join(Config.QxeImportLxProDDirPath, 'ClusMdl.pkl')
blk.debug('Cluster model path: %s' % clustermodelpath)
clusterdt = joblib.load(clustermodelpath)
blk.debug('Done loading the cluster modal data')
enginedict['clustermodal'] = clusterdt['clustermodal']
blk.info('Done loading the cluster modal')

blk.debug('Loading the ProbMat')
probmatpath = os.path.join(Config.QxeImportLxProDDirPath, 'ProbMat')
blk.debug('ProbMat path: %s' % probmatpath)
# check if probmat is finish:
if not os.path.exists(os.path.join(probmatpath, 'done.pkl')):
    blk.warning('PROB-MAT is NOT FINISH')
enginedict['config'] = joblib.load(os.path.join(probmatpath, 'config.pkl'))
if Config.QxePreloadProbMat is True:
    blk.info('Preloading the ProbMat')
    enginedict['preload'] = True
    enginedict['probmat'] = {}
    for clslbl in enginedict['config']['clslblls']:
        clsprobmatpath = os.path.join(probmatpath, str(clslbl) + '.npy')
        clsprobmat = np.load(clsprobmatpath)
        enginedict['probmat'][clslbl] = clsprobmat
    blk.info('Finish preloading the ProbMat')
else:
    enginedict['preload'] = False
    enginedict['probmatpath'] = {}
    for clslbl in enginedict['config']['clslblls']:
        clsprobmatpath = os.path.join(probmatpath, str(clslbl) + '.npy')
        enginedict['probmatpath'][clslbl] = clsprobmatpath

blk.debug('Done loading the ProbMat data')

dirtocheckls = []
if Config.QxeAutoSegmentations is True:
    from keras.models import load_model

    blk.debug('Loading the SegmenModal')
    segmenmodalpath = os.path.join(Config.QxeImportLxProDDirPath, 'segmodel.h5')
    blk.debug('SegmenModal path: %s' % segmenmodalpath)
    enginedict['segmenmodal'] = load_model(segmenmodalpath)
    blk.info('Done loading the SegmenModal')
else:
    bdrdirpath = os.path.join(Config.QxeInputDir, 'BorderImg')
    dirtocheckls.append(bdrdirpath)

inputdirpath = os.path.join(Config.QxeInputDir, 'InputImg')
donedirpath = os.path.join(Config.QxeOutputDir, 'Processed')
resultdirpath = os.path.join(Config.QxeOutputDir, 'Result')
resmetadirpath = os.path.join(Config.QxeOutputDir, 'Result-meta')
faildirpath = os.path.join(Config.QxeOutputDir, 'Failed')
dirtocheckls.extend([Config.QxeOutputDir, donedirpath, faildirpath, resultdirpath, resmetadirpath])

enginedict['dir'] = {'inputimg': inputdirpath,
                     'output': Config.QxeOutputDir,
                     'result': resultdirpath,
                     'result-meta': resmetadirpath,
                     'fail': faildirpath}
if Config.QxeAutoSegmentations is False:
    enginedict['dir']['border'] = bdrdirpath

for path in dirtocheckls:
    if not os.path.exists(path):
        blk.info('Creating the following directory: %s' % path)
        os.mkdir(path)

blk.info('Starting the engine')
anomengineobj = AnomEngine(blk, enginedict)
anamplotobj = AnomPlot(blk, enginedict['dir'])
blk.info('>>> SENAFAS QxENGINE IS READY <<<')

while True:
    # Checking if there's any new file
    inputfnls = os.listdir(inputdirpath)

    acceptedfilepath = []
    if len(inputfnls) > 0:
        blk.info('Found %s file in input directory' % len(inputfnls))
        acceptedfilels = []
        for fn in inputfnls:
            blk.debug('File found: %s' % fn)
            _, ext = os.path.splitext(fn)
            if ext in Config.OpsAcceptedImgFormat:
                acceptedfilels.append(fn)
            else:
                blk.info('File %s is rejected' % fn)
        if Config.QxeUseRedis is True:
            acceptedfilepath = acceptedfilels
        else:
            for fn in acceptedfilels:
                inputimgpath = os.path.join(inputdirpath, fn)
                acceptedfilepath.append(inputimgpath)

    if len(acceptedfilepath) > 0:
        for inputimgpath in acceptedfilepath:
            blk.info('Processing file: %s' % inputimgpath)
            try:
                # Constructing the anamoly matrix
                inputimg, anammat, gtmask = anomengineobj.analyze(inputimgpath)
                if Config.QxeSeaveAnomMatrix is True:
                    fn = ntpath.basename(inputimgpath)
                    imgid, _ = os.path.splitext(fn)
                    anomimgfn = '%s.npy' % imgid
                    anammatpath = os.path.join(anommatoutputpath, anomimgfn)
                    blk.info('Saving anom matrix: %s' % anomimgfn)
                    np.save(anammatpath, anammat)

                # Plotting the anamoly
                blk.info('Plotting the calculated anomaly')
                fn = inputimgpath.split('/')[-1:][0]
                outputimgpath = anamplotobj.plot(inputimg, anammat, gtmask, fn)

                dest = os.path.join(donedirpath, fn)
                if Config.QxeMoveInputFile is True:
                    movefile(inputimgpath, dest)
                blk.info('Done processing img: %s' % fn)

                if Config.QxeUseRedis is True:
                    blk.info('Sending finish img path to Redis')

            except ValueError:
                blk.error('Failed to process img: %s' % fn )
                e = sys.exc_info()[0]
                blk.error(e)
                dest = os.path.join(faildirpath, fn)
                if Config.QxeMoveInputFile is True:
                    movefile(inputimgpath, dest)


    blk.debug('Going to sleep for: %s' % Config.OpsQxeRefreshTime)
    time.sleep(Config.OpsQxeRefreshTime)
