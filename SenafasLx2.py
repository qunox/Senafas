## Bismillahhrirahmannirahim
import os

import Config
from Src import Util
from Src.ClusteringMdl import ClsMdl
from Src.ProbMat import ProbMat
from Src.SegMdling import SegmenMdl

# =================================================== MAIN =============================================================
blk = Util.givebalak('Lx2.log')

# Welcoming remarks and banner printing
blk.info('\n\nBismillahhirahmannirahim')
blk.info('Starting SenafasLx2 Production')
blk.debug('Print banner')
Util.printbanner(type='lx2')

currentpath = os.getcwd()
lxproddir = os.path.join(currentpath, 'LxProD')
Config.LxpExportLxProDDir = os.path.join(lxproddir, Config.LxpExportLxProDName)
blk.info('Checking output directory at: %s' % Config.LxpExportLxProDDir)

if not os.path.exists(Config.LxpExportLxProDDir):
    os.mkdir(Config.LxpExportLxProDDir)
    blk.info('New directory has been created: %s' % Config.LxpExportLxProDDir)
else:
    blk.info('WARNING: directory %s already exist, old file could be overwrite' % Config.LxpExportLxProDDir)

if Config.LxpSegmentModelCreation is True:
    segmenmdlobj = SegmenMdl(blk)
    segmenmdlobj.trainmodel()
    segmenmdlobj.saveeverything()

if Config.LxpClusterModelCreation is True:
    # Creating or loading the normal image clustering modal
    blk.info('Loading normal images from dir: %s' % Config.ClsNormalImagesDirPath)
    imgls, imgpathls = Util.giveimgls(Config.ClsNormalImagesDirPath, resize=Config.ClsNormalImagesResolutions)
    blk.debug('Total number of images loaded: %s' % len(imgls))

    clsmdl = ClsMdl(blk)
    if Config.ClsLoadCluster is True:
        blk.info('Loading normal image clustering modal from: %s' % Config.ClsLoadClusterPath)
        clsmdl.loadclustermodel(Config.ClsLoadClusterPath)
    elif Config.ClsLoadCluster is False:
        blk.info('Creating normal image clustering modal')
        clsmdl.createclustermodal()
        clsmdl.fitclustermodel(imgls)
        clsmdl.saveclustermodel(addedinfo={'imgpathls' : imgpathls})
    else:
        txt = "Unrecognised Config.LoadCluster Setting"
        blk.error(txt)
        raise ValueError(txt)

    blk.info('Predicting normal image labels: %s' % len(imgls))
    nrmimgcluslblls = clsmdl.predictclustermodel(imgls)
    blk.info('Done predicting normal image labels: %s' % len(imgls))

if Config.LxpProbMatrixCreation is True:
    # Creating the prob matrix
    blk.info('Creating the normal image probability matrix')
    probmatobj = ProbMat(blk)
    probmatdic = probmatobj.create(nrmimgcluslblls, save=True, returnprobmat=False)

blk.info('End of execution')
blk.info('Exiting')
