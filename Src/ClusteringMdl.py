# Bismillahhirrahmaninrahim

import os

from sklearn.cluster import KMeans
from sklearn.externals import joblib

import Config


# ================================================ Global Cluster Config ===============================================

class ClsMdl():

    def __init__(self, blk):
        self.blk = blk

    def __raiseunrecognisetype(self):
        txt = "Unrecognised Config.ClusterType Setting"
        self.blk.error(txt)
        raise ValueError(txt)

    def loadclustermodel(self, loadpath):
        self.blk.info('Loading the cluster modal from : %s' % loadpath)
        loaddic = joblib.load(loadpath)
        self.clusmdl = loaddic['clustermodal']

        if loaddic['type'] != Config.ClusterType:
            self.blk.warning('Cluster type mismatch between loaded and configurations')
            self.blk.warning('Setting cluster type to: %s' % loaddic['type'])
            Config.ClusterType = loaddic['type']

    def createclustermodal(self):

        if Config.ClusterType == 'KMeans':
            self.blk.debug('KMean cluster have been chosen')
            clusmdl = KMeans(n_clusters=Config.ClsNumberofCluster,
                             n_jobs=Config.ClsNumberofProcess)
            self.clusmdl = clusmdl

        else:
            txt = "Unrecognised Config.ClusterType Setting"
            self.blk.error(txt)
            raise ValueError(txt)

    def fitclustermodel(self, imgls):

        if Config.ClusterType == 'KMeans':
            self.blk.info('Training the cluster modal')
            rimgls = imgls.reshape(-1, Config.ClsImgResolutions[0] * Config.ClsImgResolutions[1])
            self.clusmdl.fit(rimgls)
            self.blk.info('Finish training the cluster modal')
        else:
            self.__raiseunrecognisetype()

    def predictclustermodel(self, imgls):

        if Config.ClusterType == 'KMeans':
            self.blk.info('Predicting the image label')
            rimgls = imgls.reshape(-1, Config.ClsImgResolutions[0] * Config.ClsImgResolutions[1])
            self.imglblls = self.clusmdl.predict(rimgls)
            return self.imglblls
        else:
            self.__raiseunrecognisetype()

    def saveclustermodel(self, addedinfo = None):

        self.blk.info('Saving the cluster modal')
        savepath = os.path.join(Config.LxpExportLxProDDir, 'ClusMdl.pkl')
        self.blk.debug('Save path: %s' % savepath)
        outputdir = {'clustermodal' : self.clusmdl,
                     'type' : Config.ClusterType}
        if addedinfo is not None:
            outputdir['addedinfo'] = addedinfo
        if hasattr(self, 'imglblls'):
            outputdir['imglblls'] = self.imglblls

        joblib.dump(outputdir, savepath)
        self.blk.info('Done saving path: %s' % savepath)
