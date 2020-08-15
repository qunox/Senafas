from datetime import datetime
from multiprocessing import cpu_count


# LxProD Configurations
LxpImportLxProDPath = ''
LxpExportLxProDName = 'Ppr_LxProD_%s' % datetime.now().strftime('%Y%m%d%H%M%S')
LxpSegmentModelCreation = False
LxpClusterModelCreation = True
LxpProbMatrixCreation = True
LxpExportLxProDDir = None # Do not change

# QxEngine Configurations
QxeUseRedis = False
QxeInputDir = '/casawaridisk/qunox/ML/Senafas2/QxInput'
QxeImportLxProDDirPath = '/casawaridisk/qunox/ML/Senafas2/LxProD/Ppr_LxProD_20200510213112'
QxePreloadProbMat = False
QxeOutputDirName = 'QxOutput_%s' % datetime.now().strftime('%Y%m%d%H%M%S')
QxeOutputDir = None # Do not change
QxeAutoSegmentations = True
QxeImgResolutions = (1024, 1024)
QxeSegmenCutVal = 0.55
QxeMoveInputFile = True
QxeSeaveAnomMatrix = True

# UxInterface Configuration
UxHost = '0.0.0.0'
UxPort = 8610
UxProcessedImgDir = '/casawaridisk/qunox/ML/Senafas2/static/processedimg'


# Segmentation Modelling Configurations
SegBdrImgDir = '/casawaridisk/Share/Dataset/CXR/SenafasLx/Adult_PA_M_Border'
SegRawImgDir = '/casawaridisk/Share/Dataset/CXR/SenafasLx/Adult_PA_M_Raw'
SegmentModelType = 'DeepCNN' # available DeepCNN, un
SegLoadModel = False
SegLoadModelPath = ''
SegImgResolutions = (256, 256, 1)
SegProcessNum = cpu_count() - 2
SegDeactiveGPU = False
SegTestModel = True
SegPrintTestImg = True

# Cluster Configurations
ClsNormalImagesDirPath = '/casawaridisk/Share/Dataset/Senafas/NIH/Normal_Img'
ClsNumberofCluster = 5
ClsNumberofProcess = cpu_count() - 2
ClsNormalImagesResolutions = (256, 256)
ClsLoadCluster = False
ClsLoadClusterPath = ''
ClsImgToBeClusterType = 'img' # imgls or maskls or insmaskls
ClusterType = 'KMeans'
ClsImgResolutions = (256, 256)

# ProbMat Configurations
ProbBinCount = 150
ProbBinResolution = 260
ProbImgResolutions = (512, 512)
ProbNeighbourDist = 5
ProbWindowLength = 105
ProbPolyOrder = 1
ProbProcessNumber = 4 # Keep this low to avoid memory error
ProbScoringSys = 'Exponential Ratio'
ProbScaledProb = True
ProbNoisyWorker = True

# AnomEngine Configurations
AecMinPixVal = 50
AecProcessCount = cpu_count() - 2

# AnaomPlot Configuratoins
ApltSaveMeta = True # Do not change
ApltFigSize = (8,8)
ApltFigDpi = 200
ApltSubPltHeight = 10
ApltSubPltWidth = 10
ApltAxisOff = False
ApltFontSize = 12

# Operational Configurations
OpsAcceptedImgFormat = ['.PNG', '.png', '.JPG', '.JPEG', '.jpg', '.jpeg']
OpsQxeRefreshTime = 5
