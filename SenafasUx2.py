# Bismillahhirrahmanirahim

import os
import cv2
import json
import redis
import time
import numpy as np

# CONFIGURE

inputpath = ''
outputpath = ''
errorpath = ''
acceptedfiletypels = ['.PNG', '.png', '.JPG', '.JPEG', '.jpg', '.jpeg']

hostname = 'awang'
port = 8610
refreshrate = 1

# FUNC

def fp(s):
    print('==> %s' % s)

def imgtojsonstr(img):
    h, w = img.shape
    imgvec = img.reshape(-1).tolist()
    imgstr = ','.join(map(str, imgvec))
    imgdict = {'img': imgstr, 'h': h, 'w': w }
    return json.dump(imgdict)

def backtoimg(imgdictstr):
    imgdict = json.loads(imgdictstr)
    strmat = imgdict['img']
    matheight = imgdict['h']
    matweidth = imgdict['w']
    fmat = strmat.split(',')
    img = np.array(fmat, dtype=int).reshape((matheight, matweidth))
    return img

# MAIN

fp('Starting SenafasUx2, Redis Version')
fp('Checking allpaths:')
for path in [inputpath, outputpath, errorpath]:
    if os.path.exists(path):
        fp('Path exist: %s' % path)
    else:
        fp('Creating path: %s' % path)
        os.mkdir(path)

fp('Connecting to redis server')
redobj = redis.Redis(
    host=hostname,
    port=port,
    decode_responses=True
)

while True:

    filels = os.listdir(inputpath)

    if len(filels) > 0:
        fp('Number of file found in inputpat: %s' % len(filels))
        acceptedfilels = []
        for fn in filels:
            _, ext = os.path.splitext(fn)
            if ext in acceptedfiletypels:
                acceptedfilels.append(fn)
            else:
                fp('File %s is rejected' % fn)
        fp('Number of file accepted: %s' % len(acceptedfilels))

        if len(acceptedfilels) > 0:
            acceptedfilepathls = [os.path.join(inputpath, fn) for fn in acceptedfilels]
            imgls = [cv2.imread(inputimgpath, cv2.IMREAD_GRAYSCALE) for inputimgpath in acceptedfilepathls]
            jsonstrls = [imgtojsonstr(img) for img in imgls]

            for jsonstr, fn in zip(jsonstrls, acceptedfilels):
                fp('Uploading img : %s' % fn)
                redobj.put('status', 'put')
                redobj.put('task', jsonstr)

                stat = True
                while stat:
                    status = redobj.get('status')
                    if status == 'get':
                        stat = False
                        returnimgjson = redobj.get('returnimg')
                        fp('Done retriving img : %s' % fn)
                    else:
                        time.sleep(1)

                returnimg = backtoimg(returnimgjson)
                outputimgpath = os.path.join(outputpath, fn)
                fp('Return img path: %s' % outputimgpath)
                cv2.imwrite(outputimgpath, returnimg)
                fp('Done processing img : %s' % fn)

    else:
        time.sleep(refreshrate)