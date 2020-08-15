# Bismillahhirahmannirahim
from keras.layers import *
from keras.models import *
from keras.optimizers import *

import Config


class SegMdlArch():

    def __init__(self, blk):

        self.blk = blk
        blk.debug('Segmentation Model Arch class has been activated')
        inputsize = Config.SegImgResolutions
        archtype = Config.SegmentModelType

        if Config.SegDeactiveGPU is True:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if archtype == 'DeepCNN':

            blk.debug('Building DeepCNN')

            inputs = Input(inputsize)
            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            drop2 = Dropout(0.5)(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)

            conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            drop3 = Dropout(0.5)(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)

            conv4 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

            up5 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(pool4))
            conv5 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up5)
            conv5 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

            up6 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv5))
            conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up6)
            conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv6))
            conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up7)
            conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv7))
            conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up8)
            conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            conv9 = Conv2D(1, 1, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv8)

            model = Model(input=inputs, output=conv9)
            blk.debug('Compiling DeepCNN')
            model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        elif archtype == 'Unet':

            blk.debug('Building unet')

            inputs = Input(inputsize)
            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
            conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
            conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
            conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
            drop4 = Dropout(0.5)(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

            conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
            conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
            drop5 = Dropout(0.5)(conv5)

            up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(drop5))
            merge6 = concatenate([drop4, up6], axis=3)
            conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
            conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

            up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv6))
            merge7 = concatenate([conv3, up7], axis=3)
            conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
            conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

            up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv7))
            merge8 = concatenate([conv2, up8], axis=3)
            conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
            conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

            up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
                UpSampling2D(size=(2, 2))(conv8))
            merge9 = concatenate([conv1, up9], axis=3)
            conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
            conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
            conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

            model = Model(input=inputs, output=conv10)
            blk.debug('Compiling unet')
            model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        else:
            txt = 'Unknown Segmentationrchitecture: %s' % archtype
            raise ValueError(txt)

        self.model = model
        blk.debug('Segmentation model has been created')