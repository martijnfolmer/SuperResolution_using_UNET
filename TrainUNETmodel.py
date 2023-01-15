from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from CreateUNETmodel import CreateUNET
'''
    This script trains a UNET model for the purposes of superresolution, meaning taking a small image, scaling it up,
    and then using machine learning to increase its resolution. 
    
    This model and training could also be used for other applications wherein the input and output of a model
    are both and image, such as denoising or semantic segmentations.
    
    Author :        Martijn Folmer 
    Date created :  13-01-2023

'''


# USER VARIABLES
pathToFolderWithImages = ''      # absolute path to the folder containing the images of your dataset. make sure these images are larger than img_size_big, otherwise you will get blurry images.

img_size_small = (64, 64)               # The size of small images that we wish to scale up from
img_size_big = (256, 256)               # the size of big images that we want to output
offset = 0.0                            # The offset and scale to preprocess the images (i.e. img = img[:,:,:] + offset) * scale)
scale = 1.0

batch_size = 8                          # the batch size during training
num_epochs = 1                        # number of epochs we train

load_model = False                      # if set to true, we want to load a previous model to train on
load_model_path = ''                    # the model we load, in case load_model = False

pathToResultingImg = 'resulting_img'            # where we save the results of our training
pathToResultingModel = 'resulting_model'        # where we save our model when finished training
tflite_name = 'UNET.tflite'                     # the name of the saved tflite

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Create the datagenerator for runtime.
class DataGenerator(keras.utils.Sequence):

    def __init__(self, all_filenames,  batch_size=8, img_size_small=(64, 64), img_size_big=(256, 256),
                 offset=0.0, scale=1.0 / 255.0, shuffle_index=True):

        self.X_filenames = all_filenames  # the paths to the images we use as trainingdata

        self.batch_size = batch_size  # the size of the batch we want to return

        self.offset = offset  # offset and scale are used to normalize the hand around a certain point
        self.scale = scale

        self.indices = np.arange(self.X_filenames.shape[0])  # used to randomize the order of the images
        self.shuffle_index = shuffle_index  # bool whether we shuffle the indices list or not

        self.imgSize_small = img_size_small  # the size of the small images on which we wish to train
        self.imgSize_big = img_size_big      # the size of the input images, which we use to multiply coordinates

    def LoadImages(self, batchxFilenames):
        '''

        :param batchxFilenames:  The absolute paths to the image files we want to augment/train with
        :return: A numpy array containing all loaded images
        '''
        allImg = [cv2.imread(filename) for filename in batchxFilenames]
        return allImg


    def randomlyFlip(self, allImg):
        """

        :param allImg: The images we wish to augment
        :return: allImg, but with roughly 50% of the images flipped horizontally
        """
        for i_img, img_c in enumerate(allImg):
            if random.random() < 0.5:
                allImg[i_img] = cv2.flip(img_c, 1)
        return allImg

    def setOffsetAndScale(self, allImg):
        """

        :param allImg: The images we wish to augment
        :return: allImg, but with the desired offset and scale
        """
        allImg = [(img[:, :, :] + self.offset) * self.scale for img in allImg]
        return allImg

    def __getitem__(self, idx):
        '''

        :param idx: The identifier of which batch we are at.
        :return: an numpy array with all images (allImg), and numpy arrays with the outputs (visibility, handedness
        and coordinates)
        '''

        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X_filenames = self.X_filenames[inds]

        # load all the images
        allImg = self.LoadImages(batch_X_filenames)
        allImg = self.randomlyFlip(allImg)      # randomly flip the images

        # this is our output, which is the original, unedited image from the celebrity dataset
        # allBigImg = [img[20:-20] for img in allImg]
        allBigImg = [img for img in allImg]
        allBigImg = [cv2.resize(img, self.imgSize_big) for img in allBigImg]

        # This is our input, which is a pixelated version of the big image
        allSmallImg = [cv2.resize(img, self.imgSize_small) for img in allBigImg]
        allSmallImg = [cv2.resize(img, self.imgSize_big) for img in allSmallImg]

        # Alter the image pixelvalues with desired offset and scale
        allBigImg = self.setOffsetAndScale(allBigImg)
        allSmallImg = self.setOffsetAndScale(allSmallImg)

        # returning the input and output
        return np.asarray(allSmallImg), np.asarray(allBigImg)

    def __len__(self):
        return (np.ceil(len(self.X_filenames) / float(self.batch_size))).astype(np.int)

    def on_epoch_end(self):
        if self.shuffle_index:
            np.random.shuffle(self.indices)


# get all of the images
X_filenames = np.asarray([pathToFolderWithImages + "/" + f for f in os.listdir(pathToFolderWithImages)])
X_train, X_test = train_test_split(X_filenames, test_size=0.2, random_state=76)

# create the datagenerators
TrainDataGen = DataGenerator(all_filenames=X_filenames, batch_size=batch_size, offset=offset, scale=scale,
                             img_size_small=img_size_small, img_size_big=img_size_big, shuffle_index=True)
TestDataGen = DataGenerator(all_filenames=X_filenames, batch_size=batch_size, offset=offset, scale=scale,
                            img_size_small=img_size_small, img_size_big=img_size_big, shuffle_index=True)


# load previous model
if load_model:
    model = keras.models.load_model(load_model_path)
    model.summary()
# build model if we don't load
else:
    CU = CreateUNET(input_shape=(img_size_big[0], img_size_big[1], 3))
    model = CU.get_model()

# model compilation
model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['acc'])

# fit the model and Save the model
hist = model.fit(TrainDataGen, epochs=num_epochs, verbose=1, validation_data=TestDataGen, shuffle=True)

# save the model in designated folder
if not os.path.exists(pathToResultingModel):
    os.mkdir(pathToResultingModel)
model.save(pathToResultingModel)


# Remove all of our resulting image
if not os.path.exists(pathToResultingImg):
    os.mkdir(pathToResultingImg)
for file in [f'{pathToResultingImg}/{filename}' for filename in os.listdir(pathToResultingImg)]:
    os.remove(file)


# Running an image through the trained model
def RunModel(trained_model, imgPath, imgSizeSmall, imgSizeBig, img_offset, img_scale):
    """

    :param trained_model: The trained model
    :param imgPath: The path to the image we want to run
    :param imgSizeSmall: The size of the input
    :param imgSizeBig: The size of the output
    :param img_offset: The offset to add to pixelvalues of the image
    :param img_scale: The scale to multiply the pixelvalues of the image with
    :return: The GroundTruth, input and output images after running the model
    """
    img_groundtruth = cv2.imread(imgPath)
    img_groundtruth = cv2.resize(img_groundtruth, imgSizeBig)

    img_input = img_groundtruth.copy()
    img_input = cv2.resize(img_input, imgSizeSmall)
    img_input = cv2.resize(img_input, imgSizeBig)
    img_input_show = img_input.copy()

    # create the input, and use the model to predict the output
    img_input = (img_input + img_offset) * img_scale
    img_input = np.reshape(img_input, (1, imgSizeBig[0], imgSizeBig[1], 3))
    img_output = trained_model.predict(img_input)

    # Postprocess our output
    img_output = np.reshape(img_output, (imgSizeBig[0], imgSizeBig[1], 3))
    img_output = img_output[:, :, :] / img_scale - img_offset
    img_output = np.clip(img_output, 0.0, 255.0)
    img_output = np.asarray(img_output, dtype=np.uint8)

    # concatenate all of them
    img_tot = np.concatenate([img_groundtruth, img_input_show, img_output], axis=1)

    return img_tot


# doing some predictions and test and training data
for ir in range(min(50, X_test.shape[0])):
    print(f"We are at {ir}")
    img_tot = RunModel(model, X_test[ir], img_size_small, img_size_big, offset, scale)
    cv2.imwrite(f'{pathToResultingImg}/test_img_{ir}.png', img_tot)


for ir in range(min(50, X_train.shape[0])):
    print(f"We are at {ir}")
    img_tot = RunModel(model, X_train[ir], img_size_small, img_size_big, offset, scale)
    cv2.imwrite(f'{pathToResultingImg}/train_img_{ir}.png', img_tot)


# Plot the history of the training
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(f'{pathToResultingImg}/loss.png')

# Compiling as tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model=model)  # path to model
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_quant_model = converter.convert()  # actually convert the model
# Save the model.
with open(tflite_name, 'wb') as f:
    f.write(tflite_quant_model)
f.close()
