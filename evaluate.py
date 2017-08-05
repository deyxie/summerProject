# By @Kevin Xu
# kevin28520@gmail.com
# Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
# The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.

# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note:
# it is suggested to restart your kenel to train the model multiple times
# (in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......
# %%

import numpy as np
import tensorflow as tf
import model_tutorial
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import time
import skimage

image_dir = '/Users/deyxie/git/summerProject/test_image/original.tif'


def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''
    wsize = 32
    hwsize = int(wsize / 2)

    image_full = Image.open(image_dir)
    image_full = np.lib.pad(image_full, ((hwsize, hwsize), (hwsize, hwsize), (0, 0)), 'symmetric')

    newfname_class = "/Users/deyxie/git/summerProject/test_image/result.png"
    # outputimage_probs = np.zeros(shape=(image_full.shape[0], image_full.shape[1], 3))  # make the output files where we'll store the data
    outputimage_class = np.zeros(shape=(image_full.shape[0], image_full.shape[1]))

    for rowi in range(hwsize + 1, image_full.shape[0] - hwsize):
        for coli in range(hwsize + 1, image_full.shape[1] - hwsize):
            image = image_full[rowi - hwsize:rowi + hwsize, coli - hwsize:coli + hwsize, :]

            with tf.Graph().as_default():
                BATCH_SIZE = 1
                N_CLASSES = 2
                # image = tf.cast(image, tf.float32)
                # image = tf.image.per_image_standardization(image)
                image = np.float32(image)
                image = np.reshape(image, [1, 32, 32, 3])
                # image = np.float32(image)

                logit = model_tutorial.inference(image, BATCH_SIZE, N_CLASSES)
                logit = tf.nn.softmax(logit)

                x = tf.placeholder(tf.float32, shape=[1, 32, 32, 3])

                # you need to change the directories to yours.
                logs_train_dir = '/Users/deyxie/git/summerProject/logs/train1/'

                saver = tf.train.Saver()


                with tf.Session() as sess:

                    print("Reading checkpoints...")
                    ckpt = tf.train.get_checkpoint_state(logs_train_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        print('Loading success, global_step is %s' % global_step)
                    else:
                        print('No checkpoint file found')

                    prediction = sess.run(logit, feed_dict={x: image})
                    max_index = np.argmax(prediction)
                    print(max_index)
                    if coli % 10 == 0:
                        print("the work is been doing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # get the argmax
                    # outputimage_probs[rowi,hwsize+1:image.shape[1]-hwsize,0:2]=prediction #save the results to our output images
                    outputimage_class[rowi, hwsize + 1:image.shape[1] - hwsize] = max_index
                    # outputimage_probs = outputimage_probs[hwsize:-hwsize, hwsize:-hwsize, :] #remove the edge padding
    outputimage_class = outputimage_class[hwsize:-hwsize, hwsize:-hwsize]

    # scipy.misc.imsave(newfname_prob,outputimage_probs) #save the files
    scipy.misc.imsave(newfname_class, outputimage_class)

evaluate_one_image()
