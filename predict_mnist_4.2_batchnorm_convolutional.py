# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import tensorflowvisu
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import sys
import predict_1 as pre
import numpy as np


def prefictint(imageRawData):
    print("Tensorflow version " + tf.__version__)
    tf.set_random_seed(0.0)

    # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    # correct answers will go here
    Y_ = tf.placeholder(tf.float32, [None, 10])
    # variable learning rate
    lr = tf.placeholder(tf.float32)
    # test flag for batch norm
    tst = tf.placeholder(tf.bool)
    iter = tf.placeholder(tf.int32)
    # dropout probability
    pkeep = tf.placeholder(tf.float32)
    pkeep_conv = tf.placeholder(tf.float32)

    def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999,
                                                           iteration)  # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        return Ylogits, tf.no_op()

    def compatible_convolutional_noise_shape(Y):
        noiseshape = tf.shape(Y)
        noiseshape = noiseshape * tf.constant([1, 0, 0, 1]) + tf.constant([0, 1, 1, 0])
        return noiseshape

    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 10 softmax neurons)
    K = 24  # first convolutional layer output depth
    L = 48  # second convolutional layer output depth
    M = 64  # third convolutional layer
    N = 200  # fully connected layer

    W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

    W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

    # The model
    # batch norm scaling is not useful with relus
    # batch norm offsets are used instead of biases
    stride = 1  # output is 28x28
    Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
    Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
    Y1r = tf.nn.relu(Y1bn)
    Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))
    stride = 2  # output is 14x14
    Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
    Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
    Y2r = tf.nn.relu(Y2bn)
    Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))
    stride = 2  # output is 7x7
    Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
    Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
    Y3r = tf.nn.relu(Y3bn)
    Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

    Y4l = tf.matmul(YY, W4)
    Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
    Y4r = tf.nn.relu(Y4bn)
    Y4 = tf.nn.dropout(Y4r, pkeep)
    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits)

    update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)

    # init
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, "model/model4.2.ckpt")
        # print ("Model restored.")

        # imageRawData = mnist.test.images[2]
        # y = tf.nn.softmax(tf.matmul(XX, W) + b)

        prediction = tf.argmax(Y, 1)
        imageRawData = np.reshape(imageRawData, [28, 28, 1])
        imagevalue = prediction.eval(feed_dict={X: [imageRawData],tst: False, pkeep: 1.0, pkeep_conv: 1.0}, session=sess)

        #print("权重W：", sess.run(W1))
        #print("位移b：", sess.run(B5))
        print("图像二进制数据：",imageRawData)

        print("识别结果：", imagevalue)

    return imagevalue


def main(argv):
    """
    Main function.
    """
    imageRawData = pre.imageprepare(argv)
    imagevalue = prefictint(imageRawData)

    print("识别结果是：", imagevalue)


if __name__ == "__main__":
    main(sys.argv[1])