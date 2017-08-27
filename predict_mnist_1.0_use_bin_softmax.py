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

import sys
import tensorflow as tf
import tensorflowvisu
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import predict_1 as pre
import numpy as np
def prefictint(imageRawData):
        print("Tensorflow version " + tf.__version__)
        tf.set_random_seed(0)

        # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        # correct answers will go here
        Y_ = tf.placeholder(tf.float32, [None, 10])
        # weights W[784, 10]   784=28*28
        W = tf.Variable(tf.zeros([784, 10]))
        # biases b[10]
        b = tf.Variable(tf.zeros([10]))

        # flatten the images into a single line of pixels
        # -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
        XX = tf.reshape(X, [-1, 784])

        # The model
        Y = tf.nn.softmax(tf.matmul(XX, W) + b)


        # init
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess, "model/model1.0.ckpt")
                #print ("Model restored.")


                #y = tf.nn.softmax(tf.matmul(XX, W) + b)

                prediction = tf.argmax(Y, 1)
                #imageRawData = np.reshape(imageRawData,[28,28,1])
                imagevalue = prediction.eval(feed_dict={X: [imageRawData]}, session=sess)

                #print("权重W：",sess.run(W))
                print("位移b：",sess.run(b))
                #print("图像二进制数据：",imageRawData)

                #直接用训练的方程计算
                imageRawData = tf.reshape(imageRawData, [-1, 784])
                Y = tf.nn.softmax(tf.matmul(imageRawData, W) + b)
                print(np.argmax(sess.run(Y)))

                #print("识别结果：",imagevalue)

        return imagevalue


def main(argv):
        """
        Main function.
        """
        mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
        imageRawData = mnist.test.images[int(argv)]

        #imageRawData = pre.imageprepare(argv)
        imagevalue = prefictint(imageRawData)

        print("识别结果是：",imagevalue)
        print("真实结果是：", np.argmax(mnist.test.labels[int(argv)]))


if __name__ == "__main__":
        main(sys.argv[1])
