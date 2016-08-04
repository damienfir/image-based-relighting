# Implementation of Image Based Relighting Using Neural Networks
# from Peiran REN, Yue DONG, Stephen LIN, Xin TONG, Baining GUO, Microsoft Research
# Author: Damien Firmenich, EPFL


import argparse
import math
import matplotlib.pyplot as pl
import numpy as np
from PIL import Image
import tensorflow as tf
import time

import data
import model


def show_image(row, size):
    pl.imshow(np.reshape(row, (size[1], size[0])).T, cmap='gray')
    pl.show()


def write_image(row, size, path):
    im = np.reshape(row, (size[1], size[0])).T
    im = 255 * ((im + 1) / 2)
    Image.fromarray(im.astype("u1")).save(path)


def run_inference(sess, X_placeholder, y_tensor, X):
    """Reconstructs image from given light conditions"""
    feed_dict = {X_placeholder: X}
    return sess.run(y_tensor, feed_dict)


def run_evaluation(sess, loss_op, X_placeholder, yt_placeholder, X):
    X_data, yt_data = data.split_input_output(np.vstack(X))
    feed_dict = {
            X_placeholder: X_data,
            yt_placeholder: yt_data
    }
    loss = sess.run(loss_op, feed_dict)
    print("Evaluation loss:", loss)


def run_training(args):
    with tf.Graph().as_default():

        # data_train, data_validation, im_size = data.load_data_random(args.n_images, im_size=(256,256), light_size=(8,8))
        # data_train, data_validation, im_size = data.load_data_smooth(args.n_images, im_size=(256,256), light_size=(8,8))
        # data_train, data_validation, im_size = data.load_data_grid(args.n_images, im_size=(256,256), light_size=(8,8))
        # data_train, data_validation = data.load_Tgray_mat(args.n_images)
        data_train, data_validation, im_size = data.load_Green_mat(args.n_images)

        X_tensor = tf.placeholder(tf.float32, shape=(None, data.INPUT_DIM), name="input")
        yt_tensor = tf.placeholder(tf.float32, shape=(None, data.OUTPUT_DIM), name="output")

        y_tensor = model.inference(X_tensor, n_units=15, output_dim=data.OUTPUT_DIM)
        loss_tensor = model.loss(y_tensor, yt_tensor)
        error_tensor = model.training_error(loss_tensor, yt_tensor)
        train_op = model.training(loss_tensor, args.learning_rate)
        
        config = tf.ConfigProto(device_count={'GPU': 0})
        if args.gpu: config = tf.ConfigProto()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        sess = tf.Session(config=config)
        sess.run(init)

        # show_image(data_train[0,...,-2], im_size)
        show_image(data_train[0,...,-1], im_size)
        # y_ = run_inference(sess, X_tensor, y_tensor, data_train[0,...,:-1])
        # show_image(y_[:,0], im_size)

        for step in range(args.max_steps):
            X_data, yt_data = data.split_input_output(data.next_batch_images(data_train, args.batch_size))
            # print(X_data.min(axis=0))
            # print(X_data.max(axis=0))
            # print(yt_data.min(axis=0))
            # print(yt_data.max(axis=0))
            feed_dict = {X_tensor: X_data, yt_tensor: yt_data}
            _, loss_value, error = sess.run([train_op, loss_tensor, error_tensor], feed_dict=feed_dict)

            if step % 5 == 0:
                epoch = step*args.batch_size / data_train.shape[0]
                print('Step %d (epoch %.2f): loss = %.2f (error = %.3f)' % (step, epoch, loss_value, error))
                # y_ = run_inference(sess, X, y_tensor, (0.5, 0.5), data.TGRAY_SIZE)
                # show_image(y_[:,0], data.TGRAY_SIZE)

            if (step + 1) % 5 == 0:
                y_ = run_inference(sess, X_tensor, y_tensor, data_train[0,...,:-1])
                # y_ = run_inference(sess, X_tensor, y_tensor, X_data[:im_size[0]*im_size[1]])
                # show_image(y_[:,0], im_size)
                write_image(y_[:,0], im_size, 'results/green-%i.jpg' % step)
                # run_evaluation(sess, loss_tensor, X_tensor, yt_tensor, data_validation)

            # if (step + 1) % 100 == 0 or (step + 1) == args.max_steps:
            #     saver.save(sess, args.train_dir+'/data', global_step=step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('input')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--max_steps', default=200000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--train_dir', default='data')
    parser.add_argument('--n_images', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    run_training(args)

