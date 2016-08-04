import h5py
import matplotlib.pyplot as pl
import numpy as np


INPUT_DIM = 4
OUTPUT_DIM = 1

TGRAY_FILENAME = "Tgray_80x80.mat"
TGRAY_SIZE = (256, 256)
GREEN_FILENAME = "Green.mat"
GREEN_SIZE = (464, 696)
GREEN_LIGHT = (32, 32)


def split_training_validation(data, s=0.7):
    idx = np.arange(data.shape[0])
    split = round(s*data.shape[0])
    return data[idx[:split]], data[idx[split:]]


def split_input_output(data):
    return data[:, :INPUT_DIM], data[:, INPUT_DIM:]


def colmaj_grid(shape, idx=None):
    if idx is not None:
        x, y = np.unravel_index(idx, (shape[1], shape[0]))
    else:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    y = y.T.reshape((-1,1)) / shape[0]
    x = x.T.reshape((-1,1)) / shape[1]
    return np.hstack([x, y])


def im_light_grid(sample_size, im_size, light_size):
    im_h, im_w = im_size
    light_h, light_w = light_size
    px = colmaj_grid(im_size)
    light_idx = np.sort(np.random.permutation(np.arange(light_h*light_w))[:sample_size])
    light = colmaj_grid(light_size, light_idx)
    return px, light


def concat_images(im_v, light, px):
    images = []
    im_avg = im_v.mean(axis=1)[:, np.newaxis]
    for i in range(light.shape[0]):
        l = np.tile([light[i,:]], (px.shape[0], 1))
        v = im_v[:,i]
        # Xi = np.hstack([px, l, im_avg, v[:, np.newaxis]])
        # Xi = np.hstack([px, l, im_avg, im_avg])
        Xi = np.hstack([px, l, v[:, np.newaxis]])
        images.append(Xi[np.newaxis])
    return np.concatenate(images, axis=0)


def make_input(LTM_mat, im_size, light_size, sample_size):
    light_idx = np.sort(np.random.permutation(np.arange(light_size[0]*light_size[1]))[:sample_size])
    light = colmaj_grid(light_size, light_idx)
    px = colmaj_grid(im_size)
    im_v = np.zeros((light.shape[0], px.shape[0]), dtype='float32')
    LTM_mat.read_direct(im_v, np.s_[light_idx, 0:px.shape[0]], np.s_[0:im_v.shape[0], 0:im_v.shape[1]])
    return concat_images(im_v.T, light, px)


def crop_image(X, crop, size):
    px = X[0,:,:2]
    x = px[:,0] * size[1]
    y = px[:,1] * size[0]
    crop_idx = (x >= crop[0]) * (x < crop[0]+crop[2]) * (y >= crop[1]) * (y < crop[1]+crop[3])
    return X[:, crop_idx]


def load_Green_mat(sample_size):
    print("loading", GREEN_FILENAME)
    mat = h5py.File(GREEN_FILENAME)
    LTM = mat['LTM']
    X = make_input(LTM, GREEN_SIZE, GREEN_LIGHT, sample_size)
    X[...,-1] /= X[...,-1].max()
    X[...,-1] = (X[...,-1] / 2**16) * 2 - 1
    # X[...,-2] = (X[...,-2] / 2**16)
    im_size = GREEN_SIZE
    crop = (400, 350, 50, 50)
    X = crop_image(X, crop, im_size)
    im_size = crop[2:]
    return split_training_validation(X) + (im_size,)


def load_Tgray_mat(sample_size):
    print("loading", TGRAY_FILENAME)
    mat = h5py.File(TGRAY_FILENAME)
    LTM = mat['Tfull']
    light_h, light_w = 80, 80
    im_h, im_w = TGRAY_SIZE
    X = make_input(LTM, im_h, im_w, light_h, light_w, sample_size)
    X[...,-1] = scale_center(X[...,-1])
    X[...,-2] = scale_center(X[...,-2])
    return split_training_validation(X)


def load_data_random(sample_size, im_size, light_size):
    px, light = im_light_grid(sample_size, im_size, light_size)
    im_v = np.random.rand(px.shape[0], sample_size)
    X = concat_images(im_v, light, px)
    return split_training_validation(X) + (im_size,)


def load_data_smooth(sample_size, im_size, light_size):
    px, light = im_light_grid(sample_size, im_size, light_size)
    im_v = np.tile(np.linspace(0, 1, im_size[1]), (im_size[0], 1)).T.flatten()[:, np.newaxis]
    X = concat_images(im_v, light, px)
    return split_training_validation(X) + (im_size,)


def load_data_grid(sample_size, im_size, light_size):
    px, light = im_light_grid(sample_size, im_size, light_size)
    ones = np.ones((round(im_size[0]/2), round(im_size[1]/2)))
    im_v = np.hstack([ones, 1-ones])
    im_v = np.vstack([im_v, 1-im_v]).T.flatten()[:, np.newaxis] * 2 - 1
    X = concat_images(im_v, light, px)
    return split_training_validation(X) + (im_size,)


def next_batch(data, batch_size, step):
    """Cycles through the data linearly (not random)"""
    n = data.shape[0]
    start = round(step * batch_size) % n
    end = (start + batch_size) % n
    if end < start:
        sample = np.vstack([data[start:], data[:end]])
    else:
        sample = data[start:end]
    return sample


def next_batch_images(data, batch_size):
    """Cycles through the data per images randomly"""
    im_idx = np.random.permutation(np.arange(data.shape[0]))[:batch_size]
    return np.vstack(data[im_idx])

