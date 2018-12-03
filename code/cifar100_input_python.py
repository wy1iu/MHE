import tensorflow as tf
import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_train(train, batch_size, i):
    start_idx = i*batch_size
    end_idx = (i+1)*batch_size
    trx_batch = train['data'][start_idx:end_idx]
    trx_batch = np.asarray(trx_batch, dtype=np.float32)
    trx_batch = np.reshape(trx_batch,(batch_size, 3, 32, 32))
    trx_batch = np.transpose(trx_batch, [0, 2, 3, 1])
    trx_batch_distorted = []
    for j in range(batch_size):
        trx_pad = np.pad(trx_batch[j], [[4,4], [4,4], [0,0]], "symmetric")
        h = np.random.randint(9)
        v = np.random.randint(9)
        trx_crop = trx_pad[h:h+32, v:v+32, :]
        flip = np.random.randint(2)
        if flip == 1:
            trx_flip = np.fliplr(trx_crop)
        else:
            trx_flip = trx_crop
        trx_batch_distorted.append(trx_flip)
    trx_batch_distorted = np.asarray(trx_batch_distorted)
    try_batch = train['fine_labels'][start_idx:end_idx]
    try_batch = np.reshape(try_batch, (batch_size,))
    return trx_batch_distorted,try_batch

def load_test(test, batch_size, i):
    start_idx = i*batch_size
    end_idx = (i+1)*batch_size
    tex_batch = test['data'][start_idx:end_idx]
    tex_batch = np.asarray(tex_batch, dtype=np.float32)
    tex_batch = np.reshape(tex_batch,(batch_size, 3, 32, 32))
    tex_batch = np.transpose(tex_batch, [0, 2, 3, 1])
    tey_batch = test['fine_labels'][start_idx:end_idx]
    tey_batch = np.reshape(tey_batch, (batch_size,))
    return tex_batch,tey_batch

