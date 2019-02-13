import scipy.special
import sys
import math
import numpy as np
import tensorflow as tf

def get_is(is_mean, is_img, inps, splits=10):
  with tf.Session() as sess:
    bs = 200
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        pred = sess.run(is_mean, {is_img: inp})
        preds.append(pred)
    sys.stdout.write("\n")
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)
