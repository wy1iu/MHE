import timeit

import os
import numpy as np
import tensorflow as tf

from libs.input_helper import Cifar10
from libs.utils import save_images, mkdir
from net import DCGANGenerator, SNDCGAN_Discrminator
import pickle as pickle
import inspect
from tf_inception import get_is

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64, '')
flags.DEFINE_integer('max_iter', 200000, '')
flags.DEFINE_integer('snapshot_interval', 1000, 'interval of snapshot')
flags.DEFINE_integer('evaluation_interval', 5000, 'interval of evalution')
flags.DEFINE_integer('display_interval', 100, 'interval of displaying log to console')
flags.DEFINE_float('adam_alpha', 0.0001, 'learning rate')
flags.DEFINE_float('adam_beta1', 0.5, 'beta1 in Adam')
flags.DEFINE_float('adam_beta2', 0.999, 'beta2 in Adam')
flags.DEFINE_integer('n_dis', 1, 'n discrminator train')

mkdir('tmp')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
INCEPTION_FILENAME = 'inception_score.pkl'
config = FLAGS.__flags

config = {k: FLAGS[k]._value for k in FLAGS}
generator = DCGANGenerator(**config)
discriminator = SNDCGAN_Discrminator(**config)
data_set = Cifar10(batch_size=FLAGS.batch_size)

global_step = tf.Variable(0, name="global_step", trainable=False)
increase_global_step = global_step.assign(global_step + 1)
is_training = tf.placeholder(tf.bool, shape=())
z = tf.placeholder(tf.float32, shape=[None, generator.generate_noise().shape[1]])
x_hat = generator(z, is_training=is_training)
x = tf.placeholder(tf.float32, shape=x_hat.shape)

d_fake = discriminator(x_hat, update_collection=None)
# Don't need to collect on the second call, put NO_OPS
d_real = discriminator(x, update_collection="NO_OPS")
# Softplus at the end as in the official code of author at chainer-gan-lib github repository
d_mhe_loss = tf.constant(0.0)
d_mhe_loss_list = tf.get_collection("d_mhe_loss")
if len(d_mhe_loss_list) > 0:
  d_mhe_loss += tf.add_n(d_mhe_loss_list)
g_mhe_loss = tf.constant(0.0)
g_mhe_loss_list = tf.get_collection("g_mhe_loss")
if len(g_mhe_loss_list) > 0:
  g_mhe_loss += tf.add_n(g_mhe_loss_list)
d_loss = tf.reduce_mean(tf.nn.softplus(d_fake) + tf.nn.softplus(-d_real))
g_loss = tf.reduce_mean(tf.nn.softplus(-d_fake))
d_loss_summary_op = tf.summary.scalar('d_loss', d_loss)
g_loss_summary_op = tf.summary.scalar('g_loss', g_loss)
d_mhe_loss_summary_op = tf.summary.scalar('d_mhe_loss', d_mhe_loss)
g_mhe_loss_summary_op = tf.summary.scalar('g_mhe_loss', g_mhe_loss)
merged_summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('snapshots')

d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.adam_alpha, beta1=FLAGS.adam_beta1, beta2=FLAGS.adam_beta2)
d_gvs = optimizer.compute_gradients(d_loss + d_mhe_loss, var_list=d_vars)
g_gvs = optimizer.compute_gradients(g_loss + g_mhe_loss, var_list=g_vars)
d_solver = optimizer.apply_gradients(d_gvs)
g_solver = optimizer.apply_gradients(g_gvs)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
best_saver = tf.train.Saver()


np.random.seed(1337)
sample_noise = generator.generate_noise()
np.random.seed()
iteration = sess.run(global_step)


is_img = tf.placeholder(tf.float32, shape=[None, None, None, 3])
is_feat = tf.nn.softmax(tf.contrib.gan.eval.run_inception(tf.image.resize_bilinear(is_img, [299, 299])))


if os.path.exists('./snapshots/best_model.ckpt.index'):
    saver.restore(sess, './snapshots/best_model.ckpt')
    sample_images = sess.run(x_hat, feed_dict={z: sample_noise, is_training: False})
    save_images(sample_images, 'tmp/{:06d}.png'.format(iteration))
    # Sample 50000 images for evaluation
    print("Evaluating...")
    num_images_to_eval = 50000
    eval_images = []
    num_batches = num_images_to_eval // FLAGS.batch_size + 1
    print("Calculating Inception Score. Sampling {} images...".format(num_images_to_eval))
    np.random.seed(0)
    for _ in range(num_batches):
      images = sess.run(x_hat, feed_dict={z: generator.generate_noise(), is_training: False})
      eval_images.append(images)
    np.random.seed()
    eval_images = np.vstack(eval_images)
    eval_images = eval_images[:num_images_to_eval]

    eval_images = np.clip(eval_images, -1.0, 1.0)
    inception_score_mean, inception_score_std = get_is(is_feat, is_img, eval_images)
    print("Inception Score: Mean = {} \tStd = {}.".format(inception_score_mean, inception_score_std))

    best_inception = inception_score_mean