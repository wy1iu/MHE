import tensorflow as tf
import numpy as np
import math

class VGG():
    def get_conv_filter(self, shape, reg, stddev):
        init = tf.random_normal_initializer(stddev=stddev)
        if reg:
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            filt = tf.get_variable('filter', shape, initializer=init,regularizer=regu)
        else:
            filt = tf.get_variable('filter', shape, initializer=init)

        return filt      

    def get_bias(self, dim, init_bias, name):
        with tf.variable_scope(name):
            init = tf.constant_initializer(init_bias)
            regu = tf.contrib.layers.l2_regularizer(self.wd)
            bias = tf.get_variable('bias', dim, initializer=init, regularizer=regu)

            return bias

    def batch_norm(self, x, n_out, phase_train):
        """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        with tf.variable_scope('bn'):

            gamma = self.get_bias(n_out, 1.0, 'gamma')
            beta = self.get_bias(n_out, 0.0, 'beta')

            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.999)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
            padding='SAME', name=name)

    def _add_thomson_constraint(self, filt, n_filt, model, power):
        filt = tf.reshape(filt, [-1, n_filt])
        if model =='half_mhe':
            filt_neg = filt*-1
            filt = tf.concat((filt,filt_neg), axis=1)
            n_filt *= 2
        filt_norm = tf.sqrt(tf.reduce_sum(filt*filt, [0], keep_dims=True) + 1e-4)
        norm_mat = tf.matmul(tf.transpose(filt_norm), filt_norm)
        inner_pro = tf.matmul(tf.transpose(filt), filt)
        inner_pro /= norm_mat

        if power =='0':
            cross_terms = 2.0 - 2.0 * inner_pro
            final = -tf.log(cross_terms + tf.diag([1.0] * n_filt))
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 1 * tf.reduce_sum(final) / cnt
        elif power =='1':
            cross_terms = (2.0 - 2.0 * inner_pro + tf.diag([1.0] * n_filt))
            final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-0.5))
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 1 * tf.reduce_sum(final) / cnt
        elif power =='2':
            cross_terms = (2.0 - 2.0 * inner_pro + tf.diag([1.0] * n_filt))
            final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-1))
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 1* tf.reduce_sum(final) / cnt
        elif power =='a0':
            acos = tf.acos(inner_pro)/math.pi
            acos += 1e-4
            final = -tf.log(acos)
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 1* tf.reduce_sum(final) / cnt
        elif power =='a1':
            acos = tf.acos(inner_pro)/math.pi
            acos += 1e-4
            final = tf.pow(acos, tf.ones_like(acos) * (-1))
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 1e-1 * tf.reduce_sum(final) / cnt
        elif power =='a2':
            acos = tf.acos(inner_pro)/math.pi
            acos += 1e-4
            final = tf.pow(acos, tf.ones_like(acos) * (-2))
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 1e-1 * tf.reduce_sum(final) / cnt

        tf.add_to_collection('thomson_loss', loss)

    def _add_thomson_constraint_final(self, filt, n_filt, power):
        filt = tf.reshape(filt, [-1, n_filt])
        filt_norm = tf.sqrt(tf.reduce_sum(filt*filt, [0], keep_dims=True) + 1e-4)
        norm_mat = tf.matmul(tf.transpose(filt_norm), filt_norm)
        inner_pro = tf.matmul(tf.transpose(filt), filt)
        inner_pro /= norm_mat

        if power =='0':
            cross_terms = 2.0 - 2.0 * inner_pro
            final = -tf.log(cross_terms + tf.diag([1.0] * n_filt))
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 10 * tf.reduce_sum(final) / cnt
        elif power =='1':
            cross_terms = (2.0 - 2.0 * inner_pro + tf.diag([1.0] * n_filt))
            final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-0.5))
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 10 * tf.reduce_sum(final) / cnt
        elif power =='2':
            cross_terms = (2.0 - 2.0 * inner_pro + tf.diag([1.0] * n_filt))
            final = tf.pow(cross_terms, tf.ones_like(cross_terms) * (-1))
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 10 * tf.reduce_sum(final) / cnt
        elif power =='a0':
            acos = tf.acos(inner_pro)/math.pi
            acos += 1e-4
            final = -tf.log(acos)
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 10 * tf.reduce_sum(final) / cnt
        elif power =='a1':
            acos = tf.acos(inner_pro)/math.pi
            acos += 1e-4
            final = tf.pow(acos, tf.ones_like(acos) * (-1))
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 1 * tf.reduce_sum(final) / cnt
        elif power =='a2':
            acos = tf.acos(inner_pro)/math.pi
            acos += 1e-4
            final = tf.pow(acos, tf.ones_like(acos) * (-2))
            final -= tf.matrix_band_part(final, -1, 0)
            cnt = n_filt * (n_filt - 1) / 2.0
            loss = 1 * tf.reduce_sum(final) / cnt

        tf.add_to_collection('thomson_final', loss)

    def _conv_layer(self, bottom, ksize, n_filt, is_training, name, stride=1, 
        pad='SAME', relu=False, reg=True, bn=True, model='baseline', power='0',  final=False):

        with tf.variable_scope(name) as scope:
            n_input = bottom.get_shape().as_list()[3]
            shape = [ksize, ksize, n_input, n_filt]
            print("shape of filter %s: %s" % (name, str(shape)))

            filt = self.get_conv_filter(shape, reg, stddev=tf.sqrt(2.0/tf.to_float(ksize*ksize*n_input)))
            if model == 'mhe' or model =='half_mhe':
                if final: self._add_thomson_constraint_final(filt, n_filt, power)
                else: self._add_thomson_constraint(filt, n_filt, model, power)

            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding=pad)

            if bn:
                conv = self.batch_norm(conv, n_filt, is_training)
                
            if relu:
                return tf.nn.relu(conv)
            else:
                return conv

    def build(self, rgb, n_class, is_training, model_name, power_s):
        self.wd = 5e-4

        feat = (rgb - 127.5) / 128.0

        ksize = 3
        n_layer = 3

        # 32X32
        n_out = 64
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, is_training, name="conv1_" + str(i), bn=True, relu=True,
                                    pad='SAME',  reg=True, model=model_name, power=power_s)
        feat = self._max_pool(feat, 'pool1')

        # 16X16
        n_out = 128
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, is_training, name="conv2_" + str(i), bn=True, relu=True,
                                    pad='SAME',  reg=True, model=model_name, power=power_s)
        feat = self._max_pool(feat, 'pool2')

        # 8X8
        n_out = 256
        for i in range(n_layer):
            feat = self._conv_layer(feat, ksize, n_out, is_training, name="conv3_" + str(i), bn=True, relu=True,
                                    pad='SAME',  reg=True, model=model_name, power=power_s)
        feat = self._max_pool(feat, 'pool3')

        self.fc6 = self._conv_layer(feat, 4, 256, is_training, "fc6", bn=False, relu=False, pad='VALID',
                                    reg=True, model=model_name, power=power_s)

        self.score = self._conv_layer(self.fc6, 1, n_class, is_training, "score", bn=False, relu=False, pad='VALID',
                                      reg=True,  model=model_name, power=power_s, final=True)

        self.pred = tf.squeeze(tf.argmax(self.score, axis=3))

