import os
import time
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from ops import *
import json
from tqdm import tqdm


class MTTG(object):
    def __init__(self, sess, img_size=256, batch_size=10, c_dim=3,
                 checkpoint_path=None, sample_path=None, data_path=None):

        self.data_path = data_path
        self.outfit_path = self.data_path + 'outfits/'
        self.feature_path = self.data_path + 'items/'
        self.checkpoint_path = checkpoint_path
        self.sample_path = sample_path

        self.data = json.load(open(self.data_path + 'outfit.json', 'r'))
        self.item_data = json.load(open(self.data_path + 'item.json', 'r'))
        self.dictionary = json.load(open(self.data_path + 'dictionary.json', 'r'))
        self.text_dim = len(self.dictionary)

        self.is_training = True
        self.drop = 0.5
        self.sess = sess
        self.batch_size = batch_size
        self.img_size = img_size
        self.c_dim = c_dim
        self.max_num = 4

    def processing(self):
        self.text_dict, self.imgArr_dict = {}, {}
        print('processing data...')
        for item_id in tqdm(self.item_data):
            ## text
            item_desc = self.item_data[item_id]['desc'].replace('.', '').split()
            wordFea = np.zeros([self.text_dim], dtype=np.float32)
            for word in item_desc:
                if word in self.dictionary:
                    wordFea[self.dictionary.index(word)] = 1
            self.text_dict[item_id] = wordFea
            ## array
            item_array = np.load(self.data_path + 'item_array/' + item_id + '.npy')
            self.imgArr_dict[item_id] = item_array

    def get_variable(self, type, shape, stddev=0.01, name=None):
        if type == 'W':
            var = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
            tf.add_to_collection('regular_losses', tf.contrib.layers.l2_regularizer(0.005)(var))
            return var
        elif type == 'b':
            var = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=tf.zeros_initializer())
            return var

    def mttg(self, item_img=None, item_num=None, item_text=None, reuse=False):
        with tf.variable_scope('mttg') as scope:
            if reuse:
                scope.reuse_variables()

            vi = []
            for item_i in tf.split(item_img, item_num):
                pad = [self.max_num-tf.shape(item_i)[0], self.img_size, self.img_size, self.c_dim]
                item_i = tf.concat([item_i, tf.zeros(pad)], axis=0)
                item_i = tf.transpose(item_i, [1, 2, 0, 3])
                item_i = tf.reshape(item_i, [1, self.img_size, self.img_size, self.c_dim*self.max_num])
                vi.append(item_i)
                
            vi = tf.concat(vi, axis=0)
            visual_code, visual_info = self.visual_encoder(vi)
            text_code, text_info = self.textual_encoder(item_text)
            Pv = self.visual_decoder(visual_code, visual_info)
            Pt = self.visual_decoder(text_info[-1], text_info, reuse=True)

            return visual_code, visual_info, text_code, text_info, Pv, Pt

    def process(self, outfit_id, data):
        try:
            P = np.array([load_image(self.outfit_path + outfit_i.replace('-0', '') + '.jpg') for outfit_i in outfit_id])
        except:
            P = []
        item_num = []
        item_img = []
        item_text = []
        for outfit_i in outfit_id:
            try:
                data_dict = data[outfit_i]['clo'][:4]
            except:
                data_dict = data[outfit_i][:4]
            item_num.append(min(4, len(data_dict)))
            text_i = np.zeros([self.text_dim * self.max_num])
            for idx, item_i in enumerate(data_dict):
                text_i[idx * self.text_dim:(idx + 1) * self.text_dim] = self.text_dict[item_i]
                item_img.append(self.imgArr_dict[item_i])
            item_text.append(text_i)
        return P, item_img, item_num, item_text

    def train(self):
        self.processing()
        self.item_img = tf.placeholder(shape=[None, self.img_size, self.img_size, self.c_dim], dtype=tf.float32, name='item_img')
        self.P = tf.placeholder(shape=[None, self.img_size, self.img_size, self.c_dim], dtype=tf.float32, name='outfit')
        self.item_num = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name='item_num')
        self.item_text = tf.placeholder(shape=[None, self.text_dim * self.max_num], dtype=tf.float32, name='item_text')

        visual_code, visual_info, text_code, text_info, Pv, Pt = self.mttg(self.item_img, self.item_num, self.item_text)

        self.P_ = self.P
        self.Pv_ = Pv
        self.Pt_ = Pt

        self.L_Gv = tf.reduce_mean(tf.abs(self.P - Pv))
        self.L_Gt = tf.reduce_mean(tf.square(visual_info[0] - text_info[0])) \
                         + tf.reduce_mean(tf.square(visual_info[1] - text_info[1])) \
                         + tf.reduce_mean(tf.square(visual_info[2] - text_info[2])) \
                         + tf.reduce_mean(tf.square(visual_info[3] - text_info[3])) \
                         + tf.reduce_mean(tf.square(visual_info[4] - text_info[4])) \
                         + tf.reduce_mean(tf.square(visual_info[5] - text_info[5])) \
                         + tf.reduce_mean(tf.square(visual_info[6] - text_info[6])) \
                         + tf.reduce_mean(tf.square(visual_info[7] - text_info[7]))

        self.Gt_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mttg/textual')
        self.Gv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mttg/visual')

        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(0.0002, self.global_step, decay_steps=1000, decay_rate=0.9, staircase=False)
        self.Gv_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.L_Gv, var_list=self.Gv_vars)
        self.Gt_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(self.L_Gt, var_list=self.Gt_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.saver = tf.train.Saver(max_to_keep=1)
        # ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        # self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        train_sample = json.load(open(self.data_path + '/data/train_idx.json', 'r'))

        for step in range(20001):
            start_time = time.time()
            batch_outfits = random.sample(train_sample, self.batch_size)
            P, item_img, item_num, item_text = self.process(batch_outfits, self.data)

            _, _, L_Gt, L_Gv = self.sess.run([self.Gv_optim, self.Gt_optim, self.L_Gt, self.L_Gv],
                                                    {self.P: P, self.item_img: item_img,
                                                     self.item_num: item_num, self.item_text: item_text,
                                                     self.is_training: True, self.drop: 0.5})
            print('Step: [%d] | Time: %.2f | Gt_loss %.3f, Gv_loss %.3f|'
                  %(step, time.time()-start_time, L_Gt, L_Gv))

            if np.mod(step, 100) == 0:
                self.sample_model(self.sample_path, step)
                # self.save(self.checkpoint_path, step)

    def save(self, checkpoint_path, step):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def sample_model(self, sample_path, step):
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        fig = plt.figure()
        fig.set_size_inches(9, int(np.ceil(self.batch_size/3)))
        img_count = 1
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.01)

        all_test_outfits = json.load(open(self.data_path+'data/test_idx.json', 'r'))[:self.batch_size]
        batch_test_outfits = [all_test_outfits[i:i+self.batch_size] for i in range(0, len(all_test_outfits), self.batch_size)]
        for idxi, test_outfits in enumerate(batch_test_outfits):
            P, item_img, item_num, item_text = self.process(test_outfits, self.data)
            Pv, Pt = self.sess.run([self.Pv_, self.Pt_], {self.item_img: item_img, self.item_num: item_num, self.item_text: item_text,
                                                          self.is_training: False, self.drop: 1})
            test_item_nums = []
            init = 0
            for i in item_num:
                test_item_nums.append(init+i)
                init=init+i
            outfit_pieces = np.split(item_img, test_item_nums[:-1])

            for idx, i in enumerate(outfit_pieces):
                fig.add_subplot(int(np.ceil(self.batch_size/3)), 9, img_count)
                plt.imshow((P[idx] + 1) / 2)
                img_count += 1
                plt.axis('off')

                fig.add_subplot(int(np.ceil(self.batch_size/3)), 9, img_count)
                plt.imshow((Pv[idx] + 1) / 2)
                img_count += 1
                plt.axis('off')

                fig.add_subplot(int(np.ceil(self.batch_size/3)), 9, img_count)
                plt.imshow((Pt[idx] + 1) / 2)
                img_count += 1
                plt.axis('off')

        plt.savefig('%s/visualization%d.png'%(sample_path, step))
        plt.close()

    def textual_encoder(self, item_text, reuse=False):
        with tf.variable_scope('textual') as scope:
            if reuse:
                scope.reuse_variables()

            alpha = self.get_variable('W', [self.text_dim * self.max_num], stddev=0.2, name='alpha')
            item_text = tf.multiply(item_text, alpha)

            text_W = self.get_variable('W', [self.text_dim * self.max_num, 512], stddev=0.2, name='text_w')
            text_b = self.get_variable('b', [512], name='text_b')
            ht0 = tf.nn.relu(tf.matmul(item_text, text_W) + text_b)

            trans_W = self.get_variable('W', [512, 512], stddev=0.2, name='trans_w')
            trans_b = self.get_variable('b', [512], name='trans_b')
            transd = tf.matmul(ht0, trans_W) + trans_b
            transd = tf.expand_dims(tf.expand_dims(transd, 1), 1)

            ht0 = tf.expand_dims(tf.expand_dims(ht0, 1), 1)

            dim1 = tf.shape(ht0)[0]

            ht1 = deconv2d(ht0, [dim1, 2, 2, 64 * 8], name='g_ht1')
            ht2 = deconv2d(tf.nn.relu(ht1), [dim1, 4, 4, 64 * 8], name='g_ht2')
            ht3 = deconv2d(tf.nn.relu(ht2), [dim1, 8, 8, 64 * 8], name='g_ht3')
            ht4 = deconv2d(tf.nn.relu(ht3), [dim1, 16, 16, 64 * 8], name='g_ht4')
            ht5 = deconv2d(tf.nn.relu(ht4), [dim1, 32, 32, 64 * 4], name='g_ht5')
            ht6 = deconv2d(tf.nn.relu(ht5), [dim1, 64, 64, 64 * 2], name='g_ht6')
            ht7 = deconv2d(tf.nn.relu(ht6), [dim1, 128, 128, 64], name='g_ht7')

            info = [ht7, ht6, ht5, ht4, ht3, ht2, ht1, transd]

            return ht0, info

    def visual_encoder(self, items, reuse=False):
        with tf.variable_scope('visual') as scope:
            if reuse:
                scope.reuse_variables()

            hv1 = conv2d(items, 64, name='g_hv1_conv')
            hv2 = batch_norm(conv2d(tf.nn.relu(hv1), 64 * 2, name='g_hv2_conv'), name='g_bn_hv2', is_training=self.is_training)
            hv3 = batch_norm(conv2d(tf.nn.relu(hv2), 64 * 4, name='g_hv3_conv'), name='g_bn_hv3', is_training=self.is_training)
            hv4 = batch_norm(conv2d(tf.nn.relu(hv3), 64 * 8, name='g_hv4_conv'), name='g_bn_hv4', is_training=self.is_training)
            hv5 = batch_norm(conv2d(tf.nn.relu(hv4), 64 * 8, name='g_hv5_conv'), name='g_bn_hv5', is_training=self.is_training)
            hv6 = batch_norm(conv2d(tf.nn.relu(hv5), 64 * 8, name='g_hv6_conv'), name='g_bn_hv6', is_training=self.is_training)
            hv7 = batch_norm(conv2d(tf.nn.relu(hv6), 64 * 8, name='g_hv7_conv'), name='g_bn_hv7', is_training=self.is_training)
            hv8 = conv2d(tf.nn.relu(hv7), 64 * 8, name='g_hv8_conv')

            info = [hv1,hv2,hv3,hv4,hv5,hv6,hv7,hv8]

            return hv8, info

    def visual_decoder(self, visual_code, info, reuse=False):
        with tf.variable_scope('visual') as scope:
            if reuse:
                scope.reuse_variables()

            s, s2, s4, s8, s16, s32, s64, s128 = 256, 128, 64, 32, 16, 8, 4, 2
            dim1 = tf.shape(visual_code)[0]
            [hv1, hv2, hv3, hv4, hv5, hv6, hv7, hv8] = info

            visual_code = batch_norm(visual_code, name = 'g_bn_hv', is_training = self.is_training)
            hv_1 = deconv2d(tf.nn.relu(visual_code), [dim1, s128, s128, 64 * 8], name='g_hv_1')
            hv_1 = tf.nn.dropout(batch_norm(hv_1, name='g_hv_1_bn', is_training=self.is_training), self.drop)
            hv_1 = tf.concat([hv_1, hv7], 3)

            hv_2 = deconv2d(tf.nn.relu(hv_1), [dim1, s64, s64, 64 * 8], name='g_hv_2')
            hv_2 = tf.nn.dropout(batch_norm(hv_2, name='g_hv_2_bn', is_training=self.is_training), self.drop)
            hv_2 = tf.concat([hv_2, hv6], 3)

            hv_3 = deconv2d(tf.nn.relu(hv_2), [dim1, s32, s32, 64 * 8], name='g_hv_3')
            hv_3 = tf.nn.dropout(batch_norm(hv_3, name='g_hv_3_bn', is_training=self.is_training), self.drop)
            hv_3 = tf.concat([hv_3, hv5], 3)

            hv_4 = deconv2d(tf.nn.relu(hv_3), [dim1, s16, s16, 64 * 8], name='g_hv_4')
            hv_4 = batch_norm(hv_4, name='g_hv_4_bn', is_training=self.is_training)
            hv_4 = tf.concat([hv_4, hv4], 3)

            hv_5 = deconv2d(tf.nn.relu(hv_4), [dim1, s8, s8, 64 * 4], name='g_hv_5')
            hv_5 = batch_norm(hv_5, name='g_hv_5_bn', is_training=self.is_training)
            hv_5 = tf.concat([hv_5, hv3], 3)

            hv_6 = deconv2d(tf.nn.relu(hv_5), [dim1, s4, s4, 64 * 2], name='g_hv_6')
            hv_6 = batch_norm(hv_6, name='g_hv_6_bn', is_training=self.is_training)
            hv_6 = tf.concat([hv_6, hv2], 3)

            hv_7 = deconv2d(tf.nn.relu(hv_6), [dim1, s2, s2, 64], name='g_hv_7')
            hv_7 = batch_norm(hv_7, name='g_hv_7_bn', is_training=self.is_training)
            hv_7 = tf.concat([hv_7, hv1], 3)

            hv_8 = deconv2d(tf.nn.relu(hv_7), [dim1, s, s, self.c_dim], name='g_hv_8')
            return tf.nn.tanh(hv_8)


# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# sess = tf.Session(config=config)
# mttg_model = MTTG(sess=sess, checkpoint_path='./checkpoint/', sample_path='./sample/',
#                         data_path = '../', batch_size=32)
#
# mttg_model.train()


