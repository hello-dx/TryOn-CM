import tensorflow as tf
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import time
import mttg
from ops import *
import sklearn.metrics
from tqdm import tqdm

def metrics(array, metrics_map):
    out = []
    for metrics in metrics_map:
        if 'AUC' in metrics:
            a = np.ones([1])
            # print(np.shape(array))
            b = np.zeros([np.shape(array)[1]-1])
            label = np.concatenate([a, b])
            aa = []
            for i in array:
                aaa = sklearn.metrics.roc_auc_score(label, i)
                aa.append(aaa)
            score = np.average(aa)
            out.append(score)

        if 'MRR' in metrics:
            a = np.argsort(-array, axis=1)
            a = np.where(a == 0)[1]
            a = 1. / (a + 1)
            score = np.average(a)
            out.append(score)

        if 'TOP@' in metrics:
            k = int(metrics.replace('TOP@', ''))
            aaaa = []
            for i in array:
                aa = i - i[0]
                aa[aa > 0] = 1
                aa[aa < 0] = 0
                aaa = np.sum(aa)
                if aaa + 1 <= k:
                    aaaa.append(1)
                else:
                    aaaa.append(0)
            score = np.average(np.array(aaaa))
            out.append(score)
    return out


class TryonCM(object):
    def __init__(self, sess, data_path, img_size=256, c_dim=3, batch_size=None,
                 checkpoint_path=None, sample_path=None):

        self.data_path = data_path
        self.outfit_path = self.data_path + 'outfits/'
        self.feature_path = self.data_path + 'items/'
        self.checkpoint_path = checkpoint_path
        self.sample_path = sample_path

        self.data = json.load(open(self.data_path + 'outfit.json', 'r'))
        self.item_data = json.load(open(self.data_path + 'item.json', 'r'))
        self.dictionary = json.load(open(self.data_path + 'dictionary.json', 'r'))
        self.text_dim = len(self.dictionary)
        self.feature_dim = 4096
        self.hidden_dim = 512

        self.sess = sess
        self.batch_size = batch_size
        self.img_size = img_size
        self.c_dim = c_dim
        self.max_num = 4

    def get_variable(self, type, shape, stddev=0.01, name=None):
        if type == 'W':
            var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(stddev=stddev))
            tf.add_to_collection('regular_losses', tf.contrib.layers.l2_regularizer(0.005)(var))
            return var
        elif type == 'b':
            var = tf.get_variable(name=name, shape=shape, dtype=tf.float32,
                                  initializer=tf.zeros_initializer())
            return var

    def processing(self):
        self.textFea_dict, self.imgFea_dict, self.imgArr_dict = {}, {}, {}
        print('processing data')
        for item_id in tqdm(self.item_data):
            ## t
            item_desc = self.item_data[item_id]['desc'].replace('.', '').split()
            wordFea = np.zeros([self.text_dim], dtype=np.float32)
            for word in item_desc:
                try:
                    wordFea[self.dictionary.index(word)] = 1
                except:
                    pass
            self.textFea_dict[item_id] = wordFea
            ## feature
            item_feature = np.load(self.data_path + 'item_feature/' + item_id + '.npy')
            self.imgFea_dict[item_id] = item_feature
            ## array
            item_array = np.load(self.data_path + 'item_array/' + item_id + '.npy')
            self.imgArr_dict[item_id] = item_array


    def build_model(self):
        self.is_training = tf.placeholder(dtype=tf.bool, shape=None, name='is_training')
        self.drop = tf.placeholder(dtype=tf.float32, shape=None, name='drop')
        self.P = tf.placeholder(shape=[None, self.img_size, self.img_size, self.c_dim], dtype=tf.float32, name='outfit')
        self.item_img = tf.placeholder(shape=[None, self.img_size, self.img_size, self.c_dim], dtype=tf.float32, name='item_img')
        self.item_num = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name='item_num')
        self.item_text = tf.placeholder(shape=[None, self.text_dim * self.max_num], dtype=tf.float32, name='item_text')
        self.feature = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_dim], name='feature')

        self.item_imgk = tf.placeholder(shape=[None, self.img_size, self.img_size, self.c_dim], dtype=tf.float32, name='item_imgk')
        self.item_numk = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name='item_numk')
        self.item_textk = tf.placeholder(shape=[None, self.text_dim * self.max_num], dtype=tf.float32, name='item_textk')
        self.featurek = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_dim], name='featurek')

        h_d_N, L_d = self.bi_lstm(self.feature, self.item_num)
        h_d_Nk, _ = self.bi_lstm(self.featurek, self.item_numk, reuse=True)

        mttg_model = mttg.MTTG(sess=self.sess, checkpoint_path=self.checkpoint_path, sample_path='./sample/',
                               data_path = self.data_path, batch_size=self.batch_size)

        h_v_K, h_v, h_t_0, h_t, Pv, Pt = mttg_model.mttg(self.item_img, self.item_num, self.item_text)
        h_v_Kk, _, h_t_0k, _, _, _ = mttg_model.mttg(self.item_imgk,self.item_numk, self.item_textk, reuse=True)

        self.P_ = self.P
        self.Pv_ = Pv
        self.Pt_ = Pt

        s = self.predict(h_v_K, h_d_N, h_t_0)
        sk = self.predict(h_v_Kk, h_d_Nk, h_t_0k, reuse=True)
        self.score = s
        
        s_pred = tf.concat([s, sk], axis=0)
        s_label = tf.concat([tf.ones(tf.shape(s)), tf.zeros(tf.shape(sk))], axis=0)

        ## losses
        self.L_Gt = tf.reduce_mean(tf.square(h_v[0] - h_t[0])) \
                     + tf.reduce_mean(tf.square(h_v[6] - h_t[6])) \
                     + tf.reduce_mean(tf.square(h_v[1] - h_t[1])) \
                     + tf.reduce_mean(tf.square(h_v[2] - h_t[2])) \
                     + tf.reduce_mean(tf.square(h_v[3] - h_t[3])) \
                     + tf.reduce_mean(tf.square(h_v[4] - h_t[4])) \
                     + tf.reduce_mean(tf.square(h_v[5] - h_t[5])) \
                     + tf.reduce_mean(tf.square(h_v[7] - h_t[7]))
        self.L_d = L_d
        self.L_Gv = tf.reduce_mean(tf.abs(self.P - Pv))
        self.L_s = tf.reduce_mean(tf.square(s_pred - s_label))
        self.L = self.L_s + self.L_d + self.L_Gv + self.L_Gt

        ## training step
        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(0.001, self.global_step, decay_steps=800, decay_rate=0.85, staircase=False)
        self.optim = tf.train.AdamOptimizer(lr).minimize(self.L)

    def predict(self, h_v_K, h_d_N, h_t_0, reuse=False):
        with tf.variable_scope("predict") as scope:
            if reuse:
                scope.reuse_variables()

            low_dim = 128

            dPre_W = self.get_variable(type='W', shape=[self.hidden_dim, low_dim], stddev=0.1, name='dPre_W')
            dPre_b = self.get_variable(type='b', shape=[low_dim], name='dPre_b')
            h_d_N = tf.tanh(tf.matmul(h_d_N, dPre_W) + dPre_b)

            h_v_K = tf.squeeze(tf.squeeze(h_v_K, axis=1), axis=1)
            vPre_W = self.get_variable(type='W', shape=[self.hidden_dim, low_dim], stddev=0.1, name='vPre_W')
            vPre_b = self.get_variable(type='b', shape=[low_dim], stddev=0.1, name='vPre_b')
            h_v_K = tf.tanh(tf.matmul(h_v_K, vPre_W) + vPre_b)

            h_t_0 = tf.squeeze(tf.squeeze(h_t_0, axis=1), axis=1)
            tPre_W = self.get_variable(type='W', shape=[self.hidden_dim, low_dim], stddev=0.1, name='tPre_W')
            tPre_b = self.get_variable(type='b', shape=[low_dim], name='tPre_b')
            h_t_0 = tf.tanh(tf.matmul(h_t_0, tPre_W) + tPre_b)

            code = tf.concat([h_d_N, h_v_K, h_t_0], axis=1)  #
            code = tf.nn.l2_normalize(code, 1)
            Pre_W = self.get_variable(type='W', shape=[code.shape[-1].value, 1], stddev=0.1, name='Pre_W')
            Pre_b = self.get_variable(type='b', shape=[1], name='Pre_b')
            code = tf.sigmoid(tf.matmul(code, Pre_W) + Pre_b)

        return code

    def bi_lstm(self, features, item_nums, reuse=False):
        with tf.variable_scope("lstm") as scope:
            if reuse:
                scope.reuse_variables()

            features = tf.nn.l2_normalize(features, dim=1)

            imgFnn_W = self.get_variable(type='W', shape=[self.feature_dim, self.hidden_dim], name='imgFnn_W')
            imgFnn_b = self.get_variable(type='b', shape=[self.hidden_dim], name='imgFnn_b')
            features = tf.matmul(features, imgFnn_W) + imgFnn_b

            features = tf.split(features, item_nums)
            rnn_img_embedding = []
            for batch in features:
                batch = tf.transpose(batch)
                batch = tf.pad(batch, [[0, 0], [0, self.max_num - tf.shape(batch)[1]]])
                batch = tf.transpose(batch)
                rnn_img_embedding.append(batch)

            self.rnn_img_embedding = tf.reshape(tf.concat(rnn_img_embedding, axis=0), [-1, self.max_num, self.hidden_dim])

            f_lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim, state_is_tuple=True)
            b_lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hidden_dim, state_is_tuple=True)

            self.lstm_outputs, code = tf.nn.bidirectional_dynamic_rnn(cell_fw=f_lstm_cell,
                                                                      cell_bw=b_lstm_cell,
                                                                      inputs=self.rnn_img_embedding,
                                                                      dtype=tf.float32,
                                                                      initial_state_fw=None,
                                                                      initial_state_bw=None,
                                                                      sequence_length=item_nums
                                                                      )
            ### training
            fw_lstm_outputs = self.lstm_outputs[0]
            bw_lstm_outputs = self.lstm_outputs[1]

            ## hidden_output
            fw_hidden = tf.reshape(fw_lstm_outputs[:, :-1, :], shape=[-1, self.hidden_dim])
            bw_hidden = tf.reshape(bw_lstm_outputs[:, :-1, :], shape=[-1, self.hidden_dim])

            ## target
            fw_target = tf.reshape(self.rnn_img_embedding[:, 1:, :], shape=[-1, self.hidden_dim])
            bw_target = tf.reverse_sequence(self.rnn_img_embedding, seq_lengths=item_nums, seq_dim=1, batch_dim=0)
            bw_target = tf.reshape(bw_target[:, 1:, :], shape=[-1, self.hidden_dim])

            x_img = tf.reshape(self.rnn_img_embedding, shape=[-1, self.hidden_dim])  # [None, embed_size]

            fw_ht = tf.reduce_sum(tf.multiply(fw_hidden, fw_target), axis=1)
            bw_ht = tf.reduce_sum(tf.multiply(bw_hidden, bw_target), axis=1)
            mask = tf.cast(fw_ht, tf.bool)

            fw_hx = tf.matmul(fw_hidden, tf.transpose(x_img))
            bw_hx = tf.matmul(bw_hidden, tf.transpose(x_img))

            fw_Pr = tf.divide(tf.exp(tf.boolean_mask(fw_ht, mask)), tf.reduce_sum(tf.exp(tf.boolean_mask(fw_hx, mask)), axis=1))
            bw_Pr = tf.divide(tf.exp(tf.boolean_mask(bw_ht, mask)), tf.reduce_sum(tf.exp(tf.boolean_mask(bw_hx, mask)), axis=1))

            L_d = - tf.reduce_mean(tf.log(fw_Pr)) - tf.reduce_mean(tf.log(bw_Pr))

        return code[0][1], L_d

    def save(self, checkpoint_path, step, acc):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.saver.save(self.sess, checkpoint_path+'%.2f'%acc, global_step=step)

    def process(self, outfit_id, data):
        try:
            P = np.array([load_image(self.outfit_path + outfit_i.replace('-0', '') + '.jpg') for outfit_i in outfit_id])
        except:
            P = []
        item_num, item_img, item_text, feature = [], [], [], []
        for outfit_i in outfit_id:
            try:
                data_dict = data[outfit_i]['clo'][:4]
            except:
                data_dict = data[outfit_i][:4]
            item_num.append(min(4, len(data_dict)))
            text_i = np.zeros([self.text_dim * self.max_num])
            for idx, item_i in enumerate(data_dict):
                feature.append(self.imgFea_dict[item_i])
                text_i[idx * self.text_dim:(idx + 1) * self.text_dim] = self.textFea_dict[item_i]
                item_img.append(self.imgArr_dict[item_i])
            item_text.append(text_i)

        return P, item_img, item_num, item_text, feature
  
    def train(self):
        self.build_model()
        self.processing()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.saver = tf.train.Saver(max_to_keep=1)

        train_sample = json.load(open(self.data_path + '/data/train_idx.json', 'r'))
        train_neg = json.load(open(self.data_path + '/data/train_neg.json', 'r'))

        max_auc = 0
        max_step = 0
        for step in range(0, 10001):
            outfit = random.sample(train_sample, self.batch_size)
            outfitk = random.sample(list(train_neg.keys()), self.batch_size)
            P, item_img, item_num, item_text, feature = self.process(outfit, self.data)
            _, item_imgk, item_numk, item_textk, featurek = self.process(outfitk, train_neg)

            _, L_s, L_Gt, L_Gv, L_d = \
                self.sess.run([self.optim, self.L_s, self.L_Gt, self.L_Gv, self.L_d],
                              {self.item_img: item_img, self.feature: feature, self.item_text: item_text, self.item_num: item_num,
                               self.item_imgk: item_imgk, self.featurek: featurek, self.item_textk: item_textk, self.item_numk: item_numk,
                               self.is_training: True, self.drop: 0.5, self.P: P, self.global_step: step})

            if step % 100==0:
                metric = self.valid()
                auc = metric[0]
                if auc > max_auc:
                    self.save(self.checkpoint_path, step, auc)
                    max_auc = auc
                    max_step = step
                log = 'Step: %d | L_s: %.3f | L_d: %.3f| L_Gv: %.3f | L_Gt: %.3f | ' \
                      'AUC-MRR-TOP@1-10-100-200: %.3f-%.3f-%.3f-%.3f-%.3f-%.3f | Saved: auc-%.3f-%d' \
                      % (step, L_s, L_d, L_Gv, L_Gt, metric[0], metric[1], metric[2], metric[3], metric[4], metric[5], max_auc, max_step)
                print(log)
        self.sess.close()
        self.test()

    def valid(self):
        METRICS_MAP = ['AUC', 'MRR', 'TOP@1', 'TOP@10', 'TOP@100', 'TOP@200']
        valid_index = json.load(open(self.data_path + '/data/valid_idx.json', 'r'))
        valid_sample = []
        for i in valid_index:
            for ii in range(500):
                valid_sample.append(i+'-'+str(ii))
        valid_data = json.load(open(self.data_path + '/data/valid_test_set.json', 'r'))

        score = np.zeros([int(len(valid_sample) / 500), 500])
        for test_i in range(int(len(valid_sample) / self.batch_size)):
            keys = valid_sample[test_i * self.batch_size:(test_i + 1) * self.batch_size]

            _, item_img, item_num, item_text, feature = self.process(keys, valid_data)
            accuracy = self.sess.run(self.score, {self.feature: feature, self.item_num: item_num,
                                                  self.item_img: item_img, self.item_text: item_text})
            accuracy = np.reshape(accuracy, [-1])

            for idx, key in enumerate(keys):
                outfit_id = key.split('-')[0]
                sample_id = int(key.split('-')[1])
                score[valid_index.index(outfit_id)][sample_id] = accuracy[idx]
        metric = metrics(score, METRICS_MAP)
        return metric

    def test(self):
        print('Start Testing...')
        from tqdm import tqdm
        METRICS_MAP = ['AUC', 'MRR', 'TOP@1', 'TOP@10', 'TOP@100', 'TOP@200']

        test_index = json.load(open(self.data_path + '/data/test_idx.json', 'r'))
        test_sample = []
        for i in test_index:
            for ii in range(500):
                test_sample.append(i + '-' + str(ii))
        test_data = json.load(open(self.data_path + '/data/valid_test_set.json', 'r'))

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        score = np.zeros([int(len(test_sample) / 500), 500])
        for test_i in tqdm(range(int(len(test_sample) / self.batch_size))):
            keys = test_sample[test_i * self.batch_size:(test_i + 1) * self.batch_size]
            _, item_img, item_num, item_text, feature = self.process(keys, test_data)
            accuracy = self.sess.run(self.score, {self.feature: feature, self.item_num: item_num,
                                                  self.item_img: item_img, self.item_text: item_text})
            accuracy = np.reshape(accuracy, [-1])

            for idx, key in enumerate(keys):
                outfit_id = key.split('-')[0]
                sample_id = int(key.split('-')[1])
                score[test_index.index(outfit_id)][sample_id] = accuracy[idx]

        metric = metrics(score, METRICS_MAP)
        log = 'Test Result: AUC-MRR-TOP@1-10-100-200: %.3f-%.3f-%.3f-%.3f-%.3f-%.3f' \
              %(metric[0], metric[1], metric[2], metric[3], metric[4], metric[5])
        print(log)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
sess = tf.Session(config=config)
model = TryonCM(sess=sess, checkpoint_path='./checkpoint/', sample_path='./sample/',
                data_path = '../', batch_size=32)

### training model
# model.train()
### using the pre-trained model
# model.test()

