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
    # print(out)
    return out


class MTTCM(object):
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

        self.model()

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
                    wordFea[self.dictionary.index(word)] = 1# / len(item_desc)
                except:
                    pass
            self.textFea_dict[item_id] = wordFea
            ## feature
            item_feature = np.load(self.data_path + 'item_feature/' + item_id + '.npy')
            self.imgFea_dict[item_id] = item_feature
            ## array
            item_array = np.load(self.data_path + 'item_array/' + item_id + '.npy')
            self.imgArr_dict[item_id] = item_array


    def model(self):
        self.is_training = tf.placeholder(dtype=tf.bool, shape=None, name='is_training')
        self.drop = tf.placeholder(dtype=tf.float32, shape=None, name='drop')
        self.outfit = tf.placeholder(shape=[None, self.img_size, self.img_size, self.c_dim], dtype=tf.float32, name='outfit')
        self.item = tf.placeholder(shape=[None, self.img_size, self.img_size, self.c_dim], dtype=tf.float32, name='item')
        self.item_num = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name='item_num')
        self.item_text = tf.placeholder(shape=[None, self.text_dim * self.max_num], dtype=tf.float32, name='item_text')
        self.feature = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_dim], name='feature')

        self.itemk = tf.placeholder(shape=[None, self.img_size, self.img_size, self.c_dim], dtype=tf.float32, name='itemk')
        self.item_numk = tf.placeholder(shape=[self.batch_size], dtype=tf.int32, name='item_numk')
        self.item_textk = tf.placeholder(shape=[None, self.text_dim * self.max_num], dtype=tf.float32, name='item_textk')
        self.featurek = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_dim], name='featurek')

        lstmcode, lstm_loss = self.bi_lstm(self.feature, self.item_num)
        lstmcodek, _ = self.bi_lstm(self.featurek, self.item_numk, reuse=True)

        mttg_model = mttg.MTTG(sess=self.sess, checkpoint_path=self.checkpoint_path, sample_path='./sample/unet_text/',
                               data_path = self.data_path, batch_size=self.batch_size, is_training=self.is_training)

        visual_code, visual_info, text_code, text_info, Pv, Pt = mttg_model.mttg(self.item, self.item_num, self.item_title)
        visual_codek, _, text_codek, _, _, _ = mttg_model.mttg(self.itemk,self.item_numk, self.item_titlek, reuse=True)

        self.Pv_ = tf.image.resize_images(Pv, [256, 160])
        self.Pt_ = tf.image.resize_images(Pt, [256, 160])
        self.outfit_ = tf.image.resize_images(self.outfit, [256, 160])

        code = self.predict(visual_code, lstmcode, text_code)
        codek = self.predict(visual_codek, lstmcodek, text_codek, reuse=True)
        self.score = code

        ## losses
        self.code_loss = 0 # tf.reduce_mean(tf.square(visual_code - text_code))
        self.info_loss = tf.reduce_mean(tf.square(visual_info[0] - text_info[0])) \
                         + tf.reduce_mean(tf.square(visual_info[1] - text_info[1])) \
                         + tf.reduce_mean(tf.square(visual_info[2] - text_info[2])) \
                         + tf.reduce_mean(tf.square(visual_info[3] - text_info[3])) \
                         + tf.reduce_mean(tf.square(visual_info[4] - text_info[4])) \
                         + tf.reduce_mean(tf.square(visual_info[5] - text_info[5])) \
                         + tf.reduce_mean(tf.square(visual_info[6] - text_info[6])) \
                         + tf.reduce_mean(tf.square(visual_info[7] - text_info[7]))

        y_pred = tf.concat([code, codek], axis=0)
        y_label = tf.concat([tf.ones(tf.shape(code)), tf.zeros(tf.shape(codek))], axis=0)

        ## define vars
        pre_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'predict')
        lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'lstm')
        text_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mttg/text')
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'mttg/visual')
        ## define losses
        self.gen_loss = tf.reduce_mean(tf.abs(self.outfit - Pv))
        self.text_loss = self.info_loss
        self.lstm_loss = lstm_loss
        self.pre_loss = tf.reduce_mean(tf.square(y_pred - y_label))
        # self.pre_loss = -tf.reduce_mean(tf.sigmoid(tf.log(code-codek)))
        self.loss = self.pre_loss+self.lstm_loss+self.gen_loss+self.text_loss

        ## training step
        self.global_step = tf.Variable(0, dtype=tf.int64, name='global_step', trainable=False)
        lr = tf.train.exponential_decay(0.001, self.global_step, decay_steps=800, decay_rate=0.85, staircase=False)
        # self.gen_optim = tf.train.AdamOptimizer(0.1*lr, beta1=0.5).minimize(self.gen_loss, var_list=gen_vars)
        # self.text_optim = tf.train.AdamOptimizer(0.2*lr, beta1=0.5).minimize(self.text_loss, var_list=text_vars)
        self.lstm_optim = tf.train.AdamOptimizer(lr).minimize(self.lstm_loss)
        self.optim = tf.train.AdamOptimizer(lr).minimize(self.loss) #, var_list=lstm_vars+pre_vars+gen_vars)

    def predict(self, visualcode, lstmcode, textcode, reuse=False):
        with tf.variable_scope("predict") as scope:
            if reuse:
                scope.reuse_variables()
            low_dim = 128

            lstmPre_W = self.get_variable(type='W', shape=[self.hidden_dim, low_dim], stddev=0.1, name='lstmPre_W')
            lstmPre_b = self.get_variable(type='b', shape=[low_dim], name='lstmPre_b')
            lstmcode = tf.tanh(tf.matmul(lstmcode[0][1], lstmPre_W) + lstmPre_b)

            visualcode = tf.squeeze(tf.squeeze(visualcode, axis=1), axis=1)
            genPre_W = self.get_variable(type='W', shape=[self.hidden_dim, low_dim], stddev=0.1, name='genPre_W')
            genPre_b = self.get_variable(type='b', shape=[low_dim], stddev=0.1, name='genPre_b')
            visualcode = tf.tanh(tf.matmul(visualcode, genPre_W) + genPre_b)

            textcode = tf.squeeze(tf.squeeze(textcode, axis=1), axis=1)
            textPre_W = self.get_variable(type='W', shape=[self.hidden_dim, low_dim], stddev=0.1, name='textPre_W')
            textPre_b = self.get_variable(type='b', shape=[low_dim], name='textPre_b')
            textcode = tf.tanh(tf.matmul(textcode, textPre_W) + textPre_b)

            code = tf.concat([lstmcode, visualcode, textcode], axis=1)  #
            code = tf.nn.l2_normalize(code, 1)
            Pre_W = self.get_variable(type='W', shape=[code.shape[-1].value, 1], stddev=0.1, name='Pre_W')
            Pre_b = self.get_variable(type='b', shape=[1], name='Pre_b')
            code = tf.sigmoid(tf.matmul(code, Pre_W) + Pre_b)

        return code

    def bi_lstm(self, img_features, item_nums, reuse=False):
        with tf.variable_scope("lstm") as scope:
            if reuse:
                scope.reuse_variables()

            img_feature = tf.nn.l2_normalize(img_features, dim=1)

            imgFnn_W = get_variable(type='W', shape=[self.feature_dim, self.hidden_dim], mean=0, stddev=0.01,
                                    name='imgFnn_W')
            imgFnn_b = get_variable(type='b', shape=[self.hidden_dim], mean=0, stddev=0.01, name='imgFnn_b')
            img_feature = tf.matmul(img_feature, imgFnn_W) + imgFnn_b

            img_feature = tf.split(img_feature, item_nums)
            rnn_img_embedding = None
            for batch in img_feature:
                batch = tf.transpose(batch)
                batch = tf.pad(batch, [[0, 0], [0, self.max_num - tf.shape(batch)[1]]])
                batch = tf.transpose(batch)

                if rnn_img_embedding == None:
                    rnn_img_embedding = batch
                else:
                    rnn_img_embedding = tf.concat([rnn_img_embedding, batch], axis=0)
            self.rnn_img_embedding = tf.reshape(rnn_img_embedding, shape=[-1, self.max_num, self.hidden_dim])

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

            ##
            x_img = tf.reshape(self.rnn_img_embedding, shape=[-1, self.hidden_dim])  # [None, embed_size]

            fw_ht = tf.reduce_sum(tf.multiply(fw_hidden, fw_target), axis=1)
            bw_ht = tf.reduce_sum(tf.multiply(bw_hidden, bw_target), axis=1)
            mask = tf.cast(fw_ht, tf.bool)

            fw_hx = tf.matmul(fw_hidden, tf.transpose(x_img))
            bw_hx = tf.matmul(bw_hidden, tf.transpose(x_img))

            fw_Pr = tf.divide(tf.exp(tf.boolean_mask(fw_ht, mask)), tf.reduce_sum(tf.exp(tf.boolean_mask(fw_hx, mask)), axis=1))
            bw_Pr = tf.divide(tf.exp(tf.boolean_mask(bw_ht, mask)), tf.reduce_sum(tf.exp(tf.boolean_mask(bw_hx, mask)), axis=1))

            lstm_loss = - tf.reduce_mean(tf.log(fw_Pr)) - tf.reduce_mean(tf.log(bw_Pr))

        return code, lstm_loss

    def sample_model(self, sample_path, fold, step):
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)
        fig = plt.figure()
        fig.set_size_inches(20, 20)
        num = 8
        img_count = 1
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0.01)

        all_test_outfits = json.load(open('./test_outfit.json', 'r'))
        batch_test_outfits = [all_test_outfits[i:i + self.batch_size] for i in
                              range(0, len(all_test_outfits), self.batch_size)]
        for idxi, test_outfits in enumerate(batch_test_outfits):
            outfit, item, item_num, title, feature = self.process(test_outfits, self.data)
            outfit_, Pv_, Pt_ = self.sess.run([self.outfit_, self.Pv_, self.Pt_],
                                               {self.outfit: outfit, self.item: item, self.feature: feature,
                                                self.item_title: title, self.item_num: item_num})
            test_item_nums = []
            init = 0
            for i in item_num:
                test_item_nums.append(init + i)
                init = init + i
            outfit_pieces = np.split(item, test_item_nums[:-1])

            for idx, i in enumerate(outfit_pieces):
                fig.add_subplot(10, 9, img_count)
                plt.imshow((outfit_[idx] + 1) / 2)
                img_count += 1
                plt.axis('off')

                fig.add_subplot(10, 9, img_count)
                plt.imshow((Pv_[idx] + 1) / 2)
                img_count += 1
                plt.axis('off')

                fig.add_subplot(10, 9, img_count)
                plt.imshow((Pt_[idx] + 1) / 2)
                img_count += 1
                plt.axis('off')


        plt.savefig('%s/train_%d.png' % (sample_path, step))
        plt.close()

    def save(self, checkpoint_path, step, acc):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.saver.save(self.sess, checkpoint_path+'%.2f'%acc, global_step=step)

    def process(self, outfit_id, data):
        try:
            outfit = np.array([load_image(self.outfit_path + outfit_i.replace('-0', '') + '.jpg') for outfit_i in outfit_id])
        except:
            outfit = []
        item_num, item, title, feature = [], [], [], []
        for outfit_i in outfit_id:
            try:
                data_dict = data[outfit_i]['clo'][:4]
            except:
                data_dict = data[outfit_i][:4]
            item_num.append(min(4, len(data_dict)))
            title_i = np.zeros([self.text_dim * self.max_num])
            for idx, item_i in enumerate(data_dict):
                feature.append(self.imgFea_dict[item_i])
                title_i[idx * self.text_dim:(idx + 1) * self.text_dim] = self.textFea_dict[item_i]
                item.append(self.imgArr_dict[item_i])

            title.append(title_i)
        return outfit, item, item_num, title, feature

    def test(self, fold):
        from tqdm import tqdm
        METRICS_MAP = ['AUC', 'MRR', 'TOP@1', 'TOP@10', 'TOP@100', 'TOP@200']
        test_sample = json.load(open(self.data_path+'/data/%dvalid_sample.json' % fold, 'r'))
        test_index = list(set([i.split('-')[0] for i in test_sample]))
        test_neg = json.load(open(self.data_path+'/data/%dtest_neg.json' % fold, 'r'))

        score = np.zeros([int(len(test_sample)/500), 500])
        for test_i in range(int(len(test_sample) / self.batch_size)): #len(test_neg) / self.batch_size
            keys = test_sample[test_i * self.batch_size:(test_i + 1) * self.batch_size]

            _, item, item_num, title, feature = self.process(keys, test_neg)
            accuracy = self.sess.run(self.score, {self.feature: feature, self.item_num: item_num,
                                                  self.item: item, self.item_title: title})
            accuracy = np.reshape(accuracy, [-1])

            for idx, key in enumerate(keys):
                outfit_id = key.split('-')[0]
                sample_id = int(key.split('-')[1])
                # print(test_index.index(outfit_id), sample_id)
                score[test_index.index(outfit_id)][sample_id] = accuracy[idx]

        metirc = metrics(score, METRICS_MAP)
        print(metirc)
        return metirc[0]


    def test_final(self, folds):
        from tqdm import tqdm
        result = []
        METRICS_MAP = ['AUC', 'MRR', 'TOP@1', 'TOP@10', 'TOP@100', 'TOP@200']
        for fold in folds:
            print(fold)
            time1 = time.time()
            test_sample = json.load(open(self.data_path+'/data/%dtest_sample.json' % fold, 'r'))
            test_index = list(set([i.split('-')[0] for i in test_sample]))
            test_neg = json.load(open(self.data_path+'/data/%dtest_neg.json' % fold, 'r'))

            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            self.sess = tf.Session(config=config)
            self.saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path+'/%dfold/'%fold)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

            score = np.zeros([int(len(test_sample)/500), 500])
            for test_i in range(int(len(test_sample) / self.batch_size)): #len(test_neg) / self.batch_size
                keys = test_sample[test_i * self.batch_size:(test_i + 1) * self.batch_size]
                # keys = test_aaa[test_i]

                _, item, item_num, title, feature = self.process(keys, test_neg)
                # print(keys, np.array(feature).shape)

                accuracy = self.sess.run(self.score, {self.feature: feature, self.item_num: item_num,
                                                      self.item: item, self.item_title: title})

                accuracy = np.reshape(accuracy, [-1])

                for idx, key in enumerate(keys):
                    outfit_id = key.split('-')[0]
                    sample_id = int(key.split('-')[1])
                    score[test_index.index(outfit_id)][sample_id] = accuracy[idx]
                    # if test_i%200==0:
                    #     print(test_index.index(outfit_id), sample_id, accuracy[idx])

            metirc = metrics(score, METRICS_MAP)
            print(metirc)
            result.append(metirc)
            np.save(self.checkpoint_path + '/%dfold-%.4f.npy' % (fold, metirc[0]), score)

        result = np.array(result)
        result = np.average(result, axis=0)
        print(result)

  
    def train(self):
        self.processing()
        for fold in [2,7]:
            print(fold)
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.5
            # config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)
            self.saver = tf.train.Saver(max_to_keep=1)

            train_sample = json.load(open(self.data_path + '/data/%dtrain_sample.json' % fold, 'r'))
            train_neg = json.load(open(self.data_path + '/data/%dtrain_neg.json' % fold, 'r'))
            # valid_sample = json.load(open(self.data_path + '/data/%dvalid_sample.json' % fold, 'r'))
            # valid_neg = json.load(open(self.data_path + '/data/%dvalid_neg.json' % fold, 'r'))

            max_mrr = 0
            max_step = 0
            mrr = 0
            for step in range(0, 6001):
                time0 = time.time()
                outfit = random.sample(train_sample, self.batch_size)
                outfitk = random.sample(list(train_neg.keys()), self.batch_size)
                outfit, item, item_num, title, feature = self.process(outfit, self.data)
                _, itemk, item_numk, titlek, featurek = self.process(outfitk, train_neg)

                # _ = \
                #     self.sess.run([self.text_optim],
                #                   {self.item: item, self.feature: feature, self.item_title: title, self.item_num: item_num,
                #                    self.itemk: itemk, self.featurek: featurek, self.item_titlek: titlek, self.item_numk: item_numk,
                #                    self.outfit: outfit, self.global_step: step})

                _, loss, text_loss, gen_loss, lstm_loss = \
                    self.sess.run([self.optim, self.pre_loss, self.text_loss, self.gen_loss, self.lstm_loss],
                                  {self.item: item, self.feature: feature, self.item_title: title, self.item_num: item_num,
                                   self.itemk: itemk, self.featurek: featurek, self.item_titlek: titlek, self.item_numk: item_numk,
                                   self.outfit: outfit, self.global_step: step})

                # if step % 50 == 0:
                #     self.sample_model(self.sample_path, 0, step)
                if step % 100==0 and step>3999:
                    mrr = self.test(fold)

                    # all_test = []
                    # for test_i in range(int(len(valid_neg) / self.batch_size)):
                    #     outfit, item, item_num, title, feature = self.process(
                    #         valid_sample[test_i * self.batch_size:(test_i + 1) * self.batch_size], valid_neg)
                    #
                    #     score = self.sess.run(self.score, {self.item: item, self.feature: feature,
                    #                                        self.item_title: title, self.item_num: item_num})
                    #
                    #     score = np.reshape(score, [-1])
                    #     all_test.append(score)
                    #
                    # all_test = np.array(all_test)
                    # [auc, mrr1, top1, top5] = metrics(all_test, ['AUC','MRR', 'TOP@1', 'TOP@5'])

                    if mrr > max_mrr:
                        self.save(self.checkpoint_path + '/%dfold/' % fold, step, mrr)
                        max_mrr = mrr
                        max_step = step

                    log = 'Step: [%d-%d] | Spend: %.2f | Loss: %.2f | LSTM: %.2f| GAN: %.2f | Text: %.2f | Metrices: %.2f | Max: %.2f-%d' \
                          % (fold, step, time.time()-time0, loss, lstm_loss, gen_loss, text_loss, mrr, max_mrr, max_step)
                    print(log)

            self.sess.close()

            self.test_final([fold])

#  
# config = tf.ConfigProto()
# model = MTTCM(sess=None, data_path='/home/share/yaofubin/', batch_size=32,
#               checkpoint_path='/home/share/yaofubin/checkpoint/mttcm_title/', sample_path = './sample/mttcm/')
# model.train()



# model = MTTCM(sess=None, data_path='/home/share/yaofubin/', batch_size=80, is_training=False,
#               checkpoint_path='/home/share/yaofubin/checkpoint/mttcm_title/', sample_path = './sample/mttcm/')
# model.test_final(0)

import os
checkpoint_path='/home/share/yaofubin/checkpoint/mttcm_title/'
npy_file = os.listdir(checkpoint_path)
a = []
for i in npy_file:
    if '.npy' in i:
        # print(checkpoint_path+i)
        aa = np.load(checkpoint_path+i)
    else:
        continue

    METRICS_MAP = ['AUC', 'MRR', 'TOP@1', 'TOP@10', 'TOP@100', 'TOP@200']
    metirc = metrics(aa, METRICS_MAP)
    print(metirc, i)
    a.append(metirc)
a = np.array(a)
print(np.average(a, axis=0))