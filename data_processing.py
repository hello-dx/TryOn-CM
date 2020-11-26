# from keras.applications.imagenet_utils import preprocess_input
# from keras.applications.vgg16 import VGG16
# from keras.models import Model
# from keras.preprocessing import image
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json
import random
import copy
import os
from collections import Counter

data_path = '../'
item_data = json.load(open(data_path + 'item.json', 'r'))
outfit_data = json.load(open(data_path + 'outfit.json', 'r'))

test_id = json.load(open(data_path + 'data/test_idx.json', 'r'))

def build_dictionary():
    dictionary = []
    word_bag = ''
    for item_id in item_data:
        word_bag += ' '+item_data[item_id]['title']

    count = Counter(word_bag.split())
    for word in count:
        if count[word]>40:
            dictionary.append(word)

    json.dump(dictionary, open(data_path+'dictionary.json', 'w'))


def extract_feature():
    import os
    from tqdm import tqdm
    img_list = os.listdir(data_path + '/items/')

    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)

    for item in tqdm(img_list):
        item_id = item[:-4]
        img_path = data_path + '/items/' + item
        ## array
        array = image.load_img(img_path, target_size=(256, 256))
        array = image.img_to_array(array)
        array = array/127 - 1
        np.save(item_path+'/item_array/%s.npy'%item_id, array)
        ## vgg feature
        vgg_feed = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(vgg_feed)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)  # fc2
        features = features[0]
        np.save(item_path+'/item_feature/%s.npy' % item_id, features)


def build_training_set(outfit_idx, outfit_data):
    candidate = {}
    for idx in outfit_idx:
        for item_idx in outfit_data[idx]['clo']:
            if item_data[item_idx]['cate'] not in candidate.keys():
                candidate[item_data[item_idx]['cate']] = [item_idx]
            else:
                candidate[item_data[item_idx]['cate']].append(item_idx)
    train_negative_set = {}
    for idx in outfit_idx:
        positive = random.sample(outfit_data[idx]['clo'], 1)[0]
        for i in range(1, 10):
            negative = positive
            while negative == positive:
                negative = random.sample(candidate[item_data[positive]['cate']], 1)[0]
            negative_outfit = copy.deepcopy(outfit_data[idx]['clo'])
            negative_outfit[negative_outfit.index(positive)] = negative
            train_negative_set[idx+'-'+str(i)] = negative_outfit
    return train_negative_set



def build_testing_set(outfit_idx, outfit_data):
    candidate = {}
    for idx in outfit_idx:
        for item_idx in outfit_data[idx]['clo']:
            if item_data[item_idx]['cate'] not in candidate.keys():
                candidate[item_data[item_idx]['cate']] = [item_idx]
            else:
                candidate[item_data[item_idx]['cate']].append(item_idx)
    testing_set = {}
    for idx in outfit_idx:
        testing_set[idx + '-0'] = outfit_data[idx]['clo']
        positive = random.sample(outfit_data[idx]['clo'], 1)[0]
        for i in range(1, 500):
            negative = positive
            while negative == positive:
                negative = random.sample(candidate[item_data[positive]['cate']], 1)[0]
            negative_outfit = copy.deepcopy(outfit_data[idx]['clo'])
            negative_outfit[negative_outfit.index(positive)] = negative
            testing_set[idx + '-' + str(i)] = negative_outfit
    return testing_set

def split_set():
    all_outfit_idx = list(outfit_data.keys())
    random.shuffle(all_outfit_idx)
    train_idx = all_outfit_idx[:9900]
    valid_test_idx = all_outfit_idx[9900:]
    valid_idx = valid_test_idx[:100]
    test_idx = valid_test_idx[100:]

    train_neg = build_training_set(train_idx, outfit_data)
    valid_test_set =  build_testing_set(valid_test_idx, outfit_data)

    if not os.path.exists(data_path+'/data/'):
        os.makedirs(data_path+'/data/')
    json.dump(train_idx, open(data_path+'/data/train_idx.json', 'w'))
    json.dump(train_neg, open(data_path+'/data/train_neg.json', 'w'))
    json.dump(valid_idx, open(data_path+'/data/valid_idx.json', 'w'))
    json.dump(test_idx, open(data_path+'/data/test_idx.json', 'w'))
    json.dump(valid_test_set, open(data_path+'/data/valid_test_set.json', 'w'))

# ## Image pixel array and feature
# extract_feature()
# ## Vocabulary dictionary
# build_dictionary()
# ## Training, validation and testing set
# split_set()


# train_data = data_path+'data/train_idx.json'
# train_idx = json.load(open(train_data, 'r'))
# a = random.sample(train_idx, 200)
# aa = build_testing_set(a, outfit_data)
# print(aa)
# print(len(aa))
# valid_data = []
# for i in a[:30]:
#     for ii in range(500):
#         valid_data.append(i + '-%d' % ii)
#
# dict = {}
# for i in valid_data:
#     dict[i] = aa[i]
#
# print(dict)
# print(len(dict))
#
# json.dump(dict, open('./train_show.json', 'w'))

