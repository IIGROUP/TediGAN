# python3.7
"""Contains the class of dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.utils.data import Dataset
from torchvision.transforms import functional as trans_fn
from torchvision import datasets
from tqdm import tqdm
import lmdb
from functools import partial
import multiprocessing
from io import BytesIO
import argparse
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

"""
load dataset for training the text encoder 
"""

def prepare_data(data, mode):
    gt_codes, imgs, captions, captions_lens = data

    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)

    real_codes = []
    real_imgs = []
    for i in range(len(gt_codes)):
        gt_codes[i] = gt_codes[i][sorted_cap_indices]
        if mode != 'txt':
            imgs[i] = imgs[i][sorted_cap_indices]
        if torch.cuda.is_available():
            real_codes.append(Variable(gt_codes[i]).cuda())
            if mode != 'txt':
                real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_codes.append(Variable(gt_codes[i]))
            if mode != 'txt':
                real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()

    if torch.cuda.is_available():
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)

    # print('lens of imgs', len(imgs))
    # print('size of imgs[0]', imgs[0].size())
    out = [real_codes, captions, sorted_cap_lens] if mode == 'txt' else [real_codes, real_imgs]
    return out


def get_codes(img_path, normalize):
    img = np.load(img_path)
    img = torch.from_numpy(img)
    return [img]


class ToOneHot(object):
	""" Convert the input PIL image to a one-hot torch tensor """
	def __init__(self, n_classes=None):
		self.n_classes = n_classes

	def onehot_initialization(self, a):
		if self.n_classes is None:
			self.n_classes = len(np.unique(a))
		out = np.zeros(a.shape + (self.n_classes, ), dtype=int)
		out[self.__all_idx(a, axis=2)] = 1
		return out

	def __all_idx(self, idx, axis):
		grid = np.ogrid[tuple(map(slice, idx.shape))]
		grid.insert(axis, idx)
		return tuple(grid)

	def __call__(self, img):
		img = np.array(img)
		one_hot = self.onehot_initialization(img)
		return one_hot

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class TextDataset(data.Dataset):
    def __init__(self, data_dir, mode, split='train'):
        self.norm = transforms.Compose([transforms.ToTensor()])
        self.embeddings_num = 10

        self.imsize = []

        self.data = []
        self.data_dir = data_dir
        self.mode = mode
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.n_words = self.load_text_data(data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'% (filenames[i], cnt))
        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')

        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                print("filepath", filepath)
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding='latin1')
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix, txt_word_num=20):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        num_words = len(sent_caption)
        x = np.zeros((txt_word_num, 1), dtype='int64')
        x_len = num_words
        if num_words <= txt_word_num:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:txt_word_num]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = txt_word_num
        return x, x_len

    def __getitem__(self, index):

        key = self.filenames[index]
        # cls_id = self.class_id[index]
        data_dir = self.data_dir

        img_name = '%s/inverted_code/%s.npy' % (data_dir, key)
    
        codes = get_codes(img_name, normalize=self.norm)
        if self.mode != 'txt':
            img = get_imgs(img_name, self.mode)       
        else:
            img = img_name
        # randomly select a sentence
        sent_ix = random.randint(0, self.embeddings_num) # randomly select a caption out of ten.
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)
        return codes, img, caps, cap_len

    def __len__(self):
        return len(self.filenames)

def get_imgs(img_path, mode):
    label_nc = 19
    if mode == 'skt':
        img_name = img_path.replace('inverted_code', 'sketch').replace('npy', 'jpg')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()])
    if mode == 'lab':
        img_name = img_path.replace('inverted_code', 'label').replace('npy', 'png')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            ToOneHot(label_nc),
            transforms.ToTensor()])
    img = Image.open(img_name)
    img = img.convert('L') if mode == 'lab' else img.convert('RGB')
    img = transform(img)
    ret = [img]
    return ret

"""
load dataset for training the image encoder.
"""

class InvDataset(Dataset):
    def __init__(self, path, transform, resolution=256):

        self.root_dir = path
        self.resolution = (resolution, resolution)
        self.transform = transform

        self.image_paths = sorted(os.listdir(self.root_dir))
        self.num_samples = len(self.image_paths)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = Image.open(os.path.join(self.root_dir, image_path))
        img = img.resize(self.resolution)

    if self.transform is not None:
        img = self.transform(img)
        return img
