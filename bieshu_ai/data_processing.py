import imp
import os
import re
import json
import pickle

import jieba

# from utils import lazy_property, DataLoaderBase
from .utils import lazy_property, DataLoaderBase


class DataLoader(DataLoaderBase):
    def __init__(self) -> None:
        super().__init__()

        self.max_len = 20  # 每个句子最长为20
        self.unk_id = self.word2id.get('<UNK>')
        self.pad_id = self.word2id.get('<PAD>')

    @lazy_property
    def class_num(self):
        return len(set(self.data.get('train_y') + self.data.get('test_y')))

    def get_data(self, mode='train'):
        """
            获取数据
            args:
                mode: ['train', 'test']
        """
        if mode not in ['train', 'test']:
            raise ('Parameter mode must is train or test')
        
        if mode == 'train':
            x = self.data.get('train_x')
            y = self.format_label(self.data.get('train_y'))
            return x, y
        
        if mode == 'test':
            x = self.data.get('test_x')
            y = self.format_label(self.data.get('test_y'))
            return x, y

    def format_label(self, y):
        format_y = []
        for i in y:
            format_i = [0] * self.class_num
            format_i[i] = 1
            format_y.append(format_i)
        return format_y

    def word_to_index(self, input_text):
        """
            对输入的字符串转化为对应的索引
            args:
                input_text: 字符串
            return:
                word_index: 固定长度的列表
                word_index_len: 真实的单词数（句子长度）
        """      
        words = list(jieba.cut(input_text))
        word_index_len = len(words) if len(words) > self.max_len else self.max_len

        word_index = [self.pad_id] * self.max_len
        for i, word in enumerate(words[:word_index_len]):
            index = self.word2id.get(word)
            if index is None:
                index = self.unk_id
            word_index[i] = index
        return word_index, word_index_len
    
    def index_to_word(self, word_index):
        """
            对输入的索引列表返回字符串
        """
        words = [self.id2word.get(i) for i in word_index]
        return re.sub(r'<PAD>|<UNK>', '', "".join(words))


    


