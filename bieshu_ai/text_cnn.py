import numpy

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class TextCNN(keras.Model):
    def __init__(self):
        super().__init__()

        self.max_len = 20  # 每个句子最大单词个数
        self.word_dim = 128  # 词向量维度
        self.word_num = 17000  # 最大单词数
        self.class_num = 10  # 类别总数
        self.dropout = 0.2

        tf.random.set_seed(520)        
        self.embeddings_layer = layers.Embedding(self.word_num, self.word_dim)

        self.conv_layer = layers.Conv1D(128, 3)
        self.pooling_layer = layers.MaxPool1D(8, padding='same')

        self.flatten_layer = layers.Flatten()

        self.dropout_layer = layers.Dropout(self.dropout)
        self.dense_layer = layers.Dense(self.class_num, activation='softmax')

    def call(self, batch_word_index):
        """
            args:
                batch_word_index: list, batch * sentence_length
            return:
                x: 
        """
        vector_x = self.embeddings_layer(tf.Variable(batch_word_index))
        x_1 = self.pooling_layer(self.conv_layer(vector_x))
        x = self.dense_layer(self.dropout_layer(self.flatten_layer(x_1)))
        return x
        


        

