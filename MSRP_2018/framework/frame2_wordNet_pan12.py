'''思路:

    动态池化技术
    A:首先加载向量文件，采用字典的方式
    B:首先加载每一行的句对，去停用词，标点符号，存储到list中
    C：将每一对单词的向量求相似度，分别对两个句子遍历。
    D：写成一个向量矩阵

 '''
import logging
import sys
import re
from math import sqrt
import numpy as np
import random
import tensorflow as tf
import operator
from functools import reduce
from sklearn.metrics import classification_report
from nltk.stem.porter import PorterStemmer

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))


class Datas:
    def __init__(self, batch_size):
        self.word2vec_model = {}
        self.wordNet = {}
        self.train_data = []
        self.batch_size = batch_size
        self.stop_list = []
        self.test_data = []

    def read_stopword(self, path):  # 加载停用词列表
        file_object = open(path, 'r', encoding='utf-8')
        try:
            all_line = file_object.readlines()
            for line in all_line:
                self.stop_list.append(line.strip())
        finally:
            file_object.close()

    def read_wordNet(self, path):  # 加载同义词列表
        file_object = open(path, 'r', encoding='utf-8')
        try:
            all_line = file_object.readlines()
            for line in all_line:
                info = line.split()
                self.wordNet.setdefault(info[0], info[1:])
        finally:
            file_object.close()

    def read_word2vec_model(self, path):  # 加载embedding词库
        file_object = open(path, encoding='utf-8')
        try:
            all_line = file_object.readlines()
            for line in all_line:
                info = line.split()
                if len(info) > 2:
                    word = info[0]
                    vec = [float(x) for x in info[1:]]
                    self.word2vec_model.setdefault(word, vec)
        finally:
            file_object.close()

    def read_train_data(self, path):  # 读取训练文件，并去除除英文外的所有子字符，停用词
        train_file = open(path, 'r', encoding='utf-8')
        try:
            lines = train_file.readlines()
            for line in lines:
                sen1 = []
                sen2 = []
                sen_pair = []
                info = line.split('\t')
                # print(info[0:])
                label = int(info[0])
                for x in re.sub('[^a-zA-Z ]', '', info[3]).split():
                    if not self.stop_list.__contains__(x):
                        sen1.append(x)
                for x in re.sub('[^a-zA-Z ]', '', info[4]).split():
                    if not self.stop_list.__contains__(x):
                        sen2.append(x)
                sen_pair.append(sen1)
                sen_pair.append(sen2)
                # if len(sen1) > 1 and len(sen2) > 1:
                self.train_data.append([label, sen_pair])
        finally:
            train_file.close()

    def read_test_data(self, path):  # 测试文件，功能同训练文件
        test_file = open(path, 'r', encoding='utf-8')
        try:
            lines = test_file.readlines()
            for line in lines:
                sen1 = []
                sen2 = []
                sen_pair = []
                info = line.split('\t')
                label = int(info[0])
                for x in re.sub('[^a-zA-Z ]', '', info[3]).split():
                    if not self.stop_list.__contains__(x):
                        sen1.append(x)
                for x in re.sub('[^a-zA-Z ]', '', info[4]).split():
                    if not self.stop_list.__contains__(x):
                        sen2.append(x)
                sen_pair.append(sen1)
                sen_pair.append(sen2)
                # if len(sen1)>1 and len(sen2)>1:
                self.test_data.append([label, sen_pair])
        finally:
            test_file.close()

    def get_train_data_batch(self, k):  # 获取第k个批次的训练数据，数据为通过词向量获取到的每一个单词组合的相似度，矩阵的形式表示
        porter_stemmer = PorterStemmer()
        batch = self.train_data[int(k) * int(self.batch_size):int(k + 1) * int(self.batch_size)]
        labels = []
        feas = []
        for item in batch:
            label_list = [0, 0]
            label_list[item[0]] = 1
            labels.append(label_list)
            fea = []

            for list in item[1]:
                for word1 in list[0]:
                    if self.word2vec_model.__contains__(word1):
                        w1 = porter_stemmer.stem(word1)
                        f = []
                        word1V = self.word2vec_model.get(word1)
                        for word2 in list[1]:
                            if self.word2vec_model.__contains__(word2):
                                word2V = self.word2vec_model.get(word2)
                                w2 = porter_stemmer.stem(word2)
                                if self.wordNet.__contains__(w1):
                                    wnlist = self.wordNet.get(w1)
                                    # print(wnlist)
                                    if w2 in wnlist:
                                        f.append(float(1.0))
                                # f.append(cosSimilar(word1V, word2V))
                                f.append(np.dot(word1V, word2V))
                        while len(f) < 30:
                            f.append(0)
                        fea.append(f)
            while len(fea) < 30:
                fea.append([float(0)] * 30)
            feas.append(reduce(operator.add, fea))
        if int(k + 1) * int(self.batch_size) == len(self.train_data):
            random.shuffle(self.train_data)
        return [labels, feas]

    def get_test_data_batch(self):  # 获取第k个批次的训练数据，数据为通过词向量获取到的每一个单词组合的相似度，矩阵的形式表示
        porter_stemmer = PorterStemmer()
        labels = []
        feas = []
        for item in self.test_data:
            label_list = [0, 0]
            label_list[item[0]] = 1
            labels.append(label_list)
            fea = []
            for list in item[1]:
                if len(list) == 2:
                    for word1 in list[0]:
                        if self.word2vec_model.__contains__(word1):
                            w1 = porter_stemmer.stem(word1)
                            f = []
                            word1V = self.word2vec_model.get(word1)
                            for word2 in list[1]:
                                if self.word2vec_model.__contains__(word2):
                                    word2V = self.word2vec_model.get(word2)
                                    w2 = porter_stemmer.stem(word2)
                                    if self.wordNet.__contains__(w1):
                                        wnlist = self.wordNet.get(w1)
                                        # print(wnlist)
                                        if w2 in wnlist:
                                            f.append(float(1.0))
                                    # f.append(cosSimilar(word1V, word2V))
                                    f.append(np.dot(word1V, word2V))
                            while len(f) < 30:
                                f.append(0)
                            fea.append(f)
            while len(fea) < 30:
                fea.append([float(0)] * 30)
            feas.append(reduce(operator.add, fea))
        return [labels, feas]

    def get_batch_num(self):  # 获取批次总量
        # print('train_data',len(self.train_data))
        # print('batch_size',self.batch_size)
        num = len(self.train_data) / self.batch_size
        return num


def cosSimilar(list1, list2):
    sum = 0
    for key in range(len(list1)):
        sum += list1[key] * list2[key]
    A = sqrt(reduce(lambda x, y: x + y, map(lambda x: x * x, list1)))
    B = sqrt(reduce(lambda x, y: x + y, map(lambda x: x * x, list2)))
    # print(sum/(A * B))
    # print(A)
    # print(B)
    return sum / (A * B)


def eval_matrix(labels, pre):
    return classification_report(labels, pre, digits=5).replace('\n\n', '\n')


def get_label(pre_one):
    if pre_one[0] > pre_one[1]:
        return 0
    else:
        return 1


def weight_variable(shape):  # 生成参数
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # 生成偏置
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):  # 定义卷积层
    # [1，x方向移动步长，y方向移动步长，1]   SAME窗口在边界的操作，SAME是宽卷积方式 VALID是窄卷积
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义max pool层
def max_pool_2x2(x, height, width):
    # ksize [1,2,2,1] 意思同strides
    return tf.nn.max_pool(x, ksize=[1, height, width, 1], strides=[1, 2, 2, 1], padding='SAME')


def k_max_pooling(x, k):
    return tf.nn.top_k(x, k, sorted=False).values


def run_cnn(filter_size=2, top_k=5, full_cell=1024, iteration_num=1000):
    xs_ = tf.placeholder(tf.float32, [None, 30 * 30])  # 输入即为图像的矩阵
    ys = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)
    xs = tf.reshape(xs_, [-1, 30, 30, 1])

    # 卷积层1   卷积核3*3     8通道
    W_conv1 = weight_variable([filter_size, filter_size, 1, 10])
    b_conv1 = bias_variable([10])
    h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1)  # 输出8个
    # print(h_conv1)
    h_conv1_reshape = tf.reshape(h_conv1, [-1, 30, 30])
    # h_pool1 = max_pool_2x2(h_conv1, 2, 2)
    h_pool1 = k_max_pooling(h_conv1_reshape, k=top_k)
    # print(h_pool1)
    h_pool1_reshape2 = tf.reshape(h_pool1, [-1, 30, top_k, 10])

    # LRN对局部神经元的活动创建竞争机制，使得响应比较大的值相对更大，比较小的值就会更小。提高了模型的泛化能力。
    # norm1 = tf.nn.lrn(h_pool1_reshape2, 4, bias=0.1, alpha=0.001 / 9.0, beta=0.75)
    # 对张量进行转置
    xt = tf.transpose(h_pool1_reshape2, perm=[0, 2, 1, 3])
    # xt = tf.transpose(h_pool1_reshape2, perm=[0, 2, 1, 3])
    # print(xt)

    # 卷积层2   卷积核3*3     8通道
    W_conv2 = weight_variable([filter_size, filter_size, 10, 20])
    b_conv2 = bias_variable([20])
    h_conv2 = tf.nn.relu(conv2d(xt, W_conv2) + b_conv2)  # 输出8个
    # print(h_conv2)

    # norm2 = tf.nn.lrn(h_conv2, 4, bias=0.1, alpha=0.001 / 9.0, beta=0.75)
    h_conv2_reshape = tf.reshape(h_conv2, [-1, top_k, 30])
    # h_pool2 = max_pool_2x2(h_conv2, 2, 2)
    h_pool2 = k_max_pooling(h_conv2_reshape, top_k)

    ## 全连接层1 ##
    h_pool2_flat = tf.reshape(h_pool2, [-1, top_k * top_k * 20])
    W_fc1 = weight_variable([top_k * top_k * 20, full_cell])
    b_fc1 = bias_variable([full_cell])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # dropout层 随机丢弃一部分节点来减轻过拟合
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # keep_prob是保留概率

    ## 全连接层2 ———softmax层##
    W_fc2 = weight_variable([full_cell, 2])
    b_fc2 = bias_variable([2])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    print(prediction)

    # 损失函数
    # cross_entropy = tf.reduce_mean(
    #     -tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)), reduction_indices=[1]))
    cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(1e-5)(W_fc1))
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(1e-5)(W_fc2))
    tf.add_to_collection("losses", cross_entropy)
    # #
    loss = tf.add_n(tf.get_collection("losses"))
    # 训练目标
    # 学习速率为1e-4
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
    # print('cross_entropy:', loss)
    # print('loss:', loss)

    ##加载数据
    data = Datas(2)
    data.read_stopword('D:\\tianliuyang\\D\\MSRP_2018\\data\\stoplist.dft')
    data.read_wordNet(r'D:\tianliuyang\D\MSRP_2018\data\wordnet_pan12.voc')
    data.read_word2vec_model(r'D:\tianliuyang\D\MSRP_2018\data\wiki_msrp100.vec')
    data.read_train_data(r'D:\tianliuyang\E\Pan12-04\pan2012-test.txt')
    data.read_test_data(r'D:\tianliuyang\E\Pan12-04\pan2012-test.txt')

    batch_num = data.get_batch_num()
    test_y, test_x = data.get_test_data_batch()
    # test_xs = tf.reshape(test_x, [-1, 30, 30, 1])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ##训练
    for i in range(iteration_num):
        # print(batch_num)
        batch_ys, batch_xs = data.get_train_data_batch(i / batch_num)
        # print(batch_xs)
        # result = sess.run(cross_entropy, feed_dict={xs_: batch_xs, ys: batch_ys, keep_prob: 0.5})  # 训练参数
        # print(result)

        if i % 50 == 0:
            total_cross_entropy = sess.run(loss, feed_dict={xs_: batch_xs, ys: batch_ys, keep_prob: 0.5})
            test_loss = sess.run(loss, feed_dict={xs_: test_x, ys: test_y, keep_prob: 0.5})
            print('total_cross_entropy', 'test_loss', i, total_cross_entropy, test_loss)
        sess.run(train_step, feed_dict={xs_: batch_xs, ys: batch_ys, keep_prob: 0.5})  # 训练参数

    pre = sess.run(prediction, feed_dict={xs_: test_x, keep_prob: 0.5})
    print(pre)

    pre_labels = []
    for i in range(0, len(pre)):
        pre_one = pre[i]
        label_one = get_label(pre_one)
        pre_labels.append(label_one)

    test_labels = []
    for i in range(0, len(pre)):
        pre_one = test_y[i]
        label_one = get_label(pre_one)
        test_labels.append(label_one)

    # print result
    print(eval_matrix(test_labels, pre_labels))
    print("filter_size:" + str(filter_size) + " top_k:" + str(top_k) + " full_cell:" + str(
        full_cell) + " iteration_num:" + str(iteration_num), end='\t')


run_cnn(filter_size=5, top_k=10, full_cell=1024, iteration_num=3000)
print('iteration_num, 为什么还是学偏了啊！！！')
