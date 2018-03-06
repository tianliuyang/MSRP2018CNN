#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
from gensim.models.word2vec import Word2Vec, LineSentence
# from gensim.models.doc2vec import Doc2Vec
# from gensim.models.word2vec import word2vec
# from word2vec import LineSentence
from gensim.models.doc2vec import TaggedLineDocument
# from gensim.models.word2vec import LineSentence
from gensim import utils

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

fileSen = r'..\data\msrp_all.txt'
filegoogle = r"D:\tianliuyang\E\wiki\wikicorp.201004_sen.txt"
filetgge = fileSen + '.wiki100'
sen = fileSen + '.sen100'
file_vec = filetgge + '.senvec100'


def merge():
    sennub = 0
    with open(filetgge, 'w',encoding='utf-8') as fw:
        fwsen = open(sen, 'w',encoding='utf-8')
        with open(fileSen, 'r',encoding='utf-8') as ftr:
            for line in ftr:
                l = line.strip().split('\t')
                fw.write(l[1].strip() + ' ' + l[2].strip() + '\n')
                fwsen.write(l[1].strip() + ' ' + l[2].strip() + '\n')
                sennub += 1
        with open(filegoogle, 'r',encoding='utf-8') as ftr:
            for line in ftr:
                fw.write(line.strip() + '\n')
    return sennub


def train():
    tagged = TaggedLineDocument(filetgge)
    model = Word2Vec(alpha=0.025, min_alpha=0.025, size=50, window=5, min_count=5, workers=8)
    model.build_vocab(tagged)
    for i in range(10):
        model.train(tagged)
        model.alpha -= 0.0002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay

    # model.save_word2vec_format(filetgge+'.model')
    model.save(filetgge + '.model')


def model2file(sennub):
    # model = Doc2Vec.load_word2vec_format(filetgge + '.model')
    model = Word2Vec.load(filetgge + '.model')
    with open(file_vec, 'w') as fiw:
        for i in range(sennub):
            list_vec = model.docvecs[i].tolist()
            for vec in list_vec:
                fiw.write(str(vec) + ' ')
            fiw.write('\n')


def getVec():
    sen_dic = {}
    with open(sen, 'r',encoding='utf-8') as sens:
        senlist = sens.readlines()
    with open(file_vec, 'r',encoding='utf-8') as vecs:
        veclist = vecs.readlines()
    for i in range(len(senlist)):
        sen_dic[senlist[i].strip()] = veclist[i].strip()
    with open(fileSen + '.vec', 'w') as fw:
        with open(fileSen, 'r') as ftr:
            ftr.readline()
            for line in ftr:
                # linen = line.strip() + '\t' + sen_dic[line.strip()] + "\n"
                l = line.strip().split('\t')
                linen = l[1].strip() + ' ' + l[2].strip()

                linen = line.strip() + '\t' + sen_dic[linen] + "\n"
                fw.write(linen)
                fw.flush()


if __name__ == '__main__':
    sennub = merge()
    train()
    model2file(sennub)
    getVec()
