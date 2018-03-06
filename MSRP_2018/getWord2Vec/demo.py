import re

s = 'TV stations that reach  percent of US households instead of the old  percent'
# # r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0-9]+'
r = '[^a-zA-Z ]'
# print(re.sub(r,'',s).split())
# from multiprocessing import cpu_count
# import multiprocessing
# print(cpu_count())
# print(multiprocessing.cpu_count())
import numpy as np
# from functools import reduce
# from math import sqrt
# list1 = [1,2]
# list2 = [4,3]
#
# def cosSimilar(list1,list2):
#     sum = 0
#     for key in range(len(list1)):
#         sum += list1[key] * list2[key]
#     A = sqrt(reduce(lambda x, y: x + y, map(lambda x: x * x, list1)))
#     B = sqrt(reduce(lambda x, y: x + y, map(lambda x: x * x, list2)))
#     print(sum)
#     print(A)
#     print(B)
#     return sum / (A * B)
#
# score = cosSimilar(list1,list2)
# print(score)



# import tensorflow as tf
# import theano as T
import numpy as np

# def fold_k_max_pooling(x, k):
#     xs = tf.reshape(x, [-1, 4, 4, 1])
#     print('xs:',xs)
#     xss =tf.reshape(xs,[-1,2,4])
#     values = tf.nn.top_k(xss, k, sorted=False).values  # [batch_size, num_filters[1], top_k]
#     return values


# t = [[100, 200, 300, 400, 500, 600, 700, 800], [900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]]
# sess = tf.Session()
# # result = sess.run(fold_k_max_pooling(x=t, k=1))
# result = sess.run(fold_k_max_pooling(x=t, k=2))
# print(result)

# a= [1,2,3]
# b= ['a',2,3]
# c=[]
# c.append(a)
# c.append(b)
# print(c)
# while len(a) < 5:
#     a.append(0)
# print(a)
# temp = temp = [0]*10
# print(temp)


# import operator
# from functools import reduce
# a = [[1,2,3], [4,5,6], [7,8,9],[0,0,0,0]]
# print(reduce(operator.add, a))

# A = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# A = np.array(
#     [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], [[[13, 14, 15], [16, 17, 18]], [[19, 20, 21], [22, 23, 24]]]])
# print(A)
# sess = tf.Session()
# result = sess.run(tf.transpose(A, [0, 2, 1, 3]))
# print(result)

#
# for full_cell in range(1,10,2):
#     print(full_cell)

# from nltk.stem.porter import PorterStemmer
# porter_stemmer = PorterStemmer()
# w = porter_stemmer.stem('Wantes')
# print(w)
# import nltk
# nltk.download()
# from nltk.corpus import wordnet as wn
# ws = wn.synsets("good")
# print(ws)
worddict = {}
s = 'a b c d'
info = s.split()
worddict.setdefault(info[0],info[1:])
if worddict.__contains__('a'):
    di = worddict.get('a')
    print(di)
    if 'b' in di:
        print('good')

