from functools import reduce
from math import sqrt

def cosSimilar(list1,list2):
    sum = 0
    for key in range(len(list1)):
        sum += list1[key] * list2[key]
    A = sqrt(reduce(lambda x, y: x + y, map(lambda x: x * x, list1)))
    B = sqrt(reduce(lambda x, y: x + y, map(lambda x: x * x, list2)))
    # print(sum/(A * B))
    # print(A)
    # print(B)
    return sum / (A * B)
# if __name__ == '__main__':
#     list1 = [1,2]
#     list2 = [2,2]
#     cosSimilar(list1,list2)