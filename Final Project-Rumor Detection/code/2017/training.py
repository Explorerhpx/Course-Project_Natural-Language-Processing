import json
import numpy as np
import os
from sklearn import svm
import random
import matplotlib.pyplot as plt

def sumMatrix(path, size):
    result_files = os.listdir(path)
    array = np.zeros((size,size))
    for (index, filename) in enumerate(result_files):
        with open(os.path.join(path, filename)) as fp:
            array = array + np.array(json.loads(fp.read()))

    return array[0:size-1,:][0:size-1,:]

def lookupTarget(path):

    files = os.listdir(path)
    result_dic = {}  ##Lookuptable between Weibo.txt and filename order
    result_list = [0 for i in range(len(files))]

    for index in range(len(files)):
        result_dic[files[index].split('.')[0]] = index

    with open('Weibo.txt') as fp:
        s = fp.read()
        target_list = s.split('\n')[0:-1]

        for target in target_list:

            target_id = target.split('\t')[0].split(':')[1]

            if result_dic.get(target_id):
                index = result_dic[target_id]  ##Lookup table
                label = int(target.split('\t')[1].split(':')[1])
                result_list[index] = label
            else:
                continue

    return result_list

def svmWrapper(matrix, result_list,alpha):
    clf = svm.SVC()  # class
    partition = int(alpha * len(result_list))
    clf.fit(matrix[0:partition],result_list[0:partition])  # training the svc model

    result = clf.predict(matrix[partition:])  # predict the target of testing samples
    print(result)
    print(result_list[partition:])  # target

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for index in range(len(result)):
        item1 = result[index]
        item2 = result_list[int(alpha * len(matrix)):][index]
        if item1 == 1 and item2 == 1:
            true_positive += 1
            continue
        if item1 == 0 and item2 == 0:
            true_negative += 1
            continue
        if item1 == 1 and item2 == 0:
            false_positive += 1
            continue
        if item1 == 0 and item2 == 1:
            false_negative += 1
            continue

    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(false_negative+true_positive)
    F1 = 2*(precision*recall)/(precision+recall)
    print(true_positive,false_negative,true_negative,false_positive)
    print(precision)
    print(recall)
    print(F1)

    return precision

def knnPredict(cost_matrix, i, shard, result_list):
    count_positive = 0
    count_negative = 0
    means_positive = 0
    means_negative = 1
    for j in range(0,shard):
        if result_list[j] == 1:
            count_positive += 1
            means_positive += cost_matrix[i][j]
        else:
            count_negative += 1
            means_negative += cost_matrix[i][j]

    means_positive = means_positive / count_positive
    means_negative = means_negative / count_negative 
    return means_positive > means_negative



def knnPredictWrapper(cost_matrix, result_list, partition_ratio):
    length =len(result_list)
    shard = int(length * partition_ratio)

    for i in range(shard+1, len(result_list)):
        prdict = knnPredict(cost_matrix, i, shard, result_list)
        if predict == result_list[i]:
            count += 1

    return (count / (len - shard))




def drop_pic(interval, cost_matrix_ptk, cost_matrix_cptk, result_list, partition_ratio):
    num_of_points = int(len(result_list)/interval)

    ptk_precisions = []
    cptk_precisions = []
    knn_precisions = []

    for index in range(num_of_points):
        ptk_precisions[index] = svmWrapper(cost_matrix_ptk(index:index+interval,:), result_list(index:index+interval,:), partition_ratio)
        cptk_precisions[index] = svmWrapper(cost_matrix_cptk(index:index+interval,:), result_list(index:index+interval,:), partition_ratio)
        knn_precisions[index] = knnPredictWrapper(cost_matrix_cptk(index:index+interval,:), result_list(index:index+interval,:), partition_ratio)

    plt.plot(range(num_of_points), ptk_precisions)
    plt.plot(range(num_of_points), cptk_precisions)
    plt.plot(range(num_of_points), knn_precisions)

    plt.legend(['ptk', 'cptk', 'knn-like'], loc='upper left')


def main():
    partition_ratio = 0.55

    result_list = lookupTarget('data')
    cost_matrix = sumMatrix('result', len(result_list))

    drop_pic(1,cost_matrix, cost_matrix, result_list, partition_ratio)
    

    ##svmWrapper(cost_matrix, result_list, partition_ratio)


if __name__ == '__main__':
    main()
