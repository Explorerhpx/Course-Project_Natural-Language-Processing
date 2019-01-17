# Process database and Train classifiers basing on extracted features
import numpy as np
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn import tree
import sklearn
import os
import pickle
import random
import time

Label = np.load('Label.npy')

## Construct training set/ development set/ test set
# Divide training set/test set at the first training process
# sample_account = len(Label)
# index = [i for i in range(sample_account)]
# random.shuffle(index)
## store order of sample
# gap = int(np.ceil(0.8*sample_account))
# print(index[gap:])
# np.save('index_train',index[:gap])
# np.save('index_test',index[gap:])

# load related data
index_test = np.load('index_test.npy')
index_train = np.load('index_train.npy')
random.shuffle(index_train)
path = "DATA"
Files_name = os.listdir(path)

Train = [];Dev = [];Test = []
Proporation = 1 # Proportion of (Train +Validation)
# Train set
for i in index_train[:int(np.ceil(0.7 * Proporation*len(index_train)))]: # 70% of Training samples in train set
    file = Files_name[i]
    data = np.load(path+'/'+file)
    data = data.reshape([1,data.size])
    Train.append(data[0])
# Test set
for i in index_test:
    file = Files_name[i]
    data = np.load(path+'/'+file)
    data = data.reshape([1, data.size])
    Test.append(data[0])
# Validation set
for i in index_train[int(np.ceil(0.7 * Proporation*len(index_train))):\
        int(np.ceil(Proporation*len(index_train)))]: # 30% of Training samples in validation set
    file = Files_name[i]
    data = np.load(path+'/'+file)
    data = data.reshape([1,data.size])
    Dev.append(data[0])
# Label
Y_train = [Label[i] for i in index_train\
    [:int(np.ceil(0.7 * Proporation*len(index_train)))]]
Y_dev = [Label[i] for i in index_train[int(np.ceil(0.7 * Proporation*len(index_train))):\
    int(np.ceil(Proporation*len(index_train)))]]
Y_test = [Label[i] for i in index_test]

# Training model
# clf = SVC(C=100, kernel='rbf', gamma=10, decision_function_shape='ovo')
# clf = SVC(C=1e-5, kernel='linear', decision_function_shape='ovo')
# knn = neighbors.KNeighborsClassifier()
DT = tree.DecisionTreeClassifier()

start = time.clock()
# clf.fit(Train, Y_train)
# knn.fit(Train,Y_train)
DT.fit(Train,Y_train)
elapsed = (time.clock() - start)
print("Time used: %.1fs:"%elapsed) # Training time

# load trained model
#Clf = open('model.pkl','rb')
#clf = pickle.load(Clf)

# store model
# Model_file = open('model.pkl','wb')
# pickle.dump(clf,Model_file)

# Test model
# predict = clf.predict(Test)
# predict = knn.predict(Test)
predict = DT.predict(Test)
precision = sklearn.metrics.precision_score(Y_test, predict)
recall = sklearn.metrics.recall_score(Y_test, predict)
accuracy = sklearn.metrics.accuracy_score(Y_test, predict)
print('precision: %.2f%%\nrecall: %.2f%%' % (100 * precision, 100 * recall))
print('accuracy: %.2f%%' % (100 * accuracy))
