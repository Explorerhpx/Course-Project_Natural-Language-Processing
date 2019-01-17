# Extract features from events(Post) and label them(events)
import json
import codecs
import numpy as np
import jieba
from Post_deal import feature_extract,Dimension
import math
import os
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import pickle
from sklearn.decomposition import PCA


# Training LDA-based topic model
## Construct training set
# path = 'origin_data'
# Files_name = os.listdir(path)
# index_train = np.load('index_train.npy')
# file_train = [Files_name[i] for i in index_train]
# train_set = []
# for file in file_train:
#     if not os.path.isdir(file):
#         with codecs.open(path + "/" + file,'r','utf-8') as load_file:
#             Data = json.load(load_file)
#             train_set.append(Data[0]['text'])
#
# stopwords = codecs.open('stopwords.txt','r','utf-8').readlines()
# stopwords = {w.strip():0 for w in stopwords}
# LDA_train_set = []
# for event in train_set:
#     event = list(jieba.cut(event))
#     LDA_train_set.append([ w for w in event if (w not in stopwords) and w != ' '])
## Construct lexicon(dictionary)
# dictionary = Dictionary(LDA_train_set)
# corpus = [ dictionary.doc2bow(text) for text in LDA_train_set]
#
## LDA training
# lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=18)
## store model
# LDA_model_file = open('LDA_model.pkl','wb')
# pickle.dump(lda,LDA_model_file)

# load LDA model
Lda = open('LDA_model.pkl','rb')
lda = pickle.load(Lda)


N = 70 # count of Time interval
D = Dimension # feature Dimension
Empty = np.zeros([1, D]) # 处理某个时间段没有post的情况

path = "origin_data"
Files_name= os.listdir(path)
Ite = 0
Size = len(Files_name[1960:])
for file in Files_name[1960:]:
    print( '%.2f'%(Ite/Size * 100),'%' )
    Ite += 1
    if not os.path.isdir(file):
        Feature = np.zeros([D, N])
        with codecs.open(path + "/" + file,'r','utf-8') as load_file:
            Data = json.load(load_file)

         # calculater interval and extract feature
        interval = 0
        if (len(Data) == 1):
            pass
             # TODO(Pingxuan Huang): 处理只有一条post的情况
        else:
            interval = math.ceil((Data[-1]['t'] - Data[0]['t']) / N)
            ite = 0
            span = 0  # sign the interval
            length = 0  # post account in each interval
            start = Data[0]['t']
            feature = np.zeros([1, D])

            while (ite < len(Data)):
                if (Data[ite]['t'] < (start + (span + 1) * interval)):
                    feature += feature_extract(Data[ite])
                    ite += 1
                    length += 1
                elif (length < 1):  # this interval have no post
                    Feature[:, span] = Empty
                    span += 1
                else:  # feature in the interval have been fully extracted
                    # feature[-1] = length * length
                    Feature[:, span] = feature / length
                    feature = np.zeros([1, D])
                    length = 0
                    span += 1
            # feature[-1] = length * length
            Feature[:, -1] = feature / length

         # Add 'slope of features' to the Feature matrix
        S = (Feature[:, 1:] - Feature[:, :-1]) / interval
        Feature = np.hstack((Feature, S))
        ## Add LDA-based topic distribution as Feature
        first_post = Data[0]['text']
        doc = list(jieba.cut(first_post))
        doc_bow = lda.id2word.doc2bow(doc)
        doc_lda = np.zeros([1, D])
        topic_distribute = lda[doc_bow]
        for topic in topic_distribute:
           doc_lda[0][topic[0]] = topic[1]
        Feature = np.hstack((Feature, doc_lda.T))
        np.save('DATA/' + file[:-5], Feature)
        del Data

# Labeling
File = open('Weibo.txt','r')
# Construct label dictionary
Label_dic = {}
for ite in File.readlines():
    item = ite.split()
    Label_dic[item[0][4:]] = int(item[1][-1])
File.close()

path = "DATA"
Files_name = os.listdir(path)
label = []
for file in Files_name:
    event = file[:-4]
    label.append(Label_dic[event])
np.save('Label',label)


# PCA
## Load training set
# path = 'DATA'
# Files_name = os.listdir(path)
# index_train = np.load('index_train.npy')
# file_train = [Files_name[i] for i in index_train]
# All_train = np.zeros((index_train.size,200*D))
# for i in range(len(file_train)):
#     file = file_train[i]
#     if not os.path.isdir(file):
#         sample = np.load(path+'/'+file)
#         sample = np.reshape(sample,(1,sample.size))
#         All_train[i] = sample
## Training model
# pca = PCA(n_components = 600)
# pca.fit(All_train)
## Do PCA on all samples
# for file in Files_name:
#     if not os.path.isdir(file):
#         sample = np.load(path+'/'+file)
#         sample = pca.transform(np.reshape(sample,(1,sample.size)))
#         np.save('DATA_PCA/' + file[:-4], sample)


if __name__ == '__main__':
    pass