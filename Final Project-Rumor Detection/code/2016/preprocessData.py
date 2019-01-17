import datetime
import json
import re
import logging
import os
import pickle
import numpy as np
import io

from gensim import corpora, utils, models
from gensim.corpora import MmCorpus
from dateutil import parser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


CORPUS_FILE_PATH = r'../resources/rumorCorpus.mm'
DICTIONARY_FILE_PATH = r'../resources/rumorDictionary.pkl'
DATA_PATH = r'../rumor/twitter_json'
RUMOR_TF_INPUTPICKLE = r'../resources/tensorInput.pkl'
TEST_SET_FILE_PATH = r'../rumor/testSet_twitter.txt' 
TRAIN_SET_FILE_PATH = r'../rumor/trainSet_twitter.txt' 
TWITTER_LABEL_PATH = r'../rumor/twitter_label.txt' 

# CORPUS_FILE_PATH = r'../resources/rumorCorpus.mm'
# DICTIONARY_FILE_PATH = r'../resources/rumorDictionary.pkl'
# DATA_PATH = r'../rumor/weibo_json'
# RUMOR_TF_INPUTPICKLE = r'../resources/tensorInput.pkl'
# TEST_SET_FILE_PATH = r'../rumor/testSet_weibo.txt' # given file
# TRAIN_SET_FILE_PATH = r'../rumor/trainSet_weibo.txt' # given file
# TWITTER_LABEL_PATH = r'../rumor/weibo_label.txt' # given file

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100


def main():
    createInputKeras(RUMOR_TF_INPUTPICKLE)
    # loadInput(RUMOR_TF_INPUTPICKLE)
    # createInput(RUMOR_TF_INPUTPICKLE)
    E = getSequenceFromFile('../rumor/weibo_json/9624944661.json')
    time_series = variable_length_time_series(E, N=20)
    test = create_tfidfs(time_series)
    print('Done preprocessing.')


def get_complete_tfidf(corpus, dictionary):
    tfidf = models.TfidfModel(corpus)
    num_unique = len(dictionary)
    input = tfidf[corpus]
    output = []
    for idx, doc in enumerate(input):
        print(doc)
        wordIDs = [t[0] for t in doc]
        word_tfidf = [t[1] for t in doc]
        added = []
        for i in list(range(num_unique)):
            if i not in wordIDs:
                added.append((i, 0))
        doc.extend(added)
        doc.sort(key=lambda tup: tup[0])
        output.append(doc)
    return output
def loadLabels(labelPath):
    LABEL_DICT = {}
    with open(labelPath) as infile:
        for line in infile:
            line = line.split('\t')
            print(line)
            key = line[1].split(':')[1]
            value = line[0].split(':')[1]
            key = key.strip('\n')
            LABEL_DICT[key] = value

    return LABEL_DICT

def loadInput(inputFile):
    with open(inputFile, "rb") as input:
        print('loading input\n')
        outputList = pickle.load(input)
        X_train, y_train = outputList

        outputList = pickle.load(input)
        X_test, y_test = outputList

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        y_test = np.asarray(y_test)

        return X_train, y_train, X_test, y_test


def createInput(inputFile):
    trainSetList = set()
    testSetList = set()

    with open(TRAIN_SET_FILE_PATH, 'rb') as f:
        for line in f:
            trainSetList.add(line.rstrip())

    with open(TEST_SET_FILE_PATH, 'rb') as f:
        for line in f:
            testSetList.add(line.rstrip())

    if (os.path.exists(CORPUS_FILE_PATH)):
        corpus = corpora.MmCorpus(CORPUS_FILE_PATH)
    else:
        corpus = createCorpus(DATA_PATH)

    if (os.path.exists(DICTIONARY_FILE_PATH)):
        dictionary = loadDictionary(DICTIONARY_FILE_PATH)

    with open(inputFile, "wb") as output:
        LABEL_DICT = loadLabels(TWITTER_LABEL_PATH)
        trainOutputList = []
        testOutputList = []

        dictionary.filter_n_most_frequent(10)

        tfidf = get_complete_tfidf(corpus, dictionary)
        tfidf = [[t[1] for t in l] for l in tfidf]

        for idx, fname in enumerate(os.listdir(DATA_PATH)):
            if bytes(fname, encoding='utf-8') in trainSetList:
                line = (tfidf[idx], int(LABEL_DICT[fname.split('.')[0]]))
                trainOutputList.append(line)
            else:
                line = (tfidf[idx], int(LABEL_DICT[fname.split('.')[0]]))
                testOutputList.append(line)

        pickle.dump(trainOutputList, output)
        pickle.dump(testOutputList, output)

def createInputKeras(inputFile):
    X_train = []
    y_train = []

    X_test = []
    y_test = []
    LABEL_DICT = loadLabels(TWITTER_LABEL_PATH)
    with open(TRAIN_SET_FILE_PATH, 'r') as f:
        for line in f:
            file_path = os.path.join(DATA_PATH, line.rstrip())
            X_train.append(getTextFromFile(file_path))
            y_train.append(int(LABEL_DICT[line.rstrip().split('.')[0]]))
    with open(TEST_SET_FILE_PATH, 'r') as f:
        for line in f:
            file_path = os.path.join(DATA_PATH, line.rstrip())
            X_test.append(getTextFromFile(file_path))
            y_test.append(int(LABEL_DICT[line.rstrip().split('.')[0]]))
    tokenizer_train = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer_test = Tokenizer(nb_words=MAX_NB_WORDS)

    tokenizer_train.fit_on_texts(X_train)
    tokenizer_test.fit_on_texts(X_test)

    sequences_train = tokenizer_train.texts_to_sequences(X_train)
    sequences_test = tokenizer_test.texts_to_sequences(X_test)

    word_index = tokenizer_train.word_index
    print('Found %s unique train tokens.' % len(word_index))

    X_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    X_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of data_train tensor:', X_train.shape)

    with open(inputFile, "wb") as output:
        pickle.dump((X_train, y_train), output)
        pickle.dump((X_test, y_test), output)

def create_tfidfs(time_series):
    tfidf_series = []
    for interval in time_series:
        document = []
        for post in interval:
            document.append(post[0])
        word_count_vector = CountVectorizer().fit_transform(document)
        tfidf_transformer = TfidfTransformer()
        tfidf_vector = tfidf_transformer.fit_transform(word_count_vector)
        tfidf_series.append(tfidf_vector)
    return tfidf_series

def variable_length_time_series(E, N):
    L = (E[-1][1] - E[0][1]).total_seconds() // 60
    l = L // N
    l = int(l)
    k = 0
    U_hat_prev = []
    while True:
        k = k + 1
        U_k = Equipartition(E, N, l)
        U_hat_k = find_longest_time_span(U_k, l)
        if(len(U_hat_k) < N and len(U_hat_k) > len(U_hat_prev)):
            l = l // 2
            U_hat_prev = U_hat_k
        else:
            return U_hat_k

def Equipartition(E, N, l):
    U_k = []
    for i in range(N):
        list = []
        for idx, rp in enumerate(E):
            left_most = datetime.timedelta(minutes = i*l) + E[0][1]
            right_most = datetime.timedelta(minutes = (i+1)*l) + E[0][1]
            if(left_most <= rp[1] and rp[1] < right_most):
                list.append(rp)
        U_k.append(list)
    return U_k

def find_longest_time_span(U_k, l):
    U_hat_k = []
    max_time_span_index = 0
    max_count = 0
    count = 0
    for idx, time_interval in enumerate(U_k):
        if time_interval:
            count = count + 1
        else:
            if count > max_count:
                max_count = count
                max_time_span_index = idx - count
            count = 0

    for i in range(max_time_span_index, max_time_span_index + max_count):
        U_hat_k.append(U_k[i])

    return U_hat_k

def createCorpus(inputPath):
    corpus = RumorTextCorpus(inputPath)
    MmCorpus.serialize(CORPUS_FILE_PATH, corpus)
    corpus.dictionary.save(DICTIONARY_FILE_PATH)
    return corpus

def getSequenceFromFile(file_path):
    E = []
    for line in open(file_path):
        print(line)
        line = re.sub(' "source":(.[^,]+)",', '', line)  # remove json.loads corrupters
        line = re.sub("(?<=[^a-z])(')(?=.)|(?<=.)(')(?=[^a-z])", '"', line)
        jsonObject = json.loads(line.strip('\n'))
        post = jsonObject['text']
        timestamp = jsonObject['t']
        timestamp = parser.parse(timestamp)
        e = (post, timestamp)
        E.append(e)
    E.sort(key=lambda x: x[1])
    return E

# def getSequenceFromFile(file_path):
#     E = []
#     f = open(file_path,'rb')
#     jsonObject = json.load(f)
#     for each in jsonObject:
#         post = each['text']
#         timestamp = each['t']
#         timestamp = parser.parse(timestamp)
#         e = (post, timestamp)
#         E.append(e)
#     E.sort(key=lambda x: x[1])
#     return E
def getTextFromFile(file_path):
    W = ''
    for line in open(file_path):
        line = re.sub(' "source":(.[^,]+)",', '', line)  # remove json.loads corrupters
        line = re.sub("(?<=[^a-z])(')(?=.)|(?<=.)(')(?=[^a-z])", '"', line)
        jsonObject = json.loads(line.strip('\n'))
        w = jsonObject['text']
        W = W + ' ' + w
    W = W[1:]
    return W

# def getTextFromFile(file_path):
#     W = ''
#     f = open(file_path,'rb')
#     jsonObject = json.load(f)
#     for each in jsonObject:
#         W = W + ' ' + each['text']
#     W = W[1:]
#     return W

def loadDictionary(dictionaryPath):
    return corpora.Dictionary.load(dictionaryPath)

def loadCorpus(corpusPath):
    return MmCorpus(corpusPath)

class RumorTextCorpus(corpora.TextCorpus):

    def __init__(self, dirname):
        self.dirname = dirname
        super(RumorTextCorpus, self).__init__(dirname)

    def get_texts(self):
        stoplist = set('for a of the and to in'.split()) # add http?
        for fname in os.listdir(self.dirname):
            W = []
            print(os.path.join(self.dirname, fname))
            for line in io.open(os.path.join(self.dirname, fname), 'r', encoding='windows-1252'):
                line = re.sub(' "source":(.[^,]+)",', '', line)  # remove json.loads corrupters
                line = re.sub("(?<=[^a-z])(')(?=.)|(?<=.)(')(?=[^a-z])", '"', line)
                w = json.loads(line)

                # tokenize and remove common words
                w = utils.tokenize(w['text'], lowercase=True)
                w = [word for word in w if word not in stoplist]

                W.extend(w)
            yield W
    # def get_texts(self):
    #     stoplist = set('for a of the and to in'.split()) # add http?
    #     for fname in os.listdir(self.dirname):
    #         W = []
    #         print(os.path.join(self.dirname, fname))
    #         f = open(os.path.join(self.dirname, fname),'rb')
    #         jsonObject = json.load(f)
    #         for w in jsonObject:
    #             # tokenize and remove common words
    #             w = utils.tokenize(w['text'], lowercase=True)
    #             w = [word for word in w if word not in stoplist]
    #             W.extend(w)
    #         yield W

if __name__ == "__main__": main()


