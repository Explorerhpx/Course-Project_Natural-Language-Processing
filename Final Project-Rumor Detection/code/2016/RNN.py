from __future__ import division, print_function, absolute_import

from keras import optimizers
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Embedding, Dense, PReLU, Dropout, regularizers
from keras.layers import LSTM,Bidirectional
from keras.models import load_model, Sequential
from keras.utils import plot_model
from tflearn.data_utils import to_categorical

from preprocessData import *

MODEL_PATH = r'../weight_adam/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
MODEL_PATH_SGD = r'../weight_sgd/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
MODEL_PATH_ADAGRAD = r'../weight_adagrad/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.hdf5'
INPUT_DIM = 200
OUTPUT_DIM = 2
NUM_EPOCHS = 20
NUM_CLASSES = 2
RUMOR_TF_INPUTPICKLE = r'../resources/tensorInput.pkl'
def main():
    trainEmbedding()

def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=20000, output_dim=EMBEDDING_DIM))
    model.add(Dropout(0.4))
    model.add(LSTM(100))
    # model.add(Bidirectional(LSTM(100)))
    # model.add(GRU(100))
    model.add(Dropout(0.4))

    model.add(Dense(784, kernel_regularizer=regularizers.l2(0.07)))
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(784, kernel_regularizer=regularizers.l2(0.07)))
    model.add(PReLU())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))
    return model

def trainEmbedding():
    X_train, y_train, X_test, y_test = loadInput(RUMOR_TF_INPUTPICKLE)

    y_train = to_categorical(y_train, nb_classes=2)
    y_test = to_categorical(y_test, nb_classes=2)
    print('X_train: ', X_train)
    print('X_test: ', X_test)
    model = build_model()

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adagrad = optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-8, decay=0.)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_checkpoint = ModelCheckpoint(MODEL_PATH_ADAGRAD, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, callbacks=[tensor_board, model_checkpoint])
    model.evaluate(X_test, y_test, show_accuracy = True)

if __name__ == "__main__": main()
