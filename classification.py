import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from keras import backend as K
from keras.models import Model
from keras.utils import to_categorical
import keras
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
import numpy as np
from keras import metrics
import struct
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import getopt
import sys

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
def fc(enco,filters):
    flat = Flatten()(enco)
    den = Dense(filters, activation='relu')(flat)
    out = Dense(10, activation='softmax')(den)
    return out

def classification(input_img,x_train,x_test,y_train,y_test,y_train1,y_test1,listconv,model,epochs,batch_size,filters):
    
    
    autoencoder = keras.models.load_model(model)
    
    half = len(autoencoder.layers)//2
    
    encoder = Model(autoencoder.layers[0].input, autoencoder.layers[half-1].output)
    
    
    x_tr = np.resize(y_train,(48000,10))
    x_te = np.resize(y_test,(12000,10))

    full_model = keras.Model(encoder.layers[0].input, fc(encoder.output,filters))
    
    for layer in full_model.layers[0:-2]:
        layer.trainable = False
    full_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy',precision_m,recall_m,f1_m])
    classify = full_model.fit(x_train, x_tr, epochs=2, batch_size=51, validation_data=(x_test,x_te), verbose=1)
    
    for layer in full_model.layers[0:-2]:
        layer.trainable = True
    full_model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy',precision_m,recall_m,f1_m])
    classify = full_model.fit(x_train, x_tr, epochs=epochs, batch_size=batch_size, validation_data=(x_test,x_te), verbose=1)

    test_eval = full_model.evaluate(x_test, x_te, verbose=0)
    
    
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    
    predicted_classes = full_model.predict(x_test)
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
    
    y_test = np.resize(y_test,(len(x_te),1))
    
    correct = np.where(predicted_classes==y_test)[0]
    incorrect = np.where(predicted_classes!=y_test)[0]
    
    print("Found %d correct labels" % len(correct))
    print("Found %d incorrect labels" % len(incorrect))
    
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(x_test[correct].reshape(28,28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
        plt.tight_layout()
    
    accuracy = classify.history['accuracy']
    val_accuracy = classify.history['val_accuracy']
    loss = classify.history['loss']
    val_loss = classify.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    epochs = range(len(loss))
    plt.plot(epochs, loss,'o',color='blue', label='Training loss')
    plt.plot(epochs, val_loss,'b',color='blue', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
    target_names = ["Class {}".format(i) for i in range(10)]
    print(classification_report(y_test, predicted_classes, target_names=target_names))
    

    
def read_data(filename):
    
    with open(filename,'rb') as f:
        magic, numofdata = struct.unpack('>II',f.read(8))
        i = numofdata
        data = f.read()
        raw = np.array(np.frombuffer(data, dtype='>u1', count=numofdata)) 
    return raw.reshape(i, 1, 1)

def read_data_image(filename):
    with open(filename,'rb') as f:
        magic, numofdata, cols, row = struct.unpack('>IIII',f.read(16))
        i = numofdata
        data = f.read()
        raw = np.array(np.frombuffer(data, dtype='>u1', count=numofdata * cols * row))
    return raw.reshape(i, cols, row)

if __name__ == "__main__":

    x_train = read_data_image(sys.argv[2])
    x_test = read_data_image(sys.argv[6])
    y_tr = read_data(sys.argv[4])
    y_te = read_data(sys.argv[8])
    y_train = y_tr
    y_test = y_te

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train/255.
    x_test = x_test/255.

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    x_train,x_test,y_tr,y_te = train_test_split(x_train,y_train,test_size=0.2,random_state=13)

    listconv = []

    input_img = keras.Input(shape=(28, 28, 1))

    model = sys.argv[-1]

    print('Give number of epochs :')
    epochs = int(input())
    print('Give number of batch size:')
    batch_size=int(input())
    print('Give number of filters for classifier:')
    filters=int(input())

    classification(input_img,x_train,x_test,y_tr,y_te,y_train,y_test,listconv,model,epochs,batch_size,filters)