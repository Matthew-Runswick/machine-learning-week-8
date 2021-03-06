import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#Part (i)
#Part A
def single_convolution(input, kernel, size):
    result = 0
    for a in range(0, size[0]):
        for  b in range(0, size[1]):
            result = result + input[a][b] * kernel[a][b]
    return result

def return_array_section(array, starting_position, size):
    temp_array = []
    for a in range(starting_position[0], starting_position[0]+size[0]):
        temp_row = []
        for b in range(starting_position[1], starting_position[1]+size[1]):
            temp_row.append(array[a][b])

        temp_array.append(temp_row)
    return temp_array

def convolve(input_array, kernel):
    input_array_size = [len(input_array), len(input_array[0])]
    kernel_size = [len(kernel), len(kernel[0])]
    output_array = []
    current_position = [0, 0]
    for a in range(0, (input_array_size[0]-kernel_size[0])+1):
        new_row = []
        for b in range(0, (input_array_size[1]-kernel_size[1]) + 1):
            temp_input = return_array_section(input_array, current_position, kernel_size)
            new_value = single_convolution(temp_input, kernel, kernel_size)
            new_row.append(new_value)
            current_position[1] = current_position[1] + 1 

        output_array.append(new_row)
        current_position[0] = current_position[0] + 1
        current_position[1] = 0 
    return output_array

#Part B
from PIL import Image
im = Image.open('triangle.png')
rgb = np.array(im.convert('RGB'))
test = rgb[:, :, 0]
Image.fromarray(np.uint8(test)).show()

kernel1 = [[-1,-1,-1], [-1,8,-1], [-1,-1,-1]]
kernel2 = [[0,-1,0], [-1,8,-1], [0,-1,0]]

kernel1_result = convolve(test, kernel1)
kernel2_result = convolve(test, kernel2)

Image.fromarray(np.uint8(kernel1_result)).show()
Image.fromarray(np.uint8(kernel2_result)).show()

#Part (ii)  (imported code)
#Part B
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True
import sys

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

def prep_data(number_of_datapoints):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    n=number_of_datapoints
    x_train = x_train[1:n]; y_train=y_train[1:n]
    #x_test=x_test[1:500]; y_test=y_test[1:500]

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("orig x_train shape:", x_train.shape)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def conv_model_training(x_train, y_train, x_test, y_test, title, penalty, architecture_type):
    start = time.time()
    use_saved_model = False
    if use_saved_model:
        model = keras.models.load_model("cifar.model")
    else:
        if(architecture_type == "stride"):
            model = keras.Sequential()
            model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
            model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
            model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
            model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(penalty)))
            model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
            model.summary()

        else:
            model = keras.Sequential()
            model.add(Conv2D(16, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(penalty)))
            model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
            model.summary()

        batch_size = 128
        epochs = 20
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        model.save("cifar.model")

        end = time.time()
        print("!!!!!!!!!!!! time to complete training", round(end - start), "s  !!!!!!!!!!!!!!!!!!!!!!!!!")

        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('{} - model accuracy'.format(title))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('{} - model loss'.format(title))
        plt.ylabel('loss'); plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    preds = model.predict(x_train)
    y_pred = np.argmax(preds, axis=1)
    y_train1 = np.argmax(y_train, axis=1)
    print(classification_report(y_train1, y_pred))
    print(confusion_matrix(y_train1,y_pred))

    preds = model.predict(x_test)
    y_pred = np.argmax(preds, axis=1)
    y_test1 = np.argmax(y_test, axis=1)
    print(classification_report(y_test1, y_pred))
    print(confusion_matrix(y_test1,y_pred))

    

x_train, y_train, x_test, y_test = prep_data(5000)
conv_model_training(x_train, y_train, x_test, y_test, "5K", 0.0001, "stride")

# Part B (iii)
x_train, y_train, x_test, y_test = prep_data(10000)
conv_model_training(x_train, y_train, x_test, y_test, "10K", 0.0001, "stride")

x_train, y_train, x_test, y_test = prep_data(20000)
conv_model_training(x_train, y_train, x_test, y_test, "20K", 0.0001, "stride")

x_train, y_train, x_test, y_test = prep_data(40000)
conv_model_training(x_train, y_train, x_test, y_test, "40K", 0.0001, "stride")

# part B (iv)
x_train, y_train, x_test, y_test = prep_data(5000)

conv_model_training(x_train, y_train, x_test, y_test, "5K L=0", 0, "stride")
conv_model_training(x_train, y_train, x_test, y_test, "5K L=0.00001 ", 0.00001, "stride")
conv_model_training(x_train, y_train, x_test, y_test, "5K L=0.0001", 0.0001, "stride")
conv_model_training(x_train, y_train, x_test, y_test, "5K L=0.001", 0.001, "stride")
conv_model_training(x_train, y_train, x_test, y_test, "5K L=0.01 ", 0.01, "stride")
conv_model_training(x_train, y_train, x_test, y_test, "5K L=1 ", 1, "stride")

#Part C (i) and (ii)
conv_model_training(x_train, y_train, x_test, y_test, "5K max Pooling", 0.0001, "maxPooling")


#part D (optional)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

start = time.time()
use_saved_model = False
if use_saved_model:
    model = keras.models.load_model("cifar.model")
else:
    model = keras.Sequential()
    model.add(Conv2D(8, (3,3), padding='same', input_shape=x_train.shape[1:],activation='relu'))
    model.add(Conv2D(8, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3,3), strides=(2,2), padding='same', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax',kernel_regularizer=regularizers.l1(0.0001)))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()

    batch_size = 128
    epochs = 40
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.save("cifar.model")

    end = time.time()
    print("!!!!!!!!!!!! time to complete training", round(end - start), "s  !!!!!!!!!!!!!!!!!!!!!!!!!")

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('deeper and wider model - model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('deeper and wider model - model loss')
    plt.ylabel('loss'); plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

preds = model.predict(x_train)
y_pred = np.argmax(preds, axis=1)
y_train1 = np.argmax(y_train, axis=1)
print(classification_report(y_train1, y_pred))
print(confusion_matrix(y_train1,y_pred))

preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)
y_test1 = np.argmax(y_test, axis=1)
print(classification_report(y_test1, y_pred))
print(confusion_matrix(y_test1,y_pred))