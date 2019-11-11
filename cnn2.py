from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras import optimizers
from mnist import MNIST
from keras.models import load_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd


mndata = MNIST('source')
(x_train, y_train) = mndata.load_training()
(x_test, y_test) = mndata.load_testing()

x_train = np.asarray(x_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.int32)
x_test = np.asarray(x_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.int32)


# x_train = tf.reshape(x_train, shape=[-1, 28, 28, 1])
# x_test = tf.reshape(x_test, shape=[-1, 28, 28, 1])
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train/255.0
x_test = x_test/255.0
print(x_train.shape)
print(x_test.shape)
print((y_train).shape)
print((y_test).shape)
print(type(x_train))


batch_size = 64

epochs = 10
lrate = 0.0001
dpout = 0.3
no_hidd_neurons = 128
hyp_param = {'batch_size': batch_size, 'epochs': epochs, 'no_neuron': no_hidd_neurons,
             'lrate': lrate, 'dpout': dpout, 'optimizer': 'adam', 'act_func': 'sigmoid', 'hidden_act': 'relu'}

# input image dimensions


def create_model(my_dict):
    # Importing the required Keras modules containing model and layers
    # Creating a Sequential Model and adding the layers
    img_rows, img_cols = 28, 28
    num_classes = 10
    input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    print('Add first conv networks')
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print('Added first max pooled')
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    print('Add second conv networks')
    model.add(MaxPooling2D(pool_size=(2, 2)))
    print('Added second max pooled')
    model.add(Flatten())  # Flattening the 2D arrays for fully connected layers so that tf.nn.relu
    model.add(Dense(my_dict['no_neuron'], activation=my_dict['hidden_act']))
    model.add(Dropout(my_dict['dpout']))  # fraction of input layers to drop
    model.add(Dense(10, activation=my_dict['act_func']))
    adam = optimizers.adam(lr=my_dict['lrate'])
    model.compile(optimizer=my_dict['optimizer'],
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def plot_error(history):
    plt.figure(figsize=(16, 10))
    plt.plot(history.epoch, history.history['val_loss'],
             '--', label='Validation Loss')
    plt.plot(history.epoch, history.history['loss'],
             '--', label='Training Loss')
    plt.title('Training Error vs Generalization Error')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.xlim([0, max(history.epoch)])
    plt.figure(figsize=(16, 10))
    plt.plot(history.epoch, history.history['val_accuracy'],
             '--', label='Validation Accuracy')
    plt.plot(history.epoch, history.history['accuracy'],
             '--', label='Training Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.legend()
    plt.xlim([0, max(history.epoch)])


def export_results():
    model = load_model('Batch Size = 64learning_rate = 0.0001trained_model.h5')
    # model = load_model('best_model2.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    predictions = pd.DataFrame(model.predict(x_test))
    index = predictions.idxmax(axis=1)
    index = index.values
    predictions.loc[:] = 0
    print(index)
    for i in range(len((index))):
        predictions.loc[i, index[i]] = 1
    predictions = predictions.astype(int)
    print(predictions)
    export_csv = predictions.to_csv('mnist.csv', index=None, header=False)


def save_hp(history, my_dict):

    row = [my_dict['batch_size'], my_dict['lrate'], my_dict['dpout'], my_dict['act_func'], my_dict['no_neuron'], history.history['val_loss'],
           history.history['val_accuracy'], history.history['loss'], history.history['accuracy'], my_dict['hidden_act']]
    with open('hp_file.csv', mode='a') as writeFile:
        hyp_writer = csv.writer(writeFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        hyp_writer.writerow(row)
    #    readFile.close()
       # writeFile.close()

    # with open('hp_file.csv', mode='w') as employee_file:
    #     hyp_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     hyp_writer.writerow(['batch_size', 'Learning_rate', 'Dropout', 'Activation_function', 'Number_hidden_layer_neurons', 'Validation_loss', 'Validation_accuracy', 'Training_loss', 'Training_accuracy' ])
    #     hyp_writer.writerow(row)


model = create_model(hyp_param)
# , validation_split=0.3
history = model.fit(x=x_train, y=y_train,
                    epochs=hyp_param['epochs'], batch_size=hyp_param['batch_size'])
model.save('Batch Size = '+str(hyp_param['batch_size']) +
           'learning_rate = '+str(hyp_param['lrate'])+'trained_model.h5')
# print('\nhistory dict:', history)
# save_hp(history, hyp_param)
# plot_error(history)

export_results()
