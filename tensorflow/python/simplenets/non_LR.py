import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler


if __name__=="__main__":
    # Create noisy data
    x_data = np.linspace(-10, 10, num=1000)
    y_data = 0.1*x_data*np.cos(x_data) + 0.1*np.random.normal(size=1000)
    print('Data created successfully')



    # Create the model
    '''model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(units = 1, activation = 'linear', input_shape=(1,)))
    model.add(tf.keras.layers.Dense(units = 32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(units = 16, activation = 'relu'))
    model.add(tf.keras.layers.Dense(units = 1, activation = 'linear'))'''
    input = tf.keras.Input(shape=(1,))
    output = tf.keras.layers.Dense(1, activation='linear')(input)
    output = tf.keras.layers.Dense(32, activation='relu')(output)
    output = tf.keras.layers.Dense(16, activation='relu')(output)
    output = tf.keras.layers.Dense(1, activation='linear')(output)
    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(loss='mse', optimizer="adam")

    # Display the model
    model.summary()

    #for x in range(100):
    # One epoch
    history=model.fit( x_data, y_data, epochs=200, verbose=1)


    """history = model.fit(x_data, y_data, validation_split=0.33, epochs=200, batch_size=10, verbose=0)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()"""


    # Compute the output
    y_predicted = model.predict(x_data)

    # Display the result
    plt.style.use('default')

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.scatter(x_data[::1], y_data[::1], s=2)
    ax1.set_title("data")
    ax2.scatter(x_data[::1], y_data[::1], s=2)
    ax2.plot(x_data, y_predicted, 'r', linewidth=4)
    ax2.set_title("data predection")

    plt.grid()
    plt.ylim(top=1.2)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-1.2)
    plt.show()
    plt.clf()

    #___________________________________________save model _______________________________________________
    #tf.keras.models.save_model(model, '../models/non_LR_model.pb')


    # load model:
    '''m=tf.keras.models.load_model('../models/non_LR_model.pb')
    m.summary()
    p=m.predict(x_data)
    print(p)'''