import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import datasets

if __name__=='__main__':
    # 0) Prepare data
    X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

    X=tf.convert_to_tensor(X_numpy, dtype=tf.float32)
    Y=tf.convert_to_tensor(y_numpy, dtype=tf.float32)

    # model:
    """ model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,))) 
    """
    input = tf.keras.Input(shape=(1,))
    output = tf.keras.layers.Dense(1, activation=None)(input)
    model = tf.keras.Model(inputs=input, outputs=output)

## Définition des paramètres d'apprentissage
    # On choisit la méthode d'optimisation
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # On compile le graphe en précisant le nom fonction de coût utlisée
    model.compile(sgd, loss='mean_squared_error', metrics=['mean_absolute_error'])

    #summary:
    model.summary()

    # On commence l'apprentissage à proprement parler
    model.fit(X, Y, batch_size=1, epochs=50, shuffle='True')



    predicted = model.predict(X)



    plt.figure(figsize=(6,4))
    plt.plot(X_numpy, y_numpy, 'ro',label="dataset")
    plt.plot(X_numpy, predicted, 'b',label="predected linear function")
    plt.title("linear regression")
    plt.legend()
    plt.grid()
    plt.show()


    #save model .pb
    tf.keras.models.save_model(model, '../models/LR_model.pb')


    #save model .h5
    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    #model.save('models/LR_model_h5.h5')

    #save format tf:
    #model.save('LR_model', save_format='tf')

