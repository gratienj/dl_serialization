import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import datasets

if __name__=="__main__":
    #______________________________________________ data preparation _____________________________________________
    X, Y = datasets.make_regression(n_samples=200, n_features=2, noise=10, random_state=10)

    x = X[:, 0]
    y = X[:, 1]
    z = Y

    print("x: ",min(x),max(x))
    print("y: ",min(y),max(y))

    x_pred = np.linspace(min(x), max(x), 30)      # range of porosity values
    y_pred = np.linspace(min(y), max(y), 30)  # range of VR values

    xx_pred, yy_pred = np.meshgrid(x_pred, y_pred)
    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T

    X_tensor=tf.convert_to_tensor(X,dtype=tf.float32)
    Y_tensor=tf.convert_to_tensor(Y,dtype=tf.float32)
    model_viz_tensor=tf.convert_to_tensor(model_viz, dtype=tf.float32)

    #_________________________________________________Model________________________________________________
    """model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(2,)))"""
    input = tf.keras.Input(shape=(2,))
    output = tf.keras.layers.Dense(1, activation=None)(input)
    model = tf.keras.Model(inputs=input, outputs=output)


    ## Définition des paramètres d'apprentissage
    # On choisit la méthode d'optimisation
    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # On compile le graphe en précisant le nom fonction de coût utlisée
    model.compile(sgd, loss='mean_squared_error', metrics=['mean_absolute_error'])

    #summary:
    model.summary()

    #_______________________________________________train____________________________________________________

    # On commence l'apprentissage à proprement parler
    model.fit(X, Y, batch_size=1, epochs=50, shuffle='True')


    #_______________________________________________predict________________________________________________
    predicted = model.predict(model_viz_tensor)

    points=tf.constant([[0,1],[0.5,1.5]])
    predicted_points=model.predict(points)
    #______________________________________________ Plot -----------------------------------------------_____

    plt.style.use('default')

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    axes = [ax1, ax2, ax3]

    for ax in axes:
        ax.plot(x, y, z, color='g', zorder=15, linestyle='none', marker='o', alpha=0.5)
        ax.scatter(xx_pred.flatten(), yy_pred.flatten(), predicted, facecolor=(0,0,0,0), s=20, edgecolor='#70b3f0')
        ax.set_xlabel('param 1', fontsize=12)
        ax.set_ylabel('param 2', fontsize=12)
        ax.set_zlabel('sortie', fontsize=12)
        ax.locator_params(nbins=4, axis='x')
        ax.locator_params(nbins=5, axis='x')

        # prédiction d'un point:
        ax.scatter(points[0],points[1],predicted_points,color='r' ,marker='o',)

    ax1.view_init(elev=27, azim=112)
    ax2.view_init(elev=16, azim=-51)
    ax3.view_init(elev=60, azim=165)

    plt.show()


    #___________________________________________save model _______________________________________________
    tf.keras.models.save_model(model, '../models/MLR_model.pb')
    #save model .h5
    # Save the entire model to a HDF5 file.
    # The '.h5' extension indicates that the model should be saved to HDF5.
    #model.save('models/MRL_model_h5.h5')
