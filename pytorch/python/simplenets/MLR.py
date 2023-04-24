import torch
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


#----------------------------------------------------model--------------------------------------------

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize=2, outputSize=1):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out



if __name__=='__main__':
    #---------------------------------------------------data -------------------------------------------
    X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=2, noise=20, random_state=4)

    x = X_numpy[:, 0]
    y = X_numpy[:, 1]
    z = y_numpy


    print(x.shape, y.shape, z.shape)

    X = torch.from_numpy(X_numpy.astype(np.float32))
    Y = torch.from_numpy(y_numpy.astype(np.float32))
    Y = Y.view(Y.shape[0], 1)

    print("shapes: ",X.shape, Y.shape)

    line_x = np.linspace(np.min(x), np.max(x), 30)
    line_y = np.linspace(np.min(y), np.max(y), 30)
    xx_pred,yy_pred= np.meshgrid(line_x, line_y)

    model_viz = np.array([xx_pred.flatten(), yy_pred.flatten()]).T


    # cast to float Tensor
    model_viz_tensor=torch.from_numpy(model_viz.astype(np.float32))

    #---------------------------------------------------train---------------------------------------------
    #instance du model:
    model=linearRegression()

    #loss function:
    criterion=torch.nn.MSELoss()

    #optimizer:
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

    # 10 epochs:
    for i in range(1000):
        #predict:
        ypred=model(X)

        #loss:
        loss=criterion(ypred, Y)

        #back propagation:
        loss.backward()

        #optimizer step:
        optimizer.step()

        if i%10==0:
            print("epoch {} : loss {} ".format(i,loss))

        #remettre grad à 0
        optimizer.zero_grad()

    #------------------------------------------------------predire:------------------------------------
    predicted=model(model_viz_tensor).detach().numpy()

    points=torch.tensor([[0,1],[0.5,1.5]])
    predicted_points=model(points).detach().numpy()
    #------------------------------------------------------plot:---------------------------------------


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

    plt.savefig("../../images/MLR.png")

    plt.show()

    #-------------------------------------------------------save----------------------------------------
    #save model:
    traced_script_module = torch.jit.trace(model,X)
    torch.jit.save(traced_script_module,"../models/MLR_model.pt")
    print('model saved *******************')

    #-------------------------------------------------------load model----------------------------------------
    """#load model:
    model_load = torch.load("../models/MLR_model.pt")
    
    y=model_load(X).detach().numpy()
    print("model loaded ", len(y))
    print("******************** fin *********************")
    """