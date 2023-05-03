import torch
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


#----------------------------------------------------model--------------------------------------------

class linearRegression(torch.nn.Module):
    def __init__(self, inputSize=1, outputSize=1):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

if __name__=='__main__':
    #---------------------------------------------------data -------------------------------------------
    X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

    print("shape numpy: X: {} , Y: {}".format(X_numpy.shape, y_numpy.shape))
    # cast to float Tensor
    X = torch.from_numpy(X_numpy.astype(np.float32))
    y = torch.from_numpy(y_numpy.astype(np.float32))
    Y = y.view(y.shape[0], 1)

    print("shape: X: {} , Y: {}".format(X.shape, Y.shape))

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
    predicted=model(X).detach().numpy()


    #------------------------------------------------------plot:---------------------------------------
    plt.style.use('default')

    fig = plt.figure(figsize=(12, 4))

    plt.plot(X_numpy, y_numpy, 'ro',label="dataset")
    plt.plot(X_numpy, predicted, 'b',label="predected linear function")
    plt.title("linear regression")
    plt.legend()
    plt.grid()
    plt.savefig("../../../results/images/LR.png")
    plt.show()


    #-------------------------------------------------------save----------------------------------------
    #save model:
    # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
    traced_script_module = torch.jit.trace(model,X)
    torch.jit.save(traced_script_module,"../../../results/models/LR_model.pt")
    print('model saved *******************')

    #-------------------------------------------------------load model----------------------------------------
    #load model:
    """model_load = torch.load("../../../results/models/LR_model.pt")
    
    x=torch.tensor([10.])
    
    y=model_load(x).detach().numpy()
    
    print("module loading ... \n",x,y)"""
