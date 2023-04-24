import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class non_LR(torch.nn.Module):
    def __init__(self):
        super(non_LR, self).__init__()
        self.Dense1 = torch.nn.Linear(1, 32)
        self.Dense2 = torch.nn.Linear(32, 32)
        self.Dense3 = torch.nn.Linear(32, 16)
        self.Dense4 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x= self.Dense1(x)
        x= F.relu(self.Dense2(x))
        x= F.relu(self.Dense3(x))
        out= self.Dense4(x)
        return out


if __name__=="__main__":
    # Create noisy data:
    x_data = np.linspace(-10, 10, num=1000)
    y_data = 0.1*x_data*np.cos(x_data) + 0.1*np.random.normal(size=1000)
    print('Data created successfully')

    # data to torch tensors:
    X = torch.from_numpy(x_data.astype(np.float32))
    Y = torch.from_numpy(y_data.astype(np.float32))

    X = X.view(X.shape[0], 1)
    Y = Y.view(Y.shape[0], 1)

    print(X.shape, Y.shape)


    #---------------------------------------------------train---------------------------------------------
    #instance du model:
    model=non_LR()

    #loss function:
    criterion=torch.nn.MSELoss()

    #optimizer:
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

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


    # Compute the output
    y_predicted = model(X).detach().numpy()

    print(y_predicted.shape)


    """
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

    #plt.savefig("../../images/non_LR.png")

    plt.show()
    plt.clf()"""




    #-------------------------------------------------------save----------------------------------------
    #save model:
    #traced_script_module = torch.jit.trace(model,X)
    #torch.jit.save(traced_script_module,"../models/non_LR_model.pt")

    # Set the model to inference mode.
    model.eval()

    # Use the exporter from torch to convert to onnx
    # model (that has the weights and net arch)
    # Create some sample input in the shape this model expects
    dummy_input = torch.randn(10, 1, 1)

    # It's optional to label the input and output layers
    input_names = [ "input" ]
    output_names = [ "output" ]

    dynamic_axes=({'input' : {0 : 'batch_size'},    # variable length axes
                  'output' : {0 : 'batch_size'}})

    torch.onnx.export(
        model,
        dummy_input,
        "non_LR_model.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    import onnxruntime
    import torch
    import torchvision.models as models
    import numpy as np

    ort_session = onnxruntime.InferenceSession("non_LR_model.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    x = torch.randn(40, 1, 1)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)
    print("predicted output: ",ort_outs )


