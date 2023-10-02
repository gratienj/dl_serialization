import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.onnx
#import tensorflow as tf
#import tf2onnx
import onnx
import onnxruntime as ort

from sklearn import datasets
import numpy as np



#TODO
PT_MODELS_DIR="../../model/dnns/pytorch"
#TF_MODELS_DIR="../tensorflow/CPU/models"

pt_models=os.listdir(PT_MODELS_DIR)
#tf_models=os.listdir(TF_MODELS_DIR)

# convertir les models Tensorflow et pytorch to onnx :
#model_rank=2
device = 'cuda'
model_pt_name=PT_MODELS_DIR+"/"+"C8DNN2.pt"
#model_tf_name=TF_MODELS_DIR+"/"+tf_models[model_rank]
print(pt_models)
print(model_pt_name)
# Model class must be defined somewhere
model_pt = torch.jit.load(model_pt_name)
model_pt.eval()
model_pt.to(device)

# Data
features=2
samples=1

X_numpy, _ = datasets.make_regression(n_samples=samples, n_features=features, noise=20, random_state=4)
input_tensor = torch.from_numpy(X_numpy.astype(np.float32)).to(device)
target= model_pt.forward(input_tensor)
print(X_numpy, target)

# convertir le model en onnx :
torch.onnx.export(model_pt,
                  torch.randn(1, samples, features).to(device),
                  "../models/C8DNN2-cuda.onnx",
                  export_params=True,  # store the trained parameter weights inside the model file
                  verbose=True,
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'],            # the model's output names
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})
                  #example_outputs=target)

'''
 torch.onnx.export(net,  # model being run
                              X,  # model input (or a tuple for multiple inputs)
                              "initializer_x32.onnx",  # where to save the model (can be a file or file-like object)
                              export_params=True,  # store the trained parameter weights inside the model file
                              opset_version=15,  # the ONNX version to export the model to
                              do_constant_folding=True,  # whether to execute constant folding for optimization
                              input_names=['input'],  # the model's input names
                              output_names=['output'],  # the model's output names
                              dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                            'output': {0: 'batch_size'}})

'''

sess = ort.InferenceSession("../models/C8DNN1.onnx")
input_name = sess.get_inputs()[0].name
#pred_onx = sess.run(None, {input_name: X_numpy.astype(np.float32)})[0]
#print(pred_onx)


"""model_tf=tf.keras.models.load_model(model_tf_name)
input_signature = [tf.TensorSpec([1, 4], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model_tf, input_signature, opset=13)
onnx.save(onnx_model, "models/DNN_tf.onnx")"""
