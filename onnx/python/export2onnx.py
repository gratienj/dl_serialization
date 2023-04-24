import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.onnx
import tensorflow as tf
import tf2onnx
import onnx
import onnxruntime as ort

from sklearn import datasets
import numpy as np



#TODO
PT_MODELS_DIR="../pytorch/models"
TF_MODELS_DIR="../tensorflow/CPU/models"

pt_models=os.listdir(PT_MODELS_DIR)
tf_models=os.listdir(TF_MODELS_DIR)

# convertir les models Tensorflow et pytorch to onnx :
model_rank=2
model_pt_name=PT_MODELS_DIR+"/"+"C8DNN1.pt"
#model_tf_name=TF_MODELS_DIR+"/"+tf_models[model_rank]
print(pt_models)
print(model_pt_name)
# Model class must be defined somewhere
model_pt = torch.jit.load(model_pt_name)
model_pt.eval()

# Data
features=1
samples=1

X_numpy, _ = datasets.make_regression(n_samples=samples, n_features=features, noise=20, random_state=4)
input_tensor = torch.from_numpy(X_numpy.astype(np.float32))
target= model_pt.forward(input_tensor)
print(X_numpy, target)

# convertir le model en onnx :
torch.onnx.export(model_pt,
                  torch.randn(1, samples, features),
                  "models/C8DNN1.onnx",
                  verbose=True,
                  input_names = ['input'],              # the model's input names
                  output_names = ['output'],            # the model's output names
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
                  example_outputs=target)



sess = ort.InferenceSession("models/C8DNN1.onnx")
input_name = sess.get_inputs()[0].name
pred_onx = sess.run(None, {input_name: X_numpy.astype(np.float32)})[0]
print(pred_onx)


"""model_tf=tf.keras.models.load_model(model_tf_name)
input_signature = [tf.TensorSpec([1, 4], tf.float32, name='x')]
# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model_tf, input_signature, opset=13)
onnx.save(onnx_model, "models/DNN_tf.onnx")"""