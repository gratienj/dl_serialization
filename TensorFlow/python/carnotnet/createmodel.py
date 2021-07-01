'''
Create Carnot Net Model and save model
'''
import tensorflow as tf

model = tf.keras.applications.ResNet50(weights='imagenet')
#model = tf.keras.applications.ResNet152V2(weights='imagenet')

# Export the model to a SavedModel
model.save('model', save_format='tf')
