import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('D:\First Aid Detection\ssd_mobilenet_v2_2') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
