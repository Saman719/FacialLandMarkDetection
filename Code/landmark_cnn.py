import time
from importlib.resources import path
from operator import mod
from struct import pack
from unicodedata import name
import scipy.io
import os
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras import regularizers
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import data_handler as my_data_handler

train_data, train_lbl, test_data, test_lbl = my_data_handler.train_test()
# print(train_data)
for data in train_data:
    if data.ndim != 3:
        print(data)
train_data = np.array(train_data, dtype=np.float64) / 255
train_lbl = np.array(train_lbl, dtype=np.float64)
test_data = np.array(test_data, dtype=np.float64) / 255
test_lbl = np.array(test_lbl, dtype=np.float64)
print('train_data.shape: ', train_data.shape)
print('train_lbl.shape: ', train_lbl.shape)

vgg = VGG19(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(my_data_handler.WIDTH, my_data_handler.HEIGHT, 3)))
# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False
# flatten the max-pooling output of VGG
vgg_output = vgg.output
# averagePooling = GlobalAveragePooling2D()(vgg_output)
flatten = Flatten()(vgg_output)

bboxHead = Dense(1024, activation="relu")(flatten)
# bboxHead = Dense(512, activation="relu")(bboxHead)
# bboxHead = Dense(32, activation="relu")(bboxHead)
# bboxHead = Dense(16, activation="relu")(bboxHead)
bboxHead = Dense(136, name="bounding_box")(bboxHead)


model = Model(
    inputs=vgg.input,
    outputs=(bboxHead))
'''
MY FITING
'''
# optimizer = keras.optimizers.RMSprop()

# loss_fn = keras.losses.MeanSquaredError()
# train_acc_metric = tf.keras.metrics.MeanSquaredError()
# val_acc_metric = tf.keras.metrics.MeanSquaredError()


# @tf.function
# def train_step(x, y):
#     with tf.GradientTape() as tape:
#         logits = model(x, training=True)
#         loss_value = loss_fn(y, logits)
#     grads = tape.gradient(loss_value, model.trainable_weights)
#     optimizer.apply_gradients(zip(grads, model.trainable_weights))
#     train_acc_metric.update_state(y, logits)
#     return loss_value


# @tf.function
# def test_step(x, y):
#     val_logits = model(x, training=False)
#     val_acc_metric.update_state(y, val_logits)


# batch_size = 50
# epochs = 3

# train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_lbl))
# train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# val_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_lbl))
# val_dataset = val_dataset.batch(batch_size)


# for epoch in range(epochs):
#     print("\nStart of epoch %d" % (epoch,))
#     start_time = time.time()

#     # Iterate over the batches of the dataset.
#     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#         loss_value = train_step(x_batch_train, y_batch_train)

#         # Log every 200 batches.
#         if step % 200 == 0:
#             print(
#                 "Training loss (for one batch) at step %d: %.4f"
#                 % (step, float(loss_value))
#             )
#             print("Seen so far: %d samples" % ((step + 1) * batch_size))

#     # Display metrics at the end of each epoch.
#     train_acc = train_acc_metric.result()
#     print("Training acc over epoch: %.4f" % (float(train_acc),))

#     # Reset training metrics at the end of each epoch
#     train_acc_metric.reset_states()

#     # Run a validation loop at the end of each epoch.
#     for x_batch_val, y_batch_val in val_dataset:
#         test_step(x_batch_val, y_batch_val)

#     val_acc = val_acc_metric.result()
#     val_acc_metric.reset_states()
#     print("Validation acc: %.4f" % (float(val_acc),))
#     print("Time taken: %.2fs" % (time.time() - start_time))
'''
MY FITING
'''
losses = {
    "bounding_box": "mean_squared_error",
    # "class_label": "sparse_categorical_crossentropy",
}
lossWeights = {
    "bounding_box": 1.0,
    # "class_label": 1.0,
}
model.compile(loss=losses, optimizer='rmsprop', metrics=[
              "accuracy"], loss_weights=lossWeights)
print(model.summary())
trainTargets = {
    "bounding_box": train_lbl,
    # "class_label": train_lbl[:, 4],
}
testTargets = {
    "bounding_box": test_lbl,
    # "class_label": train_lbl[:, 4],
}
print("[INFO] training model...")
H = model.fit(
    train_data, trainTargets,
    validation_data=(test_data, testTargets),
    batch_size=50,
    epochs=3,
    verbose=1,
    use_multiprocessing=True
)

model.save('./Code/landmark-only.h5')
