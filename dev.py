import tensorflow_addons as tfa
import tensorflow as tf
import keras.backend as K
import wandb
from wandb.keras import WandbCallback
import numpy as np
import os
import numpy as np
from sklearn.model_selection import train_test_split
from vit_keras import vit, utils

x_dataset = np.load('/NOBACKUP/zhao2/proj3_train_5_channel.npy').transpose((1,0,2))
y_dataset = np.zeros((x_dataset.shape[0],x_dataset.shape[1],2))
y_dataset[: ,:, 0] = x_dataset[:, :, 45] == 0
y_dataset[:, :, 1] = x_dataset[:, :, 45] > 0

x_train, x_test, y_train, y_test = train_test_split(x_dataset[:,:,:45], y_dataset, test_size=0.2)

print(x_train.shape)
print(y_train.shape)
batch_size=256
MAX_EPOCHS = 20
learning_rate = 0.001
weight_decay = 0.0001

num_classes=3
input_shape=(10,45)
projection_dim = 128
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim,]
transformer_layers = 8
mlp_head_units = [2048, 1024]

wandb.login()
wandb.init(project="vit_model", entity="zhaoyutim")
wandb.config = {
  "learning_rate": learning_rate,
  "weight_decay": weight_decay,
  "epochs": MAX_EPOCHS,
  "batch_size": batch_size,
  "num_heads":num_heads,
  "transformer_layers": transformer_layers
}
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

model = vit.vit_b16(
    input_shape=input_shape,
    classes=2,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=True
)

optimizer = tfa.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)

model.compile(
    optimizer=optimizer,
    loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, gamma=0.01),
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy")
    ],
)
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=MAX_EPOCHS,
    validation_split=0.1,
    callbacks=[WandbCallback()],
)
model.save('/NOBACKUP/zhao2/proj3_transformer_b16_pretrained')