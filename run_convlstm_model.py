import argparse

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from wandb.integration.keras import WandbCallback
from segmentation_models.metrics import iou_score, f1_score
from segmentation_models import Unet, Linknet, PSPNet, FPN
from keras_unet_collection import models
from model.convlstm_models.convlstm_models import get_convlstm_unet2, unet


def get_dateset(batch_size):
    train_dataset = np.load('/geoinfo_vol1/zhao2/proj3_train_img_v2.npy').transpose((0,1,3,4,2))
    # train_dataset = np.load('/Users/zhaoyu/PycharmProjects/T4Fire/data/proj3_train_img_v2.npy').transpose((0,1,3,4,2))
    print(train_dataset.shape)
    y_dataset = train_dataset[:,:,:,:,5]>0
    x_train, x_val, y_train, y_val = train_test_split(train_dataset[:,:,:,:,:5].astype(float), y_dataset.astype(float), test_size=0.2)
    def make_generator(inputs, labels):
        def _generator():
            for input, label in zip(inputs, labels):
                yield input, label

        return _generator


    train_dataset = tf.data.Dataset.from_generator(make_generator(x_train, y_train),
                                                   (tf.float32, tf.float32))
    val_dataset = tf.data.Dataset.from_generator(make_generator(x_val, y_val),
                                                 (tf.float32, tf.float32))

    train_dataset = train_dataset.shuffle(batch_size).repeat(MAX_EPOCHS).batch(batch_size)
    val_dataset = val_dataset.shuffle(batch_size).repeat(MAX_EPOCHS).batch(batch_size)

    steps_per_epoch = x_train.shape[0]//batch_size
    validation_steps = x_val.shape[0]//batch_size

    return train_dataset, val_dataset, steps_per_epoch, validation_steps

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true,[-1])
    y_pred_f = tf.reshape(y_pred,[-1])
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    smooth = 1.0
    return 1-(2.0*intersection+smooth)/(tf.math.reduce_sum(y_true_f)+tf.math.reduce_sum(y_pred_f)+smooth)

def wandb_config(model_name, backbone):
    wandb.login()
    wandb.init(project=str(model_name)+'_'+str(backbone), entity="zhaoyutim")
    # wandb.config = {
    #   "learning_rate": learning_rate,
    #   "weight_decay": weight_decay,
    #   "epochs": MAX_EPOCHS,
    #   "batch_size": batch_size,
    #   "num_heads":num_heads,
    #   "transformer_layers": transformer_layers
    # }

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-p', type=str, help='Load trained weights')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-bb', type=str, help='backbone')

    args = parser.parse_args()
    model_name = args.m
    load_weights = args.p
    backbone = args.bb
    sm.set_framework('tf.keras')
    batch_size=args.b
    MAX_EPOCHS=100
    learning_rate = 0.0001
    weight_decay = 0.00001

    train_dataset, val_dataset, steps_per_epoch, validation_steps = get_dateset(batch_size)

    wandb_config(model_name, backbone)

    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    if model_name == 'convlstm_unet':
        model = get_convlstm_unet2((10,224,224,5))
    model.summary()

    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(optimizer, loss=dice_coef, metrics=[iou_score, f1_score])

    # options = tf.data.Options()
    # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    # train_dataset = train_dataset.with_options(options)
    # val_dataset = val_dataset.with_options(options)

    if load_weights== 'yes':
        model.load_weights('/geoinfo_vol1/zhao2/proj3_'+model_name+'_pretrained_'+backbone)
    else:
        print('training in progress')
        history = model.fit(
            train_dataset,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            epochs=MAX_EPOCHS,
            callbacks=[WandbCallback()],
        )
        model.save('/geoinfo_vol1/zhao2/proj3_'+model_name+'_pretrained_'+backbone)
