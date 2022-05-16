import argparse

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from sklearn.model_selection import train_test_split
from wandb.integration.keras import WandbCallback

from model.lstm.lstm_model import LSTMModel
from model.gru.gru_model import GRUModel
from model.vit_keras import vit
def get_dateset(window_size, batch_size):
    x_dataset = np.load('/geoinfo_vol1/zhao2/proj3_train_v2_w'+str(window_size)+'.npy')
    # x_dataset = x_dataset[:,::-1,:]
    y_dataset = np.zeros((x_dataset.shape[0],x_dataset.shape[1],2))
    y_dataset[: ,:, 0] = x_dataset[:, :, pow(window_size,2)*5] == 0
    y_dataset[:, :, 1] = x_dataset[:, :, pow(window_size,2)*5] > 0

    x_train, x_val, y_train, y_val = train_test_split(x_dataset[:,:,:pow(window_size,2)*5], y_dataset, test_size=0.2, random_state=0)

    def make_generator(inputs, labels):
        def _generator():
            for input, label in zip(inputs, labels):
                yield input, label

        return _generator


    train_dataset = tf.data.Dataset.from_generator(make_generator(x_train, y_train),
                                                   (tf.float32, tf.int16))
    val_dataset = tf.data.Dataset.from_generator(make_generator(x_val, y_val),
                                                 (tf.float32, tf.int16))

    train_dataset = train_dataset.shuffle(batch_size).repeat(MAX_EPOCHS).batch(batch_size)
    val_dataset = val_dataset.shuffle(batch_size).repeat(MAX_EPOCHS).batch(batch_size)

    steps_per_epoch = x_train.shape[0]//batch_size
    validation_steps = x_val.shape[0]//batch_size

    return train_dataset, val_dataset, steps_per_epoch, validation_steps


def wandb_config(window_size, model_name, run):
    wandb.login()
    wandb.init(project="tokenized_window_size" + str(window_size) + str(model_name) + 'run' + str(run), entity="zhaoyutim")
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
    parser.add_argument('-w', type=int, help='Window size')
    parser.add_argument('-p', type=str, help='Load trained weights')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-r', type=int, help='run')

    args = parser.parse_args()
    model_name = args.m
    load_weights = args.p
    window_size = args.w
    batch_size=args.b
    run = args.r
    print(run)

    MAX_EPOCHS = 50
    learning_rate = 0.00001
    weight_decay = 0.000001

    num_classes=2
    input_shape=(10,pow(window_size,2)*5)

    train_dataset, val_dataset, steps_per_epoch, validation_steps = get_dateset(window_size, batch_size)

    # wandb_config(window_size, model_name, run)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if model_name == 'vit_small':
            model = vit.vit_small(
                input_shape=input_shape,
                classes=num_classes,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=True
            )
        elif model_name=='vit_tiny':
            model = vit.vit_tiny(
                input_shape=input_shape,
                classes=num_classes,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=True
            )
        elif model_name=='vit_tiny_3':
            model = vit.vit_tiny_3(
                input_shape=input_shape,
                classes=num_classes,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=True
            )
        elif model_name=='vit_tiny_4':
            model = vit.vit_tiny_4(
                input_shape=input_shape,
                classes=num_classes,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=True
            )
        elif model_name=='vit_tiny_6':
            model = vit.vit_tiny_6(
                input_shape=input_shape,
                classes=num_classes,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=True
            )
        elif model_name=='vit_tiny_6_2':
            model = vit.vit_tiny_6_2(
                input_shape=input_shape,
                classes=num_classes,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=True
            )
        elif model_name=='vit_tiny_6_3':
            model = vit.vit_tiny_6_3(
                input_shape=input_shape,
                classes=num_classes,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=True
            )
        elif model_name == 'gru5':
            gru = GRUModel(input_shape, num_classes)
            model = gru.get_model_5_layers(input_shape, num_classes)

        elif model_name == 'gru2':
            gru = GRUModel(input_shape, num_classes)
            model = gru.get_model_2_layers(input_shape, num_classes)
        elif model_name == 'gru3':
            gru = GRUModel(input_shape, num_classes)
            model = gru.get_model_3_layers(input_shape, num_classes)
        elif model_name == 'gru4':
            gru = GRUModel(input_shape, num_classes)
            model = gru.get_model_4_layers(input_shape, num_classes)

        elif model_name == 'gru5_bi':
            gru = GRUModel(input_shape, num_classes)
            model = gru.get_model_bi(input_shape, num_classes)
        elif model_name == 'lstm5':
            lstm = LSTMModel(input_shape, num_classes)
            model = lstm.get_model_5_layers(input_shape, num_classes)
        elif model_name == 'lstm2':
            lstm = LSTMModel(input_shape, num_classes)
            model = lstm.get_model_2_layers(input_shape, num_classes)
        elif model_name == 'lstm3':
            lstm = LSTMModel(input_shape, num_classes)
            model = lstm.get_model_3_layers(input_shape, num_classes)
        elif model_name == 'lstm4':
            lstm = LSTMModel(input_shape, num_classes)
            model = lstm.get_model_4_layers(input_shape, num_classes)

        elif model_name == 'lstm5_bi':
            lstm = LSTMModel(input_shape, num_classes)
            model = lstm.get_model_bi(input_shape, num_classes)
        elif model_name=='vit_base':
            model = vit.vit_base(
                input_shape=input_shape,
                classes=num_classes,
                activation='sigmoid',
                pretrained=False,
                include_top=True,
                pretrained_top=True
            )
        else:
            raise('no suport model')

        model.summary()

        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy")
            ],
        )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)

    if load_weights== 'yes':
        model.load_weights('/geoinfo_vol1/zhao2/proj3_' + model_name + 'w' + str(window_size) + '_nopretrained'+'_run'+str(run))
    else:
        print('training in progress')
        history = model.fit(
            x=train_dataset,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            epochs=MAX_EPOCHS,
            # callbacks=[WandbCallback()],
        )
        model.save('/geoinfo_vol1/zhao2/proj3_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run))
