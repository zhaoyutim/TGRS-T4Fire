import argparse
import platform

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
from wandb.integration.keras import WandbCallback

from model.lstm.lstm_model import LSTMModel
from model.gru.gru_model import GRUModel
from model.tcn.tcn import compiled_tcn
from model.validation_metrics import ValidationAccuracy
from model.vit_keras import vit

if platform.system() == 'Darwin':
    root_path = '/Users/zhaoyu/PycharmProjects/T4Fire/data'

else:
    root_path = '/geoinfo_vol1/zhao2'
def get_dateset(window_size, batch_size):
    x_dataset = np.load(os.path.join(root_path, 'proj3_train_v2_w' + str(window_size) + '.npy'))
    # x_dataset = np.load('/geoinfo_vol1/zhao2/proj3_allfire_w' + str(window_size) + '.npy')
    # x_dataset = x_dataset[:,::-1,:]
    y_dataset = np.zeros((x_dataset.shape[0],x_dataset.shape[1],2))
    y_dataset[: ,:, 0] = x_dataset[:, :, pow(window_size,2)*5] == 0
    y_dataset[:, :, 1] = x_dataset[:, :, pow(window_size,2)*5] > 0
    
    x_dataset_val1 = np.load(os.path.join(root_path, 'proj3_walker_fire_w'+str(window_size)+'.npy'))
    x_dataset_val2 = np.load(os.path.join(root_path, 'proj3_hanceville_fire_w'+str(window_size)+'.npy'))

    x_dataset_val = np.concatenate((x_dataset_val1, x_dataset_val2), axis=0)
    y_dataset_val = np.zeros((x_dataset_val.shape[0],x_dataset_val.shape[1],2))
    y_dataset_val[: ,:, 0] = x_dataset_val[:, :, pow(window_size,2)*5] == 0
    y_dataset_val[:, :, 1] = x_dataset_val[:, :, pow(window_size,2)*5] > 0

    # x_train, x_val, y_train, y_val = train_test_split(x_dataset[:,:,:pow(window_size,2)*5+1], y_dataset, test_size=0.2, random_state=0)
    x_train, x_val, y_train, y_val = x_dataset[:,:,:pow(window_size,2)*5], x_dataset_val[:,:,:pow(window_size,2)*5], y_dataset, y_dataset_val
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
    val_dataset = val_dataset.repeat(MAX_EPOCHS).batch(224*224)
    steps_per_epoch = x_train.shape[0]//batch_size
    validation_steps = x_val.shape[0]//(224*224)
    return train_dataset, val_dataset, steps_per_epoch, validation_steps
def wandb_config(window_size, model_name, run, num_heads, num_layers, mlp_dim, hidden_size):
    wandb.login()
    # wandb.init(project="tokenized_window_size" + str(window_size) + str(model_name) + 'run' + str(run), entity="zhaoyutim")
    wandb.init(project="proj3_"+model_name+"_grid_search", entity="zhaoyutim")
    wandb.run.name = 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)
    wandb.config = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": MAX_EPOCHS,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "mlp_dim": mlp_dim,
        "embed_dim": hidden_size
    }

if __name__=='__main__':
    import os
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-w', type=int, help='Window size')
    parser.add_argument('-p', type=str, help='Load trained weights')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-r', type=int, help='run')
    parser.add_argument('-lr', type=float, help='learning rate')

    parser.add_argument('-nh', type=int, help='number-of-head')
    parser.add_argument('-md', type=int, help='mlp-dimension')
    parser.add_argument('-ed', type=int, help='embedding dimension')
    parser.add_argument('-nl', type=int, help='num_layers')

    args = parser.parse_args()
    model_name = args.m
    load_weights = args.p
    window_size = args.w
    batch_size=args.b
    num_heads=args.nh
    mlp_dim=args.md
    num_layers=args.nl
    hidden_size=args.ed

    run = args.r
    lr = args.lr
    MAX_EPOCHS = 50
    learning_rate = lr
    weight_decay = lr / 10
    num_classes=2

    input_shape=(10,pow(window_size,2)*5)
    train_dataset, val_dataset, steps_per_epoch, validation_steps = get_dateset(window_size, batch_size)

    wandb_config(window_size, model_name, run, num_heads, mlp_dim, num_layers, hidden_size)

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
        elif model_name=='vit_tiny_custom':
            model = vit.vit_tiny_custom(
                input_shape=input_shape,
                classes=num_classes,
                activation='sigmoid',
                pretrained=True,
                include_top=True,
                pretrained_top=True,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                hidden_size=hidden_size
            )
        elif model_name == 'gru_custom':
            gru = GRUModel(input_shape, num_classes)
            model = gru.get_model_custom(input_shape, num_classes, num_layers, hidden_size)
        elif model_name == 'gru3_bi':
            gru = GRUModel(input_shape, num_classes)
            model = gru.get_model_bi(input_shape, num_classes)
        elif model_name == 'lstm_custom':
            lstm = LSTMModel(input_shape, num_classes)
            model = lstm.get_model_custom(input_shape, num_classes, num_layers, hidden_size)
        elif model_name == 'lstm3_bi':
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
        elif model_name=='tcn':
            model = compiled_tcn(return_sequences=True,
                                 num_feat=input_shape[-1],
                                 num_classes=num_classes,
                                 nb_filters=mlp_dim,
                                 kernel_size=hidden_size,
                                 dilations=[2 ** i for i in range(9)],
                                 nb_stacks=num_layers,
                                 max_len=input_shape[0],
                                 use_weight_norm=True,
                                 use_skip_connections=True)
        else:
            raise('no suport model')

        model.summary()
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        if model_name == 'vit_tiny_custom':
            checkpoint = ModelCheckpoint(os.path.join(root_path, 'proj3_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_'+str(num_heads)+'_'+str(mlp_dim)+'_'+str(hidden_size)+'_'+str(num_layers)+'_'+str(batch_size)), monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        elif model_name == 'gru_custom' or model_name == 'lstm_custom':
            checkpoint = ModelCheckpoint(os.path.join(root_path, 'proj3_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_'+str(hidden_size)+'_'+str(num_layers)), monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        else:
            checkpoint = ModelCheckpoint(os.path.join(root_path, 'proj3_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)), monitor="val_loss", mode="min", save_best_only=True, verbose=1)

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
    metrics = ValidationAccuracy(val_dataset, validation_steps)
    if load_weights== 'yes':
        model.load_weights(os.path.join(root_path, 'proj3_' + model_name + 'w' + str(window_size) + '_nopretrained'+'_run'+str(run)))
    else:
        print('training in progress')
        history = model.fit(
            x=train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            epochs=MAX_EPOCHS,
            callbacks=[WandbCallback()],
        )
        if model_name == 'vit_tiny_custom':
            model.save(os.path.join(root_path, 'proj3_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_'+str(num_heads)+'_'+str(mlp_dim)+'_'+str(hidden_size)+'_'+str(num_layers)+'_'+str(batch_size)))
        elif model_name == 'gru_custom' or model_name == 'lstm_custom':
            model.save(os.path.join(root_path, 'proj3_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_'+str(hidden_size)+'_'+str(num_layers)))
        else:
            model.save(os.path.join(root_path, 'proj3_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)))

