import argparse
import platform
import os
# os.environ['CUDA_VISIBLE_DEVICES']='9'
os.environ['TF_GPU_THREAD_MODE']='gpu_private'
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import wandb
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint
from wandb.integration.keras import WandbCallback

from data_processor.tokenize_processor import TokenizeProcessor
from model.lstm.lstm_model import LSTMModel
from model.gru.gru_model import GRUModel
from model.tcn.tcn import compiled_tcn
from model.validation_metrics import ValidationAccuracy
from model.vit_keras import vit

if platform.system() == 'Darwin':
    root_path = '/Users/zhaoyu/PycharmProjects/T4Fire/data'
else:
    root_path = '/geoinfo_vol1/zhao2/proj5_dataset'
def get_dateset(window_size, batch_size, mode, ts_length):
    if mode == 'sw':
        print('Loading Dataset')
        img_dataset = np.load(os.path.join(root_path,'proj5_train_img_seqtoone_l'+str(ts_length)+'.npy')).transpose((0, 3, 4, 1, 2))
        label_dataset = np.load(os.path.join(root_path, 'proj5_train_label_seqtoone_l' + str(ts_length) + '.npy')).transpose((0, 2, 3, 1))
        img_val = np.load(os.path.join(root_path,'proj5_val_img_seqtoone_l'+str(ts_length)+'.npy')).transpose((0, 3, 4, 1, 2))
        label_val = np.load(os.path.join(root_path,'proj5_val_label_seqtoone_l'+str(ts_length)+'.npy')).transpose((0, 2, 3, 1))
        img_dataset, label_dataset, img_val, label_val = img_dataset.reshape(-1, img_dataset.shape[-2], img_dataset.shape[-1]),\
                                                         label_dataset.reshape(-1, label_dataset.shape[-1]),\
                                                         img_val.reshape(-1, img_val.shape[-2], img_val.shape[-1]),\
                                                         label_val.reshape(-1, label_val.shape[-1])
        print('Dataset Loaded')
    else:
        print('Loading Dataset')
        img_dataset = np.load(os.path.join(root_path,'proj5_train_img_seqtoseq_l'+str(ts_length)+'.npy')).transpose((0, 3, 4, 1, 2))
        label_dataset = np.load(os.path.join(root_path, 'proj5_train_label_seqtoseq_l' + str(ts_length) + '.npy')).transpose((0, 3, 4, 1, 2))
        img_val = np.load(os.path.join(root_path,'proj5_val_img_seqtoseq_l'+str(ts_length)+'.npy')).transpose((0, 3, 4, 1, 2))
        label_val = np.load(os.path.join(root_path,'proj5_val_label_seqtoseq_l'+str(ts_length)+'.npy')).transpose((0, 3, 4, 1, 2))
        img_dataset, label_dataset, img_val, label_val = img_dataset.reshape(-1, img_dataset.shape[-2], img_dataset.shape[-1]),\
                                                         label_dataset.reshape(-1, label_dataset.shape[-2], label_dataset.shape[-1]),\
                                                         img_val.reshape(-1, img_val.shape[-2], img_val.shape[-1]),\
                                                         label_val.reshape(-1, label_val.shape[-2], label_val.shape[-1])
        print('Dataset Loaded')
    print(img_dataset.shape, label_dataset.shape, img_val.shape, label_val.shape)
    if mode != 'sw':
        y_dataset = np.zeros((label_dataset.shape[0], label_dataset.shape[1], 2))
        y_dataset[: ,:, 0] = label_dataset[:, :, 0] == 0
        y_dataset[:, :, 1] = label_dataset[:, :, 0] > 0

        y_val = np.zeros((label_val.shape[0], label_val.shape[1], 2))
        y_val[: ,:, 0] = label_val[:, :, 0] == 0
        y_val[:, :, 1] = label_val[:, :, 0] > 0
    else:
        y_dataset = np.zeros((label_dataset.shape[0], 2))
        y_dataset[:, 0] = label_dataset[:, 0] == 0
        y_dataset[:, 1] = label_dataset[:, 0] > 0
        y_val = np.zeros((label_val.shape[0], 2))
        y_val[:, 0] = label_val[:, 0] == 0
        y_val[:, 1] = label_val[:, 0] > 0

    x_train, x_val, y_train, y_val = img_dataset, img_val, y_dataset, y_val

    class_0_indices = np.where(y_train[..., 0] == 1)
    class_1_indices = np.where(y_train[..., 1] == 1)

    num_class_1_samples = len(class_1_indices[0])

    # Randomly sample class 0 indices to get the same number of samples as class 1
    sampled_class_0_indices = np.random.choice(len(class_0_indices[0]), num_class_1_samples, replace=False)

    # Get the sampled class 0 indices
    sampled_class_0_indices = (class_0_indices[0][sampled_class_0_indices])

    # Concatenate class 1 and sampled class 0 indices
    balanced_indices = (np.concatenate((class_1_indices[0], sampled_class_0_indices)))

    # Create the balanced dataset
    x_train = x_train[balanced_indices, :, :]
    if mode != 'sw':
        y_train = y_train[balanced_indices, :, :]
    else:
        y_train = y_train[balanced_indices, :]

    # def make_generator(inputs, labels):
    #     def _generator():
    #         for input, label in zip(inputs, labels):
    #             yield input, label
    #
    #     return _generator
    # train_dataset = tf.data.Dataset.from_generator(make_generator(x_train, y_train),
    #                                                (tf.float32, tf.int16))
    # val_dataset = tf.data.Dataset.from_generator(make_generator(x_val, y_val),
    #                                              (tf.float32, tf.int16))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    train_dataset = train_dataset.shuffle(batch_size).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    steps_per_epoch = x_train.shape[0]//batch_size
    validation_steps = x_val.shape[0]//(512*512)
    return train_dataset, val_dataset, steps_per_epoch, validation_steps

def wandb_config(model_name, run, num_heads, num_layers, mlp_dim, hidden_size, mode):
    wandb.login()
    # wandb.init(project="tokenized_window_size" + str(window_size) + str(model_name) + 'run' + str(run), entity="zhaoyutim")
    wandb.init(project="proj5_"+model_name+"_grid_search", entity="zhaoyutim", sync_tensorboard=True)
    wandb.run.name = 'num_heads_' + str(num_heads) + 'num_layers_'+ str(num_layers)+ 'mlp_dim_'+str(mlp_dim)+'hidden_size_'+str(hidden_size)+'batchsize_'+str(batch_size)+'run_' + str(run) + 'mode_' + mode
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

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-m', type=str, help='Model to be executed')
    parser.add_argument('-b', type=int, help='batch size')
    parser.add_argument('-r', type=int, help='run')
    parser.add_argument('-lr', type=float, help='learning rate')

    parser.add_argument('-nh', type=int, help='number-of-head')
    parser.add_argument('-md', type=int, help='mlp-dimension')
    parser.add_argument('-ed', type=int, help='embedding dimension')
    parser.add_argument('-nl', type=int, help='num_layers')

    parser.add_argument('-mode', type=str, help='sliding window mode or seq tp seq mode')

    args = parser.parse_args()
    model_name = args.m
    batch_size=args.b
    num_heads=args.nh
    mlp_dim=args.md
    num_layers=args.nl
    hidden_size=args.ed
    mode = args.mode

    if mode == 'sw':
        # Sequence to One
        return_sequence = False
    else:
        # Sequence to Sequnce
        return_sequence = True

    run = args.r
    lr = args.lr
    MAX_EPOCHS = 50
    learning_rate = lr
    weight_decay = lr / 10
    num_classes=2
    window_size=1

    input_shape=(10,pow(window_size,2)*6)
    train_dataset, val_dataset, steps_per_epoch, validation_steps = get_dateset(window_size=window_size, batch_size=batch_size, mode=mode, ts_length=10)

    wandb_config(model_name, run, num_heads, mlp_dim, num_layers, hidden_size, mode)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if model_name=='vit_tiny_custom':
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
                hidden_size=hidden_size,
                return_sequence=return_sequence
            )
        elif model_name == 'gru_custom':
            gru = GRUModel(input_shape, num_classes)
            model = gru.get_model_custom(input_shape, num_classes, num_layers, hidden_size, return_sequence)
        elif model_name == 'lstm_custom':
            lstm = LSTMModel(input_shape, num_classes)
            model = lstm.get_model_custom(input_shape, num_classes, num_layers, hidden_size, return_sequence)
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
            checkpoint = ModelCheckpoint('/geoinfo_vol1/zhao2/proj5_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_'+str(num_heads)+'_'+str(mlp_dim)+'_'+str(hidden_size)+'_'+str(num_layers)+'_'+str(batch_size)+'_m'+mode, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        elif model_name == 'gru_custom' or model_name == 'lstm_custom':
            checkpoint = ModelCheckpoint('/geoinfo_vol1/zhao2/proj5_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_'+str(hidden_size)+'_'+str(num_layers)+'_m'+mode, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
        else:
            checkpoint = ModelCheckpoint('/geoinfo_vol1/zhao2/proj5_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_m'+mode, monitor="val_loss", mode="min", save_best_only=True, verbose=1)

        model.compile(
            optimizer=optimizer,
            loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="accuracy")
            ],
        )

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir='logs',
                                                 histogram_freq=1,
                                                 profile_batch='5,10')
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)
    metrics = ValidationAccuracy(val_dataset, validation_steps)
    print('training in progress')
    history = model.fit(
        x=train_dataset,
        batch_size=batch_size,
        # steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        # validation_steps=validation_steps,
        epochs=MAX_EPOCHS,
        callbacks=[WandbCallback()],
        use_multiprocessing = True,
        workers=16
    )
    if model_name == 'vit_tiny_custom' or model_name == 'tcn':
        model.save('/geoinfo_vol1/zhao2/proj5_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_'+str(num_heads)+'_'+str(mlp_dim)+'_'+str(hidden_size)+'_'+str(num_layers)+'_'+str(batch_size)+'_m'+mode)
    elif model_name == 'gru_custom' or model_name == 'lstm_custom':
        model.save('/geoinfo_vol1/zhao2/proj5_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_'+str(hidden_size)+'_'+str(num_layers)+'_m'+mode)
    else:
        model.save('/geoinfo_vol1/zhao2/proj5_'+model_name+'w' + str(window_size) + '_nopretrained'+'_run'+str(run)+'_m'+mode)

