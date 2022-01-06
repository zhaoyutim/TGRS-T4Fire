import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow_addons as tfa
from model.vit.utilities.patches import Patches
from model.vit.utilities.weights_loader import load_weights_numpy
from model.vit.vit_model import VisionTransformerGenerator
import matplotlib.pyplot as plt


# def visualizalize_patches():
#     plt.figure(figsize=(4, 4))
#     image = dataset_train[0][np.random.choice(range(dataset_train[0].shape[0]))]
#     plt.imshow(image.astype("uint8"))
#     plt.axis("off")
#
#     resized_image = tf.image.resize(
#         tf.convert_to_tensor([image]), size=(image_size, image_size)
#     )
#     patches = Patches(patch_size)(resized_image)
#     print(f"Image size: {image_size} X {image_size}")
#     print(f"Patch size: {patch_size} X {patch_size}")
#     print(f"Patches per image: {patches.shape[1]}")
#     print(f"Elements per patch: {patches.shape[-1]}")
#
#     n = int(np.sqrt(patches.shape[1]))
#     plt.figure(figsize=(4, 4))
#     for i, patch in enumerate(patches[0]):
#         ax = plt.subplot(n, n, i + 1)
#         patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
#         plt.imshow(patch_img.numpy().astype("uint8"))
#         plt.axis("off")

if __name__=='__main__':
    # MAX_EPOCHS = 20
    # BATCH_SIZE = 1024
    # dataset = np.load('../data/proj3_train_5_channel_full.npy').transpose((1,0,2))[:,:,:45]
    # y_dataset = np.load('../data/y_dataset.npy')
    # def make_generator(inputs, labels):
    #     def _generator():
    #         for input, label in zip(inputs, labels):
    #             yield input, label
    #     return _generator
    # dataset = tf.data.Dataset.from_generator(make_generator(dataset, y_dataset), (tf.float32, tf.int16))
    # shuffled_dataset = dataset.shuffle(10000)
    # train_dataset = shuffled_dataset.take(int(y_dataset.shape[0]*0.8)).batch(BATCH_SIZE)
    # val_dataset = shuffled_dataset.skip(int(y_dataset.shape[0]*0.8))
    # val_dataset = val_dataset.take(int(y_dataset.shape[0])).batch(BATCH_SIZE)
    # positive_sample = dataset[(dataset[:,:,45]>0).any(axis=1)]
    # negative_sample = dataset[(dataset[:,:,45]==0).any(axis=1)]
    # negative_sample = negative_sample[np.random.choice(negative_sample.shape[0], positive_sample.shape[0])]
    # dataset = np.concatenate((positive_sample,negative_sample), axis=0)
    # y_dataset = np.zeros((dataset.shape[0],dataset.shape[1],3))
    # fire = dataset[:, :, 46] > 0
    # non_fire = dataset[:, :, 46] ==0
    # cloud = dataset[:,:,45] ==0
    # non_cloud = dataset[:,:,45] > 0
    # y_dataset[: ,:, 0] = np.logical_and(fire, non_cloud)
    # y_dataset[:, :, 1] = np.logical_and(non_fire, non_cloud)
    # y_dataset[:, :, 2] = cloud
    # np.save('y_dataset.npy', y_dataset)
    # x_train, x_test, y_train, y_test = train_test_split(dataset[:,:,:45], y_dataset, test_size=0.2)

    # input_shape = dataset[0,:,:45].shape
    # Patch parameters
    # visualizalize_patches()
    batch_size = 256
    dataset = np.load('../data/proj3_test.npy').transpose((1, 0, 2))
    print(dataset.shape)

    # positive_sample = dataset[(dataset[:,:,45]>0).any(axis=1)]
    # negative_sample = dataset[(dataset[:,:,45]==0).any(axis=1)]
    # negative_sample = negative_sample[np.random.choice(negative_sample.shape[0], positive_sample.shape[0])]
    # dataset = np.concatenate((positive_sample,negative_sample), axis=0)
    # y_dataset = np.zeros((dataset.shape[0],dataset.shape[1],2))
    # y_dataset[: ,:, 0] = dataset[:, :, 45] > 0
    # y_dataset[:, :, 1] = dataset[:, :, 45] == 0

    x_train, x_test, y_train, y_test = train_test_split(dataset[:,:,:5], dataset[:,:,:5], test_size=0.2)
    # dataset = np.load('../data/proj3_test_img.npy').transpose((2, 3, 0, 1))
    # print(dataset.shape)
    # shape = 112
    #
    # # positive_sample = dataset[(dataset[:,:,45]>0).any(axis=1)]
    # # negative_sample = dataset[(dataset[:,:,45]==0).any(axis=1)]
    # # negative_sample = negative_sample[np.random.choice(negative_sample.shape[0], positive_sample.shape[0])]
    # # dataset = np.concatenate((positive_sample,negative_sample), axis=0)
    # y_dataset = np.zeros((shape,shape,10,2))
    # y_dataset[: ,:, :, 0] = dataset[:shape,:shape,:,5] > 0
    # y_dataset[:, :, :, 1] = dataset[:shape,:shape,:,5] == 0
    # num_classes=2

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(batch_size)
    # image_size = 72  # We'll resize input images to this size
    # patch_size = 6  # Size of the patches to be extract from the input images
    # num_patches = (image_size // patch_size) ** 2

    # Transforer parameters
    projection_dim = 128
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 16

    # Size of the dense layers of the final classifier
    mlp_head_units = [2048, 1024]



    # x_train, x_test, y_train, y_test = train_test_split(dataset[np.newaxis, :,:,:,:5], dataset[np.newaxis, :,:,:,5], test_size=0.2)
    vit_gen = VisionTransformerGenerator((10, 45), projection_dim, transformer_layers, num_heads, mlp_head_units, 2)
    optimizer = tfa.optimizers.AdamW(
        learning_rate=0.001, weight_decay=0.0001
    )
    vit_gen.model.summary()

    # vit_gen.model.compile(
    #     optimizer=optimizer,
    #     loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    #     metrics=[
    #         keras.metrics.CategoricalAccuracy(name="accuracy")
    #     ],
    # )
    # loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
    # epochs = 2
    # for epoch in range(epochs):
    #     print("\nStart of epoch %d" % (epoch,))
    #     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    #         with tf.GradientTape() as tape:
    #             logits = vit_gen.model(x_batch_train, training=True)  # Logits for this minibatch
    #             loss_value = loss_fn(y_batch_train, logits)
    #         grads = tape.gradient(loss_value, vit_gen.model.trainable_weights)
    #         optimizer.apply_gradients(zip(grads, vit_gen.model.trainable_weights))
    #         if step % 200 == 0:
    #             print(
    #                 "Training loss (for one batch) at step %d: %.4f"
    #                 % (step, float(loss_value))
    #             )
    #             print("Seen so far: %s samples" % ((step + 1) * batch_size))

    # history = vit_gen.model.fit(
    #     x=dataset[np.newaxis,np.newaxis, :,:,:,:5],
    #     y=y_dataset[np.newaxis,np.newaxis, :,:,:,:],
    #     batch_size=1,
    #     epochs=20,
    # )
    # load_weights_numpy(vit_gen.model, '/Users/zhaoyu/PycharmProjects/ViirsTimeSeriesModel/weights/ViT-B_16.npz', False, 256, 256)
    # history = vit_gen.run_experiment(train_dataset, val_dataset, batch_size=BATCH_SIZE, num_epochs=MAX_EPOCHS, learning_rate=0.001, weight_decay=0.0001)
