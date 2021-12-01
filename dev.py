import keras.backend as K

from vit_keras import vit

# x_dataset = np.load('/NOBACKUP/zhao2/proj3_train_5_channel.npy').transpose((1,0,2))
# y_dataset = np.zeros((x_dataset.shape[0],x_dataset.shape[1],2))
# y_dataset[: ,:, 0] = x_dataset[:, :, 45] == 0
# y_dataset[:, :, 1] = x_dataset[:, :, 45] > 0
#
# x_train, x_test, y_train, y_test = train_test_split(x_dataset[:,:,:45], y_dataset, test_size=0.2)

# print(x_train.shape)
# print(y_train.shape)
batch_size=256
MAX_EPOCHS = 20
learning_rate = 0.001
weight_decay = 0.0001

num_classes=3
input_shape=(10,45)

# wandb.login()
# wandb.init(project="vit_model", entity="zhaoyutim")
# wandb.config = {
#   "learning_rate": learning_rate,
#   "weight_decay": weight_decay,
#   "epochs": MAX_EPOCHS,
#   "batch_size": batch_size,
#   "num_heads":num_heads,
#   "transformer_layers": transformer_layers
# }

model = vit.vit_small(
    input_shape=input_shape,
    classes=2,
    activation='sigmoid',
    pretrained=True,
    include_top=True,
    pretrained_top=True
)
model.summary()
# optimizer = tfa.optimizers.AdamW(
#     learning_rate=learning_rate, weight_decay=weight_decay
# )
#
# model.compile(
#     optimizer=optimizer,
#     loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=False, gamma=0.01),
#     metrics=[
#         tf.keras.metrics.CategoricalAccuracy(name="accuracy")
#     ],
# )
# history = model.fit(
#     x=x_train,
#     y=y_train,
#     batch_size=batch_size,
#     epochs=MAX_EPOCHS,
#     validation_split=0.1,
#     callbacks=[WandbCallback()],
# )
# model.save('/NOBACKUP/zhao2/proj3_transformer_b16_pretrained')