import tensorflow as tf
from tensorflow import keras

from model.tokenize_layer.tokenize_layer import TokenizeLayer
from model.vit.utilities.patch_encoder import PatchEncoder
from model.vit.utilities.patches import Patches
from tensorflow.keras import layers
import tensorflow_addons as tfa
import keras.backend as K
class VisionTransformerGenerator:
    def __init__(self, input_shape, projection_dim, transformer_layers, num_heads, mlp_head_units, num_classes):
        # dataset should be N*h*l*c
        self.input_shape = input_shape
        self.projection_dim = projection_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.mlp_head_units = mlp_head_units
        self.num_classes = num_classes
        # self.resize_size = resize_size
        # self.patch_size = patch_size

        # data augmentation and adaption on training data
        # self.data_augmentation_layer = self.data_augmentation()
        # self.data_augmentation_layer.layers[0].adapt(train_dataset)
        self.model = self.create_vit_classifier()

    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    # def data_augmentation(self):
    #     data_augmentation = keras.Sequential(
    #         [
    #             layers.Normalization(),
    #             layers.Resizing(self.resize_size, self.resize_size),
    #             layers.RandomFlip("horizontal"),
    #             layers.RandomRotation(factor=0.02),
    #             layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    #         ],
    #         name="data_augmentation",
    #     )
    #     return data_augmentation


    def create_vit_classifier(self, batch_size=1):
        inputs = layers.Input(shape=self.input_shape)
        num_timestamps = 10
        # Augment data.
        # augmented = self.data_augmentation_layer(inputs)
        # Create patches.
        # patches = Patches(self.patch_size)(augmented)
        # Encode patches.
        # num_patches = (self.resize_size // self.patch_size) ** 2
        # tokenized_images = TokenizeLayer(3, batch_size)(inputs)
        encoded_patches = PatchEncoder(num_timestamps, self.projection_dim)(inputs)
        transformer_units = [
            self.projection_dim * 2,
            self.projection_dim,
        ]
        # Create multiple layers of the Transformer block.
        for _ in range(self.transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = self.mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        # representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # representation = layers.Flatten()(representation)
        # representation = layers.Dropout(0.5)(representation)
        # Add MLP for classification.
        # features = self.mlp(encoded_patches, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(self.num_classes)(encoded_patches)
        logits = tf.reshape(logits, (-1, self.input_shape[0], self.input_shape[1], 10, 2))
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def run_experiment(self, x_train, y_train, batch_size=256, num_epochs=100, learning_rate=0.001, weight_decay=0.0001, callbacks=[]):
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                self.f1_m
                # keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        checkpoint_filepath = "/tmp/checkpoint"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
        callbacks.append(checkpoint_callback)

        history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_split=0.1,
            callbacks=callbacks,
        )

        self.model.load_weights(checkpoint_filepath)
        # _, accuracy, top_5_accuracy = self.model.evaluate(x_test, y_test)
        # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        return history