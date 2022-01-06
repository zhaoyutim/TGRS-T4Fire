import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.vit_keras import vit

if __name__=='__main__':
    input_shape=(10,45)
    num_classes=2
    model = vit.vit_base(
        input_shape=input_shape,
        classes=num_classes,
        activation='sigmoid',
        pretrained=False,
        include_top=True,
        pretrained_top=True
    )
    model.summary()

    model_weights=tf.keras.Model(model.input, model.layers[-3].output)
    weights=model_weights.predict(np.zeros((1,10,45)))
    att = weights[1][0, 0, :, :]
    plt.imshow(att)
    plt.colorbar()
    plt.show()