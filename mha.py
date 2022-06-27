import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import MultiHeadAttention

from model.vit_keras.layers import MultiHeadSelfAttention

if __name__ == '__main__':
    imat = np.random.rand(5, 5, 96)
    its = tf.convert_to_tensor(imat, dtype=tf.float32)
    mha = MultiHeadSelfAttention(num_heads=3)
    z = mha(its)

    print(z)