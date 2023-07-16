import tensorflow as tf
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import Loss


def unet(input_size=(None, None, None, 1)):
    inputs = Input(input_size)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv7)

    conv8 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])
    model.summary()
    # model.compile(optimizer = Adam(lr = 1e-4),
    # loss = cross_entropy_balanced, metrics = ['accuracy'])
    return model


class CrossEntropyBalanced(Loss):
    def __init__(self, name='cross_entropy_balanced'):
        super(CrossEntropyBalanced, self).__init__(name=name)

    def call(self, y_true, y_pred):
        _epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), dtype=y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
        y_pred = tf.math.log(y_pred / (1 - y_pred))

        y_true = tf.cast(y_true, tf.float32)

        count_neg = tf.reduce_sum(1. - y_true)
        count_pos = tf.reduce_sum(y_true)

        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1 - beta)

        cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

        cost = tf.reduce_mean(cost * (1 - beta))

        return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)
