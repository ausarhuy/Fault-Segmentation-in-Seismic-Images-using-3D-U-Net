import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation, \
    Add, Concatenate, Dropout, \
    GlobalAveragePooling3D, Reshape, Dense, Multiply


class SEBlock(Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # assuming channels_last data format
        self.num_channels = input_shape[-1]
        self.reshape = Reshape((1, 1, 1, self.num_channels))
        self.dense1 = Dense(self.num_channels // self.reduction_ratio, activation='relu', use_bias=False)
        self.dense2 = Dense(self.num_channels, activation='sigmoid', use_bias=False)
        self.dense3 = Dense(self.num_channels)
        super(SEBlock, self).build(input_shape)

    def call(self, input_tensor):
        # Step 1: Squeeze operation
        se_tensor = GlobalAveragePooling3D()(input_tensor)
        # Step 2: Excitation operation
        # First FC layer (W1)
        se_tensor = self.reshape(se_tensor)
        se_tensor = self.dense1(se_tensor)  # W1*z, followed by ReLU
        # Second FC layer (W2)
        se_tensor = self.dense2(se_tensor)  # W2*(output from first FC layer), followed by sigmoid
        se_tensor = self.dense3(se_tensor)
        # Step 3: Scale the input
        output_tensor = Multiply()(
            [input_tensor, se_tensor])  # element-wise multiplication of input_tensor and se_tensor
        return output_tensor


def se_unet(input_shape=(None, None, None, 1)):
    inputs = Input(input_shape)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    se1 = SEBlock()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(se1)

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
    se2 = SEBlock()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(se2)

    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
    se3 = SEBlock()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(se3)

    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
    se4 = SEBlock()(conv4)

    up5 = Concatenate()([UpSampling3D(size=(2, 2, 2))(se4), se3])
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = Concatenate()([UpSampling3D(size=(2, 2, 2))(conv5), se2])
    conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate()([UpSampling3D(size=(2, 2, 2))(conv6), se1])
    conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv7)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=outputs, name='SE-Unet')
    # model.summary()
    return model


def inception_unet(input_shape=(None, None, None, 1)):
    inputs = Input(input_shape)

    def inception_block(input_tensor):
        conv11 = Conv3D(4, 3, activation='relu', padding='same')(input_tensor)
        conv121 = Conv3D(4, 3, activation='relu', padding='same')(input_tensor)
        conv122 = Conv3D(4, 3, activation='relu', padding='same')(conv121)
        conv131 = Conv3D(4, 3, activation='relu', padding='same')(input_tensor)
        conv132 = Conv3D(4, 3, activation='relu', padding='same')(conv131)
        concate1 = Concatenate(axis=-1)([conv11, conv122, conv132])
        conv2 = Conv3D(4, 3, activation='relu', padding='same')(concate1)
        return Concatenate(axis=-1)([input_tensor, conv2])

    # downsampling
    block1 = inception_block(inputs)  # 128
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(block1)  # 64
    block2 = inception_block(pool1)  # 64
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(block2)  # 32
    block3 = inception_block(pool2)  # 32
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(block3)  # 16
    block4 = inception_block(pool3)  # 16
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(block4)  # 8

    # upsampling
    up6 = Conv3D(32, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(pool4))  # 8
    merge6 = Concatenate(axis=-1)([block4, up6])
    conv6 = Conv3D(32, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv3D(32, 3, activation='relu', padding='same')(conv6)

    up7 = Conv3D(16, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv6))  # 16
    merge7 = Concatenate(axis=-1)([block3, up7])
    conv7 = Conv3D(16, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv3D(16, 3, activation='relu', padding='same')(conv7)

    up8 = Conv3D(8, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv7))  # 32
    merge8 = Concatenate(axis=-1)([block2, up8])
    conv8 = Conv3D(8, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv3D(8, 3, activation='relu', padding='same')(conv8)

    up9 = Conv3D(4, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv8))  # 64
    merge9 = Concatenate(axis=-1)([block1, up9])
    conv9 = Conv3D(4, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv3D(4, 3, activation='relu', padding='same')(conv9)
    outputs = Conv3D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs, name='Inception-Unet')
    # model.summary()
    return model


def bistream_se_unet(input_shape=(None, None, None, 1)):
    inputs = Input(input_shape)

    conv01 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv01 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(conv01)
    se01 = SEBlock()(conv01)
    conv11 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(se01)
    conv11 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv11)
    se11 = SEBlock()(conv11)
    pool11 = MaxPooling3D(pool_size=(2, 2, 2))(se11)

    conv21 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool11)
    conv21 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv21)
    se21 = SEBlock()(conv21)
    pool21 = MaxPooling3D(pool_size=(2, 2, 2))(se21)
    conv31 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool21)
    conv31 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv31)
    se31 = SEBlock()(conv31)
    drop31 = Dropout(0.5)(se31)
    pool31 = MaxPooling3D(pool_size=(2, 2, 2))(drop31)
    conv41 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool31)
    conv41 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv41)
    se41 = SEBlock()(conv41)
    drop41 = Dropout(0.5)(se41)

    conv02 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv02 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(conv02)
    se02 = SEBlock()(conv02)
    conv12 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(se02)
    conv12 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv12)
    se12 = SEBlock()(conv12)
    pool12 = MaxPooling3D(pool_size=(2, 2, 2))(se12)
    conv22 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool12)
    conv22 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv22)
    se22 = SEBlock()(conv22)
    pool22 = MaxPooling3D(pool_size=(2, 2, 2))(se22)
    conv32 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool22)
    conv32 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv32)
    se32 = SEBlock()(conv32)
    drop32 = Dropout(0.5)(se32)
    pool32 = MaxPooling3D(pool_size=(2, 2, 2))(drop32)
    conv42 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool32)
    conv42 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv42)
    se42 = SEBlock()(conv42)
    drop42 = Dropout(0.5)(se42)

    merge5 = Concatenate()([drop41, drop42])

    up6 = Conv3D(64, (2, 2, 2), activation='relu', padding='same')(UpSampling3D()(merge5))
    merge6 = Concatenate()([drop31, drop32, up6])
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv3D(32, (2, 2, 2), activation='relu', padding='same')(UpSampling3D()(conv6))
    merge7 = Concatenate()([se21, se22, up7])
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv3D(16, (2, 2, 2), activation='relu', padding='same')(UpSampling3D()(conv7))
    merge8 = Concatenate()([se11, se12, up8])
    conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(merge8)
    conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv3D(8, (2, 2, 2), activation='relu', padding='same')(conv8)
    merge9 = Concatenate()([se01, se02, up9])
    conv9 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(merge9)
    conv9 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=inputs, outputs=outputs, name='Bistream-SE-Unet')
    # model.summary()
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
