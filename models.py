from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Input, Conv3D, MaxPooling3D, UpSampling3D, BatchNormalization, Activation, \
    Add, Concatenate, Dropout, \
    GlobalAveragePooling3D, Reshape, Dense, Multiply


class SEBlock(Layer):
    def __init__(self, reduction_ratio, **kwargs):
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

    def call(self, inputs):
        # Step 1: Squeeze operation
        squeeze = GlobalAveragePooling3D()(inputs)
        # Step 2: Excitation operation
        # First FC layer (W1)
        excitation = self.reshape(squeeze)
        excitation = self.dense1(excitation)  # W1*z, followed by ReLU
        # Second FC layer (W2)
        excitation = self.dense2(excitation)  # W2*(output from first FC layer), followed by sigmoid
        excitation = self.dense3(excitation)
        # Step 3: Scale the inputs
        scale = Multiply()([inputs, excitation])  # element-wise multiplication of inputs and squeeze-and-excitation 
        return scale


def se_unet(input_shape=(None, None, None, 1)) -> Model:
    inputs = Input(input_shape)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    se1 = SEBlock(reduction_ratio=8)(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(se1)

    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
    se2 = SEBlock(reduction_ratio=8)(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(se2)

    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
    se3 = SEBlock(reduction_ratio=8)(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(se3)

    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
    se4 = SEBlock(reduction_ratio=8)(conv4)

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

    return Model(inputs=inputs, outputs=outputs, name='SE-U-Net')


def inception_unet(input_shape=(None, None, None, 1)) -> Model:
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

    return Model(inputs=inputs, outputs=outputs, name='Inception-U-Net')


def bistream_unet(input_shape=(None, None, None, 1)) -> Model:
    inputs = Input(input_shape)

    # branch 1
    conv11 = Conv3D(4, 3, activation='relu', padding='same')(inputs)
    conv11 = Conv3D(4, 3, activation='relu', padding='same')(conv11)
    pool11 = MaxPooling3D(pool_size=(2, 2, 2))(conv11)
    conv21 = Conv3D(8, 3, activation='relu', padding='same')(pool11)
    conv21 = Conv3D(8, 3, activation='relu', padding='same')(conv21)
    pool21 = MaxPooling3D(pool_size=(2, 2, 2))(conv21)
    conv31 = Conv3D(16, 3, activation='relu', padding='same')(pool21)
    conv31 = Conv3D(16, 3, activation='relu', padding='same')(conv31)
    pool31 = MaxPooling3D(pool_size=(2, 2, 2))(conv31)
    conv41 = Conv3D(32, 3, activation='relu', padding='same')(pool31)
    conv41 = Conv3D(32, 3, activation='relu', padding='same')(conv41)
    drop41 = Dropout(0.5)(conv41)
    pool41 = MaxPooling3D(pool_size=(2, 2, 2))(drop41)
    conv51 = Conv3D(64, 3, activation='relu', padding='same')(pool41)
    conv51 = Conv3D(64, 3, activation='relu', padding='same')(conv51)
    drop51 = Dropout(0.5)(conv51)

    # branch 2
    conv12 = Conv3D(4, 3, activation='relu', padding='same')(inputs)
    conv12 = Conv3D(4, 3, activation='relu', padding='same')(conv12)
    pool12 = MaxPooling3D(pool_size=(2, 2, 2))(conv12)
    conv22 = Conv3D(8, 3, activation='relu', padding='same')(pool12)
    conv22 = Conv3D(8, 3, activation='relu', padding='same')(conv22)
    pool22 = MaxPooling3D(pool_size=(2, 2, 2))(conv22)
    conv32 = Conv3D(16, 3, activation='relu', padding='same')(pool22)
    conv32 = Conv3D(16, 3, activation='relu', padding='same')(conv32)
    pool32 = MaxPooling3D(pool_size=(2, 2, 2))(conv32)
    conv42 = Conv3D(32, 3, activation='relu', padding='same')(pool32)
    conv42 = Conv3D(32, 3, activation='relu', padding='same')(conv42)
    drop42 = Dropout(0.5)(conv42)
    pool42 = MaxPooling3D(pool_size=(2, 2, 2))(drop42)
    conv52 = Conv3D(64, 3, activation='relu', padding='same')(pool42)
    conv52 = Conv3D(64, 3, activation='relu', padding='same')(conv52)
    drop52 = Dropout(0.5)(conv52)

    # upsampling
    merge5 = Concatenate(axis=-1)([drop51, drop52])
    up6 = Conv3D(32, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(merge5))
    merge6 = Concatenate(axis=-1)([drop41, drop42, up6])
    conv6 = Conv3D(32, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv3D(32, 3, activation='relu', padding='same')(conv6)

    up7 = Conv3D(16, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = Concatenate(axis=-1)([conv31, conv32, up7])
    conv7 = Conv3D(16, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv3D(16, 3, activation='relu', padding='same')(conv7)

    up8 = Conv3D(8, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = Concatenate(axis=-1)([conv21, conv22, up8])
    conv8 = Conv3D(8, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv3D(8, 3, activation='relu', padding='same')(conv8)

    up9 = Conv3D(4, 2, activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv8))
    merge9 = Concatenate(axis=-1)([conv11, conv12, up9])
    conv9 = Conv3D(4, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv3D(4, 3, activation='relu', padding='same')(conv9)

    outputs = Conv3D(1, 1, activation='sigmoid')(conv9)

    return Model(inputs=inputs, outputs=outputs, name='Bi-stream-U-Net')


def bistream_se_unet(input_shape=(None, None, None, 1)) -> Model:
    inputs = Input(input_shape)

    conv01 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv01 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(conv01)
    se01 = SEBlock(reduction_ratio=8)(conv01)
    conv11 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(se01)
    conv11 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv11)
    se11 = SEBlock(reduction_ratio=8)(conv11)
    pool11 = MaxPooling3D(pool_size=(2, 2, 2))(se11)
    conv21 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool11)
    conv21 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv21)
    se21 = SEBlock(reduction_ratio=8)(conv21)
    pool21 = MaxPooling3D(pool_size=(2, 2, 2))(se21)
    conv31 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool21)
    conv31 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv31)
    se31 = SEBlock(reduction_ratio=8)(conv31)
    drop31 = Dropout(0.5)(se31)
    pool31 = MaxPooling3D(pool_size=(2, 2, 2))(drop31)
    conv41 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool31)
    conv41 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv41)
    se41 = SEBlock(reduction_ratio=8)(conv41)
    drop41 = Dropout(0.5)(se41)

    conv02 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv02 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(conv02)
    se02 = SEBlock(reduction_ratio=8)(conv02)
    conv12 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(se02)
    conv12 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv12)
    se12 = SEBlock(reduction_ratio=8)(conv12)
    pool12 = MaxPooling3D(pool_size=(2, 2, 2))(se12)
    conv22 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool12)
    conv22 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv22)
    se22 = SEBlock(reduction_ratio=8)(conv22)
    pool22 = MaxPooling3D(pool_size=(2, 2, 2))(se22)
    conv32 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool22)
    conv32 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv32)
    se32 = SEBlock(reduction_ratio=8)(conv32)
    drop32 = Dropout(0.5)(se32)
    pool32 = MaxPooling3D(pool_size=(2, 2, 2))(drop32)
    conv42 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool32)
    conv42 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv42)
    se42 = SEBlock(reduction_ratio=8)(conv42)
    drop42 = Dropout(0.5)(se42)

    merge5 = Concatenate()([drop41, drop42])

    up6 = Conv3D(64, (2, 2, 2), activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(merge5))
    merge6 = Concatenate()([drop31, drop32, up6])
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge6)
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv3D(32, (2, 2, 2), activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv6))
    merge7 = Concatenate()([se21, se22, up7])
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge7)
    conv7 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv3D(16, (2, 2, 2), activation='relu', padding='same')(UpSampling3D(size=(2, 2, 2))(conv7))
    merge8 = Concatenate()([se11, se12, up8])
    conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(merge8)
    conv8 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv3D(8, (2, 2, 2), activation='relu', padding='same')(conv8)
    merge9 = Concatenate()([se01, se02, up9])
    conv9 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(merge9)
    conv9 = Conv3D(8, (3, 3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

    return Model(inputs=inputs, outputs=outputs, name='Bi-stream-SE-U-Net')
