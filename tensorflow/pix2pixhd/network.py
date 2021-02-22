import tensorflow as tf


def Input(*a, **k):
    return tf.keras.layers.Input(*a, **k)

def Concat(*a, **k):
    return tf.keras.layers.Concatenate(*a, **k) 

def Conv2D(*a, **k):
    return tf.keras.layers.Conv2D(*a, **k)

def Conv2DTranspose(*a, **k):
    return tf.keras.layers.Conv2DTranspose(*a, **k)

def BatchNorm2D(*a, **k):
    return tf.keras.layers.BatchNormalization(*a, **k)#(beta1=0.5, beta2=0.999, eps=1e-8)

def LeakyReLU(*a, **k):
    return tf.keras.layers.LeakyReLU(*a, **k)

def ReLU(*a, **k):
    return tf.keras.layers.ReLU(*a, **k)

def Sigmoid(*a, **k):
    return tf.keras.layers.Activation('sigmoid', *a, **k)

def Tanh(*a, **k):
    return tf.keras.layers.Activation('tanh', *a, **k)

def ZeroPadding2D(*a, **k):
    return tf.keras.layers.ZeroPadding2D(*a, **k)

def Cropping2D(*a, **k):
    return tf.keras.layers.Cropping2D(*a, **k)

def Dropout(*a, **k):
    return tf.keras.layers.Dropout(*a, **k)

def AveragePooling2D(*a, **k):
    return tf.keras.layers.AveragePooling2D(*a, **k)

def UpSampling2D(*a, **k):
    return tf.keras.layers.UpSampling2D(*a, **k)


class Identity(tf.keras.layers.Layer):
    def __init__(self, unused_param1=None, unused_param2=None, unused_param3=None):
        super(Identity, self).__init__()
    def call(self, inp):
        return inp


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, ch_inp):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2D(ch_inp, kernel_size=3, strides=1,
                             padding='valid', use_bias=False)
        self.norm1 = BatchNorm2D()
        self.conv2 = Conv2D(ch_inp, kernel_size=3, strides=1,
                             padding='valid', use_bias=False)
        self.norm2 = BatchNorm2D()

    def call(self, inp):
        x = ZeroPadding2D(1) (inp)
        x = self.conv1(x)
        x = self.norm1(x)
        x = ReLU() (x)
        x = ZeroPadding2D(1) (x)
        x = self.conv2(x)
        x = self.norm2(x)
        x += inp
        return x


class ReflectPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding):
        super(ReflectPadding2D, self).__init__()
        if type(padding) == int :
            padding = (padding, padding)
        self.padding = ((0, 0), (padding[0], padding[0]), 
                        (padding[1], padding[1]), (0, 0))
    def call(self, inp):
        return tf.pad(inp, paddings=self.padding, mode='REFLECT')


def Discriminator(opt):
    
    inp = Input(shape=(None, None, opt.ch_inp), dtype=tf.float32)
    tar = Input(shape=(None, None, opt.ch_tar), dtype=tf.float32)
    features = []

    nb_feature = opt.nb_feature_D_init

    layer = Concat(axis=-1) ([inp, tar])

    if opt.nb_layer_D == 0 :
        layer = Conv2D(filters=nb_feature, kernel_size=1, strides=1, use_bias=True) (layer)
        layer = LeakyReLU(0.2) (layer)
        features.append(layer)
        nb_feature *= 2
        layer = Conv2D(filters=nb_feature, kernel_size=1, strides=1, use_bias=False) (layer)
        layer = BatchNorm2D() (layer)
        layer = LeakyReLU(0.2) (layer)
        features.append(layer)
        nb_feature = 1
        layer = Conv2D(filters=nb_feature, kernel_size=1, strides=1, use_bias=True) (layer)

    elif opt.nb_layer_D > 0 :
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(filters=nb_feature, kernel_size=4, strides=2, use_bias=True) (layer)
        layer = LeakyReLU(0.2) (layer)
        features.append(layer)

        for i in range(opt.nb_layer_D - 1) :
            nb_feature = min(nb_feature*2, opt.nb_feature_D_max)
            layer = ZeroPadding2D(1) (layer)
            layer = Conv2D(filters=nb_feature, kernel_size=4, strides=2, use_bias=False) (layer)
            layer = BatchNorm2D() (layer)
            layer = LeakyReLU(0.2) (layer)
            features.append(layer)

        nb_feature = min(nb_feature*2, opt.nb_feature_D_max)
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(filters=nb_feature, kernel_size=4, strides=1, use_bias=False) (layer)
        layer = BatchNorm2D() (layer)
        layer = LeakyReLU(0.2) (layer)
        features.append(layer)

        nb_feature = 1
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(filters=nb_feature, kernel_size=4, strides=1, use_bias=True) (layer)

    if opt.use_sigmoid == True :
        layer = Sigmoid() (layer)

    model = tf.keras.Model(inputs=[inp, tar], outputs=[layer, features])
    model.summary()
    return model


def MultiDiscriminator(opt):
    inp = Input(shape=(None, None, opt.ch_inp), dtype=tf.float32)
    tar = Input(shape=(None, None, opt.ch_tar), dtype=tf.float32)

    inp_layer = inp
    tar_layer = tar

    Ds = []
    outputs = []
    features = []
    for n in range(opt.nb_D):
        Ds.append(Discriminator(opt))
    for n in range(opt.nb_D):
        if n > 0 :
            inp_layer = AveragePooling2D(2) (inp_layer)
            tar_layer = AveragePooling2D(2) (tar_layer)
        output, feature = Ds[n] ([inp_layer, tar_layer])
        outputs.append(output)
        features.append(feature)

    model = tf.keras.Model(inputs=[inp, tar], outputs=[outputs, features])
    model.summary()
    return model


def ResidualGenerator(opt):
    
    inp = tf.keras.layers.Input(shape=(None, None, opt.ch_inp))
    nb_feature = opt.nb_feature_G_init

    layer = ReflectPadding2D(3) (inp)
    layer = Conv2D(nb_feature, kernel_size=7, strides=1,
                   padding='valid', use_bias=False) (layer)
    layer = BatchNorm2D() (layer)
    layer = ReLU() (layer)

    for i in range(opt.nb_down_G):
        nb_feature *= 2
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(nb_feature, kernel_size=3, strides=2,
                       padding='valid', use_bias=False) (layer)
        layer = BatchNorm2D() (layer)
        layer = ReLU() (layer)

    for j in range(opt.nb_block_G):
        layer = ResidualBlock(nb_feature) (layer)

    for k in range(opt.nb_down_G):
        nb_feature //= 2
        layer = UpSampling2D(2) (layer)
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(nb_feature, kernel_size=3, strides=1,
                       padding='valid', use_bias=False) (layer)
        layer = BatchNorm2D() (layer)
        layer = ReLU() (layer)

    layer = ReflectPadding2D(3) (layer)
    layer = Conv2D(opt.ch_tar, kernel_size=7, strides=1,
                   padding='valid', use_bias=True) (layer)

    if opt.use_tanh == True :
        layer = Tanh() (layer)
    
    model = tf.keras.Model(inputs=[inp], outputs=[layer])
    model.summary()
    return model


if __name__ == '__main__' :
    from option import TrainOption
    opt = TrainOption().parse()
    network_D = MultiDiscriminator(opt)
    network_G = ResidualGenerator(opt)
    
    inp = tf.ones((opt.batch_size, opt.height, opt.width, opt.ch_inp))
    tar = tf.ones((opt.batch_size, opt.height, opt.width, opt.ch_tar))
    gen = network_G([inp])

    print(inp.shape, tar.shape, gen.shape)

    disk_D, features_D = network_D([inp, tar])
    for n in range(opt.nb_D):
        disk_D_ = disk_D[n]
        features_D_ = features_D[n]
        print(disk_D_.shape, len(features_D_))

