import tensorflow as tf

def get_norm_layer(type_norm='none'):
    if type_norm == 'batch' :
        norm = tf.keras.layers.BatchNormalization#(beta1=0.5, beta2=0.999, eps=1e-8)
        use_bias = False
    elif type_norm == 'none' :
        norm = Identity
        use_bias = True
    else :
        raise NotImplementedError('%s: invalid normalization type'%(type_norm))
    return norm, use_bias

def Conv2D(*a, **k):
    return tf.keras.layers.Conv2D(*a, **k)

def Conv2DTranspose(*a, **k):
    return tf.keras.layers.Conv2DTranspose(*a, **k)

def UpSampling2D(*a, **k):
    return tf.keras.layers.UpSampling2D(*a, **k)

def ReLU(*a, **k):
    return tf.keras.layers.ReLU(*a, **k)

def LeakyReLU(*a, **k):
    return tf.keras.layers.LeakyReLU(*a, **k)

def ZeroPadding2D(*a, **k):
    return tf.keras.layers.ZeroPadding2D(*a, **k)

def Cropping2D(*a, **k):
    return tf.keras.layers.Cropping2D(*a, **k)

class Identity(tf.keras.layers.Layer):
    def __init__(self, unused_param1=None, unused_param2=None, unused_param3=None):
        super(Identity, self).__init__()
    def call(self, inp):
        return inp

class ReflectPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding):
        super(ReflectPadding2D, self).__init__()
        if type(padding) == int :
            padding = (padding, padding)
        self.padding = ((0, 0), (padding[0], padding[0]), 
                        (padding[1], padding[1]), (0, 0))
    def call(self, inp):
        return tf.pad(inp, paddings=self.padding, mode='REFLECT')

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, ch_inp, type_norm='none'):
        super(ResidualBlock, self).__init__()
        Norm, use_bias = get_norm_layer(type_norm)
        self.Norm = Norm
        self.use_bias = use_bias

        self.conv1 = Conv2D(ch_inp, kernel_size=3, strides=1,
                             padding='valid', use_bias=use_bias)
        self.norm1 = Norm()
        self.conv2 = Conv2D(ch_inp, kernel_size=3, strides=1,
                             padding='valid', use_bias=use_bias)
        self.norm2 = Norm()

    def call(self, inp, training=False):
        x = self.conv1(inp)
        x = self.norm1(x)
        x = ReLU() (x)
        x = self.conv2(x)
        x = self.norm2(x)
        x += inp
        return x


def PatchDiscriminator(opt):
    inp = tf.keras.layers.Input(shape=(None, None, opt.ch_inp))
    tar = tf.keras.layers.Input(shape=(None, None, opt.ch_tar))
    layer = tf.keras.layers.Concatenate(-1) ([inp, tar])
    features = []
    Norm, use_bias = get_norm_layer(opt.type_norm)
    nb_feature = 64
    nb_feature_max = 512
    
    layer = ZeroPadding2D(1) (layer)
    layer = Conv2D(nb_feature, kernel_size=4, strides=2,
                   padding='valid', use_bias=True) (layer)
    layer = LeakyReLU(0.2) (layer)
    features.append(layer)

    for n in range(nb_layer - 1):
        nb_feature = min(nb_feature*2, nb_feature_max)
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(nb_feature, kernel_size=4, strides=2,
                       padding='valid', use_bias=use_bias) (layer)
        layer = Norm() (layer)
        layer = LeakyReLU(0.2) (layer)
        features.append(layer)

    nb_feature = min(nb_feature*2, nb_feature_max)
    layer = ZeroPadding2D(1) (layer)
    layer = Conv2D(nb_feature, kernel_size=4, strides=1,
                   padding='valid', use_bias=use_bias) (layer)
    layer = Norm() (layer)
    layer = LeakyReLU(0.2) (layer)
    features.append(layer)

    layer = ZeroPadding2D(1) (layer)
    layer = Conv2D(ch_tar, kernel_size=4, strides=1,
                   padding='valid', use_bias=True) (layer)
    if opt.use_sigmoid == True :
        layer = tf.keras.layers.Activation('sigmoid') (layer)
    return tf.keras.Model(inputs=[inp, tar], outputs=[layer, features])

def ResidualGenerator(opt):
    
    inp = tf.keras.layers.Input(shape=(None, None, opt.ch_inp))
    layer = inp
    Norm, use_bias = get_norm_layer(opt.type_norm)
    nb_feature = 64

    layer = ReflectPadding2D(3) (layer)
    layer = Conv2D(nb_feature, kernel_size=7, strides=1,
                   padding='valid', use_bias=use_bias) (layer)
    layer = Norm() (layer)
    layer = ReLU() (layer)

    for i in range(nb_down):
        nb_feature *= 2
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(nb_feature, kernel_size=3, strides=2,
                       padding='valid', use_bias=use_bias) (layer)
        layer = Norm() (layer)
        layer = ReLU() (layer)

    for j in range(nb_block):
        layer = ResidualBlock(nb_feature, type_norm=opt.type_norm) (layer)

    for k in range(nb_down):
        nb_feature //= 2
        layer = UpSampling2D(2) (layer)
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(nb_feature, kernel_size=3, strides=1,
                       padding='valid', use_bias=use_bias) (layer)
        layer = Norm() (layer)
        layer = ReLU() (layer)

    layer = ReflectPadding2D(3) (layer)
    layer = Conv2D(ch_tar, kernel_size=7, strides=1,
                   padding='valid', use_bias=True) (layer)

    if use_tanh == True :
        layer = tf.keras.layers.Activation('tanh') (layer)
    return tf.keras.Model(inputs=[inp], outputs=[layer])


if __name__ == '__main__' :
    from option import TrainOption
    opt = TrainOption().parse()
    network_D = PatchDiscriminator(opt)

    inp = torch.ones((opt.batch_size, opt.height, opt.width, opt.ch_inp))
    tar = torch.ones((opt.batch_size, opt.height, opt.width, opt.ch_tar))

    print(inp.shape, tar.shape)

    disk_D, features_D = network_D([inp, tar])
    print(disk_D.shape)