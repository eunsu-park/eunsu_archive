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

    Ds = []
    outputs = []
    features = []
    for n in range(opt.nb_D):
        Ds.append(Discriminator(opt))
    for n in range(opt.nb_D):
        if n > 0 :
            inp = AveragePooling2D(2) (inp)
            tar = AveragePooling2D(2) (tar)
        output, feature = Ds[n] ([inp, tar])
        outputs.append(output)
        features.append(feature)

    model = tf.keras.Model(inputs=[inp, tar], outputs=[outputs, features])
    model.summary()
    return model


if __name__ == '__main__' :
    from option import TrainOption
    opt = TrainOption().parse()
    network_D = MultiDiscriminator(opt)
    
    inp = torch.ones((opt.batch_size, opt.height, opt.width, opt.ch_inp))
    tar = torch.ones((opt.batch_size, opt.height, opt.width, opt.ch_tar))

    print(inp.shape, tar.shape)

    disk_D, features_D = network_D([inp, tar])
    for n in range(opt.nb_D):
        disk_D_ = disk_D[n]
        features_D_ = features_D[n]
        print(disk_D_.shape, len(features_D_))
