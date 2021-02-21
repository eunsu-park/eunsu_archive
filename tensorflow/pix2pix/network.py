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


def Discriminator(opt):

    inp = Input(shape=(None, None, opt.ch_inp), dtype=tf.float32)
    tar = Input(shape=(None, None, opt.ch_tar), dtype=tf.float32)

    nb_feature = opt.nb_feature_D_init
    features = []

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



def Generator(opt):

    inp = Input(shape=(None, None, opt.ch_inp), dtype=tf.float32)
    nb_features = []
    layers = []

    nb_feature = opt.nb_feature_G_init
    nb_features.append(nb_feature)
    layer = inp
    layer = ZeroPadding2D(1) (layer)
    layer = Conv2D(filters=nb_feature, kernel_size=4, strides=2, use_bias=True) (layer)
    layers.append(layer)

    for e in range(opt.nb_down_G - 2):
        layer = LeakyReLU(0.2) (layer)
        nb_feature = min(nb_feature*2, opt.nb_feature_G_max)
        nb_features.append(nb_feature)
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(filters=nb_feature, kernel_size=4, strides=2, use_bias=False) (layer)
        layer = BatchNorm2D() (layer)
        layers.append(layer)

    layer = LeakyReLU(0.2) (layer)
    nb_feature = min(nb_feature*2, opt.nb_feature_G_max)
    layer = ZeroPadding2D(1) (layer)
    layer = Conv2D(filters=nb_feature, kernel_size=4, strides=2, use_bias=True) (layer)
    layer = ReLU() (layer)

    layer = Conv2DTranspose(filters=nb_feature, kernel_size=4, strides=2, use_bias=False) (layer)
    layer = Cropping2D(1) (layer)
    layer = BatchNorm2D() (layer)
    layer = Dropout(0.5) (layer)

    layers = list(reversed(layers))
    nb_features = list(reversed(nb_features[:-1]))

    for d in range(opt.nb_down_G - 2) :
        layer = Concat(axis=-1) ([layer, layers[d]])
        nb_feature = nb_features[d]
        layer = ReLU() (layer)
        layer = Conv2DTranspose(filters=nb_feature, kernel_size=4, strides=2, use_bias=False) (layer)
        layer = Cropping2D(1) (layer)
        layer = BatchNorm2D() (layer)
        if d < 2 :
            layer = Dropout(0.5) (layer)

    layer = Concat(axis=-1) ([layer, layers[-1]])
    nb_feature = opt.ch_tar
    layer = ReLU() (layer)
    layer = Conv2DTranspose(filters=nb_feature, kernel_size=4, strides=2, use_bias=True) (layer)
    layer = Cropping2D(1) (layer)
    
    if opt.use_tanh == True :
        layer = Tanh (layer)
    
    model = tf.keras.Model(inputs = [inp], outputs = [layer])
    model.summary()

    return model


if __name__ == '__main__' :

    from option import TrainOption
    opt = TrainOption().parse()

    network_D = Discriminator(opt)
    network_G = Generator(opt)

    inp = tf.zeros((opt.batch_size, opt.height, opt.width, opt.ch_inp), dtype=tf.float32)
    tar = tf.zeros((opt.batch_size, opt.height, opt.width, opt.ch_tar), dtype=tf.float32)

    gen = network_G([inp])
    print(gen.shape)
    output_D_real = network_D([inp, tar])
    print(output_D_real[0].shape, len(output_D_real[1]))
    output_D_fake = network_D([inp, gen])
    print(output_D_fake[0].shape, len(output_D_fake[1]))

    
    