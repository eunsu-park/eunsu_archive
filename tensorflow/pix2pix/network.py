import tensorflow as tf

def PixelDiscriminator(opt):
    inp = tf.keras.layers.Input(shape=(None, None, opt.ch_inp))
    tar = tf.keras.layers.Input(shape=(None, None, opt.ch_tar))
    layer = tf.keras.layers.Concatenate(-1) ([inp, tar])

    ndf = opt.nb_feat_init_D

    features = []
    Norm, use_bias = get_norm_layer(opt.norm_type)
    layer = Conv2D(ndf, kernel_size=1, strides=1,
                   padding='valid', use_bias=True) (layer)
    layer = LeakyReLU(0.2) (layer)
    features.append(layer)
    layer = Conv2D(ndf*2, kernel_size=1, strides=1,
                   padding='valid', use_bias=use_bias) (layer)
    layer = Norm() (layer)
    layer = LeakyReLU(0.2) (layer)
    features.append(layer)
    layer = Conv2D(1, kernel_size=1, strides=1,
                   padding='valid', use_bias=True) (layer)
    if opt.use_sigmoid == True :
        layer = tf.keras.layers.Activation('sigmoid') (layer)
    return tf.keras.Model(inputs=[inp, tar], outputs=[layer, features])