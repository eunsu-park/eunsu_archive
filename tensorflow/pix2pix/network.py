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

    layer = Concat(axis=-1) ([inp, tar])

    if opt.nb_layer_D == 0 :
        layer = Conv2D(filters=nb_feature, kernel_size=1, strides=1, use_bias=True) (layer)
        layer = LeakyReLU(0.2) (layer)
        nb_feature *= 2
        layer = Conv2D(filters=nb_feature, kernel_size=1, strides=1, use_bias=False) (layer)
        layer = BatchNorm2D() (layer)
        layer = LeakyReLU(0.2) (layer)
        nb_feature = 1
        layer = Conv2D(filters=nb_feature, kernel_size=1, strides=1, use_bias=True) (layer)

    elif opt.nb_layer_D > 0 :
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(filters=nb_feature, kernel_size=4, strides=2, use_bias=True) (layer)
        layer = LeakyReLU(0.2) (layer)

        for i in range(opt.nb_layer_D - 1) :
            nb_feature = min(nb_feature*2, opt.nb_feature_D_max)
            layer = ZeroPadding2D(1) (layer)
            layer = Conv2D(filters=nb_feature, kernel_size=4, strides=2, use_bias=False) (layer)
            layer = BatchNorm2D() (layer)
            layer = LeakyReLU(0.2) (layer)

        nb_feature = min(nb_feature*2, opt.nb_feature_D_max)
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(filters=nb_feature, kernel_size=4, strides=1, use_bias=False) (layer)
        layer = BatchNorm2D() (layer)
        layer = LeakyReLU(0.2) (layer)

        nb_feature = 1
        layer = ZeroPadding2D(1) (layer)
        layer = Conv2D(filters=nb_feature, kernel_size=4, strides=1, use_bias=True) (layer)

    if opt.use_sigmoid == True :
        layer = Sigmoid() (layer)

    model = tf.keras.Model(inputs=[inp, tar], outputs=[layer])
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



class TrainStep:
    def __init__(self, opt):
        self.weight_l1_loss = opt.weight_l1_loss
        if opt.type_gan == 'lsgan' :
            self.loss_function_gan = tf.keras.losses.MeanSquaredError()
        elif opt.type_gan == 'gan' :
            self.loss_function_gan = tf.keras.losses.BinaryCrossentropy()
        self.loss_function_l1 = tf.keras.losses.MeanAbsoluteError()

        lr_schedule_D = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=opt.initial_learning_rate,
            decay_steps=opt.decay_steps,
            decay_rate=opt.decay_rate,
            staircase=True)

        lr_schedule_G = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=opt.initial_learning_rate,
            decay_steps=opt.decay_steps,
            decay_rate=opt.decay_rate,
            staircase=True)

        self.optimizer_D = tf.keras.optimizers.Adam(learning_rate=lr_schedule_D,
            beta_1=opt.beta_1, beta_2=opt.beta_2, epsilon=opt.epsilon)
        self.optimizer_G = tf.keras.optimizers.Adam(learning_rate=lr_schedule_G,
            beta_1=opt.beta_1, beta_2=opt.beta_2, epsilon=opt.epsilon)

    def __call__(self, inp, tar, network_D, network_G):

        with tf.GradientTape() as tape_G, tf.GradientTape() as tape_D:

            gen = network_G(inputs=[inp], training=True)

            output_D_real = network_D(inputs=[inp, tar], training=True)
            output_D_fake = network_D(inputs=[inp, gen], training=True)
            
            loss_D_real = self.loss_function_gan(tf.ones_like(output_D_real[0]), output_D_real[0])
            loss_D_fake = self.loss_function_gan(tf.zeros_like(output_D_fake[0]), output_D_fake[0])
            loss_D = (loss_D_real + loss_D_fake)/2.

            loss_G_fake = self.loss_function_gan(tf.ones_like(output_D_fake[0]), output_D_fake[0])
            loss_L = self.loss_function_l1(tar, gen) * self.weight_l1_loss
            loss_G = loss_G_fake + loss_L

        gradient_G = tape_G.gradient(loss_G, network_G.trainable_variables)
        gradient_D = tape_D.gradient(loss_D, network_D.trainable_variables)

        self.optimizer_G.apply_gradients(zip(gradient_G, network_G.trainable_variables))
        self.optimizer_D.apply_gradients(zip(gradient_D, network_D.trainable_variables))

        return loss_D, loss_G_fake, loss_L
        

if __name__ == '__main__' :

    from option import TrainOption
    opt = TrainOption().parse()

    network_D = Discriminator(opt)
    network_G = Generator(opt)
    train_step = TrainStep(opt)

    inp = tf.ones((opt.batch_size, opt.height, opt.width, opt.ch_inp), dtype=tf.float32)
    tar = tf.ones((opt.batch_size, opt.height, opt.width, opt.ch_tar), dtype=tf.float32)

    gen = network_G([inp])
    print(gen.shape)
    output_D_real = network_D([inp, tar])
    print(output_D_real.shape)
    output_D_fake = network_D([inp, gen])
    print(output_D_fake.shape)
    print(gen[0,0:4,0:4,0])

    train_step(inp, tar, network_D, network_G)
    gen = network_G([inp])
    print(gen[0,0:4,0:4,0])
    
    train_step(inp, tar, network_D, network_G)
    gen = network_G([inp])
    print(gen[0,0:4,0:4,0])
