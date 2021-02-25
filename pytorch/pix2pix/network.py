import torch
import torch.nn as nn
from torch.nn import init

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None :
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None :
            nn.init.zeros_(m.bias)


class PixelDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PixelDiscriminator, self).__init__()
        self.opt = opt
        self.build()
        print(self)

    def build(self):
        nb_feat = self.opt.nb_feature_init_D
        norm = get_norm_layer(self.opt.type_norm)

        block = [nn.Conv2d(self.opt.ch_inp+self.opt.ch_tar, nb_feat, kernel_size=1, stride=1, padding=0),
                 nn.LeakyReLU(0.2),
                 nn.Conv2d(nb_feat, nb_feat*2, kernel_size=1, stride=1, padding=0),
                 norm(nb_feat*2), nn.LeakyReLU(0.2),
                 nn.Conv2d(nb_feat*2, 1, kernel_size=1, stride=1, padding=0)]
        if self.opt.use_sigmoid :
            block += [nn.Sigmoid()]
        self.block = nn.Sequential(*block)

    def forward(self, inp):
        return self.block(inp)


class PatchDiscriminator(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator, self).__init__()
        self.opt = opt
        self.build()
        print(self)

    def build(self):
        nb_feat = self.opt.nb_feature_init_D
        norm = get_norm_layer(self.opt.type_norm)
        
        blocks = []
        block = [nn.Conv2d(self.opt.ch_inp+self.opt.ch_tar, nb_feat, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2)]
        blocks.append(block)

        for n in range(1, self.opt.nb_layer_D):
            block = [nn.Conv2d(nb_feat, nb_feat*2, kernel_size=4, stride=2, padding=1),
                     norm(nb_feat*2), nn.LeakyReLU(0.2)]
            blocks.append(block)
            nb_feat *= 2

        block = [nn.Conv2d(nb_feat, nb_feat*2, kernel_size=4, stride=1, padding=1),
                 norm(nb_feat*2), nn.LeakyReLU(0.2)]
        blocks.append(block)
        nb_feat *= 2

        block = [nn.Conv2d(nb_feat, 1, kernel_size=4, stride=1, padding=1)]
        if self.opt.use_sigmoid :
            block += [nn.Sigmoid()]
        blocks.append(block)

        self.nb_blocks = len(blocks)
        for i in range(self.nb_blocks):
            setattr(self, 'block_%d'%(i), nn.Sequential(*blocks[i]))

    def forward(self, inp):
        result = [inp]
        for n in range(self.nb_blocks):
            block = getattr(self, 'block_%d'%(n))
            result.append(block(result[-1]))
        return result[1:]


class UNetGenerator(nn.Module):
    def __init__(self, opt):
        super(UNetGenerator, self).__init__()

        self.build()
        print(self)

    def build(self, opt):
        nb_feature = opt.nb_feature_G_init
        self.dconv1 = nn.Conv2d(opt.ch_inp, nb_feature, kernel_size=4, strides=2, use_bias=True)
        self.dconv2 = nn.Conv2d(nb_feature, 2*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.dnorm2 = nn.BatchNorm2d(2*nb_feature)
        self.dconv3 = nn.Conv2d(2*nb_feature, 4*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.dnorm3 = nn.BatchNorm2d(4*nb_feature)
        self.dconv4 = nn.Conv2d(4*nb_feature, 8*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.dnorm4 = nn.BatchNorm2d(8*nb_feature)
        self.dconv5 = nn.Conv2d(8*nb_feature, 8*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.dnorm5 = nn.BatchNorm2d(8*nb_feature)
        self.dconv6 = nn.Conv2d(8*nb_feature, 8*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.dnorm6 = nn.BatchNorm2d(8*nb_feature)
        self.dconv7 = nn.Conv2d(8*nb_feature, 8*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.dnorm7 = nn.BatchNorm2d(8*nb_feature)
        self.dconv8 = nn.Conv2d(8*nb_feature, 8*nb_feature, kernel_size=4, strides=2, use_bias=True)

        self.uconv8 = nn.ConvTranspose2d(8*nb_feature, 8*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.unorm8 = nn.BatchNorm2d(8*nb_feature)
        self.uconv7 = nn.ConvTranspose2d(8*nb_feature, 8*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.unorm7 = nn.BatchNorm2d(8*nb_feature)
        self.uconv6 = nn.ConvTranspose2d(8*nb_feature, 8*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.unorm6 = nn.BatchNorm2d(8*nb_feature)
        self.uconv5 = nn.ConvTranspose2d(8*nb_feature, 8*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.unorm5 = nn.BatchNorm2d(8*nb_feature)
        self.uconv4 = nn.ConvTranspose2d(8*nb_feature, 4*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.unorm4 = nn.BatchNorm2d(4*nb_feature)
        self.uconv3 = nn.ConvTranspose2d(4*nb_feature, 2*nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.unorm3 = nn.BatchNorm2d(2*nb_feature)
        self.uconv2 = nn.ConvTranspose2d(2*nb_feature, nb_feature, kernel_size=4, strides=2, use_bias=False)
        self.unorm2 = nn.BatchNorm2d(nb_feature)
        self.uconv1 = nn.ConvTranspose2d(nb_feature, opt.ch_tar, kernel_size=4, strides=2, use_bias=True)

    def forward(self, inp):
        layer = self.dconv1(inp)
        layer = nn.LeakyReLU(0.2) (layer)
        layer = self.dconv





        

        self.parser.add_argument('--nb_feature_G_init', type=int, default=64)
        self.parser.add_argument('--nb_feature_G_max', type=int, default=512)
        self.parser.add_argument('--use_tanh', type=bool, default=False)





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
