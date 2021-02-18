import argparse


class BaseOption():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--prefix', type=str, default='pix2pixhd')
        self.parser.add_argument('--seed', type=int, default=1220)
        self.parser.add_argument('--height', type=int, default=128)
        self.parser.add_argument('--width', type=int, default=128)
        self.parser.add_argument('--ch_inp', type=int, default=1)
        self.parser.add_argument('--ch_tar', type=int, default=1)
        self.parser.add_argument('--type_gan', type=str, default='lsgan',
                                 help='[lsgan, gan]')
        self.parser.add_argument('--type_norm', type=str, default='batch',
                                 help='[none, instance]') 
        self.parser.add_argument('--nb_D', type=int, default=3)
        self.parser.add_argument('--nb_layer', type=int, default=3)
        self.parser.add_argument('--nb_feat_init_D', type=int, default=64)
        self.parser.add_argument('--nb_feat_init_G', type=int, default=64)
        self.parser.add_argument('--nb_down', type=int, default=4)
        self.parser.add_argument('--nb_block', type=int, default=9)
        self.parser.add_argument('--use_tanh', type=bool, default=False)
        self.parser.add_argument('--root_data', type=str, default='/path/to/data')
        self.parser.add_argument('--root_save', type=str, default='/path/to/save')

    def parse(self):
        opt = self.parser.parse_args()
        if opt.type_gan == 'lsgan' :
            opt.use_sigmoid = True
        elif opt.type_gan == 'gan' :
            opt.use_sigmoid = False
        else :
            assert True
        return self.parser.parse_args()


class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=True)
        self.parser.add_argument('--gpu_id', type=int, default=0)
        self.parser.add_argument('--batch_size', type=int, default=64)
        self.parser.add_argument('--workers', type=int, default=4)
        self.parser.add_argument('--initial_learning_rate', type=float, default=0.001)
        self.parser.add_argument('--decay_steps', type=int, default=1000)
        self.parser.add_argument('--decay_rate', type=float, default=0.96)
        self.parser.add_argument('--beta_1', type=float, default=0.5)
        self.parser.add_argument('--beta_2', type=float, default=0.999)
        self.parser.add_argument('--epsilon', type=float, default=1e-8)
        self.parser.add_argument('--weight_FM_loss', type=float, default=10.)


class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=False)
        self.parser.add_argument('--gpu_id', type=int, default=3)
        self.parser.add_argument('--batch_size', type=int, default=8)
        self.parser.add_argument('--epoch', type=int, default=200)
