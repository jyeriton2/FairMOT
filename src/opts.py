import argparse
import os, sys

from log.logging_wrapper import LoggingWrapper


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('task', default='mot', help='mot')
        self.parser.add_argument('--dataset', default='jde', help='jde')
        self.parser.add_argument('--exp_id', default='default')
        # select operation [train, test, api, download&create, show&save]
        self.parser.add_argument('--op', default='default',
                                 help='Which operation do you want? '
                                      'c(create), t(train), T(test), s(show), a(api)')
        self.parser.add_argument('--test', action='store_true')
        # self.parser.add_argument('--load_model', default='../models/ctdet_coco_dla_2x.pth',
        # help='path to pretrained model')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU')
        self.parser.add_argument('--num_workers', type=int, default=8,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        # log
        self.parser.add_argument('--log_path', default='log/log',
                                 help='log file save path')
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.') # check; (how to control save epoch) val_intervals 
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.5,
                                 help='visualization threshold.')

        # download & create
        self.parser.add_argument('--video2images', default=False,
                                 help='video file name (full path) for change video to images')
        self.parser.add_argument('--GT', default=False,
                                 help='ground truth file name (full path) for directory you want to use')
        self.parser.add_argument('--create_images', default=False, help='images directory (full path)')

        # mot
        # dataset -> train&test
        self.parser.add_argument('--data_cfg', type=str, default='./cfg/mot20.json',
                                 help='load data from cfg')  # Detailed Address in data_dir
        self.parser.add_argument('--data_dir', type=str, default='./data')

        # model
        self.parser.add_argument('--arch', default='dla_34',    # vit_912512
                                 help='model architecture. Currently tested'
                                      'resdcn_34 | resdcn_50 | resfpndcn_34 |'
                                      'dla_34 | hrnet_32')
        self.parser.add_argument('--head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      '0 for no conv layer'
                                      '-1 for default setting: '
                                      '256 for resnets and 256 for dla.')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                      'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr', type=float, default=1e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='20,27',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=30,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=6,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1, help='batch size on the master gpu.')  # if use multi-gpu , what is main gpu (# of gpu = 2, batch = 3 -> '0' gpu : batch 2, '1' gpu : batch 1
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                 help='include validation in training and '
                                      'test on test set')

        # test
        self.parser.add_argument('--K', type=int, default=128, help='max number of output objects.') # if MOT20, 256 (# of objects > 128)
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                 help='not use parallel data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                 help='fix testing resolution or keep '
                                      'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                 help='keep the original resolution'
                                      ' during validation.')
        # tracking
        self.parser.add_argument('--test_mot16', default=False, help='test mot16')
        self.parser.add_argument('--val_mot15', default=False, help='val mot15')
        self.parser.add_argument('--test_mot15', default=False, help='test mot15')
        self.parser.add_argument('--val_mot16', default=False, help='val mot16 or mot15')
        self.parser.add_argument('--test_mot17', default=False, help='test mot17')
        self.parser.add_argument('--val_mot17', default=False, help='val mot17')
        self.parser.add_argument('--val_mot20', default=False, help='val mot20')
        self.parser.add_argument('--test_mot20', default=False, help='test mot20')
        self.parser.add_argument('--test_video', default=False, help='video tracking test')
        self.parser.add_argument('--conf_thres', type=float, default=0.6, help='confidence thresh for tracking')    # MOT20 : 0.4
        self.parser.add_argument('--det_thres', type=float, default=0.3, help='confidence thresh for detection')
        self.parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresh for nms')
        self.parser.add_argument('--track_buffer', type=int, default=30, help='tracking buffer')
        self.parser.add_argument('--min_box_area', type=float, default=200, help='filter out tiny boxes')
        self.parser.add_argument('--input_video', type=str, default='video.mp4', help='path to the input video')

        # show&save
        self.parser.add_argument('--show_task', default='one_show',
                                 help='select show task what you want '
                                      '1. one shot image test (one_shot) '
                                      '2. video test (video) '
                                      '3. images from video (images)')
        self.parser.add_argument('--show_image', action='store_true',
                                 help='image show on(--show_image)/off(Do not give option)')  # if 'show_image' has value, True
        self.parser.add_argument('--save_image', action='store_true',
                                 help='test image result save on(--save_image)/off(Do not give option)')
        self.parser.add_argument('--save_video', action='store_true',
                                 help='test result save format video on(--save_video)/off(Do not give option)')
        self.parser.add_argument('--show_test_name', default=False,
                                 help='give video name or image name or images directory name what you want visualize tracking result')
        self.parser.add_argument('--save_path', default='./',
                                 help='tracking result visualize image or video save path')
        self.parser.add_argument('--show_wait', default=0, help='opencv image show waitkey')
        self.parser.add_argument('--show_ratio', default=0.5, type=float, help='opencv image show window size ratio')

        # api
        self.parser.add_argument('--host', type = str, default='0.0.0.0',
                                 help='host address for use API module (return model result)')
        self.parser.add_argument('--port', type = str, default='7000',
                                 help='address port for use API module (return model result)')
        self.parser.add_argument('--prefix', type = str, default='/',
                                 help='input data save rout for use API module (return model result)')


        # loss
        # Do not Touch
        self.parser.add_argument('--mse_loss', action='store_true',
                                 help='use mse loss or focal loss to train '
                                      'keypoint heatmaps.')

        self.parser.add_argument('--reg_loss', default='l1',
                                 help='regression loss: sl1 | l1 | l2')
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for keypoint heatmaps.')
        self.parser.add_argument('--off_weight', type=float, default=1,
                                 help='loss weight for keypoint local offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='loss weight for bounding box size.')
        self.parser.add_argument('--id_loss', default='ce',
                                 help='reid loss: ce | triplet')
        self.parser.add_argument('--id_weight', type=float, default=1,
                                 help='loss weight for id')
        self.parser.add_argument('--reid_dim', type=int, default=512,
                                 help='feature dim for reid')

        self.parser.add_argument('--norm_wh', action='store_true',
                                 help='L1(\hat(y) / y, 1) or L1(\hat(y), y)')
        self.parser.add_argument('--dense_wh', action='store_true',
                                 help='apply weighted regression near center or '
                                      'just apply regression on center point.')
        self.parser.add_argument('--cat_spec_wh', action='store_true',
                                 help='category specific bounding box size.')
        self.parser.add_argument('--not_reg_offset', action='store_true',
                                 help='not regress local offset.')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        self.logger = LoggingWrapper(__name__)
        self.logger.add_file_handler(opt.log_path, LoggingWrapper.DEBUG, None)
        self.logger.add_stream_handler(sys.stdout, LoggingWrapper.INFO, None)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        opt.fix_res = not opt.keep_res
        print('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')
        opt.reg_offset = not opt.not_reg_offset

        if opt.head_conv == -1:  # init default head_conv
            opt.head_conv = 256 if 'dla' in opt.arch else 256
        opt.pad = 31
        opt.num_stacks = 1

        if opt.trainval:
            opt.val_intervals = 100000000

        ''' for use multi-gpu
        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)
        '''


        opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        self.logger.info("The output will be saved to {}".format(opt.save_dir))

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def update_dataset_info_and_set_heads(self, opt, dataset):
        self.logger = LoggingWrapper(__name__)
        self.logger.add_file_handler(opt.log_path, LoggingWrapper.DEBUG, None)
        self.logger.add_stream_handler(sys.stdout, LoggingWrapper.INFO, None)

        input_h, input_w = dataset.default_resolution
        opt.mean, opt.std = dataset.mean, dataset.std
        opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = opt.input_res if opt.input_res > 0 else input_h
        input_w = opt.input_res if opt.input_res > 0 else input_w
        opt.input_h = opt.input_h if opt.input_h > 0 else input_h
        opt.input_w = opt.input_w if opt.input_w > 0 else input_w
        opt.output_h = opt.input_h // opt.down_ratio
        opt.output_w = opt.input_w // opt.down_ratio
        opt.input_res = max(opt.input_h, opt.input_w)
        opt.output_res = max(opt.output_h, opt.output_w)

        if opt.task == 'mot':
            opt.heads = {'hm': opt.num_classes,
                         'wh': 2 if not opt.cat_spec_wh else 2 * opt.num_classes,
                         'id': opt.reid_dim}
            if opt.reg_offset:
                opt.heads.update({'reg': 2})
            opt.nID = dataset.nID
            opt.img_size = (1088, 608)
            # opt.img_size = (512, 512)
        else:
            assert 0, 'task not defined!'
        self.logger.info('heads: {}'.format(opt.heads))
        return opt

    def init(self, args=''):
        default_dataset_info = {
            'mot': {'default_resolution': [608, 1088],# [512, 512],# [608, 1088], 
                    'num_classes': 1,
                    'mean': [0.408, 0.447, 0.470], 
                    'std': [0.289, 0.274, 0.278],
                    'dataset': 'jde', 'nID': 14455},
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.update_dataset_info_and_set_heads(opt, dataset)
        return opt
