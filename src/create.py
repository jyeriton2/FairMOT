import os, sys
import json
import shutil
import numpy as np

from .opts import opts
from .lib import video_utils

from log.logging_wrapper import LoggingWrapper

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)

class Create:
    def __init__(self, opt):
        '''
        create
        ground file must be written in this structure.
        frame, obj_id, bb_left, bb_top, bb_width, bb_height, mark, label, ???
        '''

        self.opt = opt

        self.data_root = os.path.join(opt.root_dir, opt.data_dir)
        mkdir_if_missing(self.data_root)
        self.data_cfg_root = os.path.join(opt.root_dir, 'cfg')
        mkdir_if_missing(self.data_cfg_root)
        self.data_cfg_name = 'created.json'
        self.img_path_file_name = './data/created.train'
        self.data_cfg = os.path.join(self.data_cfg_root, 'created.json')
        self.img_path_file_full_name = os.path.join(self.data_root, 'created.train')  # train image path written
        self.save_data_root = os.path.join(self.data_root, 'created/train')
        mkdir_if_missing(self.save_data_root)

        self.logger = LoggingWrapper(__name__)
        self.logger.add_file_handler(opt.log_path, LoggingWrapper.DEBUG, None)
        self.logger.add_stream_handler(sys.stdout, LoggingWrapper.INFO, None)
        self.logger.info("Create module is initialized.")

    def write_config_json(self):
        data_cfg_root = self.data_cfg_root
        data_root = self.data_root
        data_cfg_name = self.data_cfg_name
        img_path_file_name = self.img_path_file_name

        cfg_file = os.path.join(data_cfg_root, data_cfg_name)
        if os.path.isfile(cfg_file):
            with open(cfg_file, 'r', encoding='utf-8') as f:
                cfg_data = json.loads(f.read())
            if cfg_data['root'] != data_root:
                self.logger.critical("config json file is wrong. check your config json file at root")
                sys.exit()
            if cfg_data['train']['created'] != img_path_file_name:
                self.logger.critical("config json file is wrong. check your config json file at train")
                sys.exit()
        else:
            cfg_data = {
                'root': data_root,
                'train': {'created': img_path_file_name},
                'test_emb': {'created': img_path_file_name},
                'test': {'created': img_path_file_name}
            }
            with open(cfg_file, 'w', encoding='utf-8') as f:
                json.dump(cfg_data, f, ensure_ascii=False, indent='\t')

    def _gen_labels(self, gt_file_name, seqinfo_file_name, data_name):
        label_root = os.path.join(self.data_root, 'created')
        label_root = os.path.join(label_root, 'labels_with_ids/train')
        mkdir_if_missing(label_root)

        tid_curr = 0
        tid_last = -1

        seq_info = open(seqinfo_file_name).read()
        seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

        gt = np.loadtxt(gt_file_name, dtype=np.float64, delimiter=',')

        seq_label_root = os.path.join(label_root, data_name, 'img1')
        mkdir_if_missing(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _ in gt:
            if mark == 0 or not label == 1:
                continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = os.path.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)

    def image_files_listup(self, save_data_root, data_name):
        image_path_file_name = self.img_path_file_full_name
        if os.path.isfile(image_path_file_name):
            with open(image_path_file_name, 'r') as f:
                img_files = f.readlines()
                img_files = [x.strip() for x in img_files]
                img_files = list(filter(lambda x: len(x) > 0, img_files))
        else:
            img_files = []

        add_img_files = sorted(os.listdir(os.path.join(save_data_root, 'img1')))
        for i in add_img_files:
            img_files.append(os.path.join('created/train', data_name, 'img1', i))
        with open(image_path_file_name, 'w+') as f:
            for i in img_files:
                img_path = i + '\n'
                f.write(img_path)

    def create(self):
        opt = self.opt
        if opt.video2images != False and opt.create_images == False:
            data_name = opt.video2images
            if os.path.exists(data_name) is not True:
                self.logger.critical("Do not exists data. check your data path.")
                sys.exit()
            data_name = data_name.split('/')[-1].split('.')[0]
        elif opt.video2images == False and opt.create_images != False:
            data_name = opt.create_images
            if os.path.exists(data_name) is not True:
                self.logger.critical("Do not exists data. check your data path.")
                sys.exit()
            if data_name[-1] == '/':
                data_name = data_name.split('/')[-2]
            else:
                data_name = data_name.split('/')[-1]
        else:
            self.logger.critical("give one data create option. --video2images or --create_images.")
            sys.exit()
        if opt.GT == False:
            self.logger.critical("give ground truth file. --GT.")
            sys.exit()
        if os.path.exists(opt.GT) is not True:
            self.logger.critical("Do not exists ground truth file. check your ground truth file path.")
            sys.exit()

        self.write_config_json()
        data_cfg = self.data_cfg

        save_data_root = self.save_data_root

        save_data_root = os.path.join(save_data_root, data_name)
        mkdir_if_missing(save_data_root)

        if opt.video2images is not False:
            c = video_utils.get_images_seqinfo_file(opt.video2images, save_data_root)
            if c is not True:
                self.logger.error("opencv video read error")
                self.logger.error("error frame : {}".format(c))
                sys.exit()
        if opt.create_images is not False:
            shutil.copytree(os.path.join(opt.create_images, 'img1'),
                            os.path.join(save_data_root, 'img1'))  # all images copy
            if os.path.isfile(os.path.join(opt.create_images, 'seqinfo.ini')) is not True:
                self.logger.critical("seqinfo file do not exists. check your seqinfo file path.")
                sys.exit()
            shutil.copy2(os.path.join(opt.create_images, 'seqinfo.ini'), os.path.join(save_data_root, 'seqinfo.ini'))

        seqinfo_file_name = os.path.join(save_data_root, 'seqinfo.ini')
        '''
        # ground truth file backup
        save_gt_root = os.path.join(save_data_root,'gt')
        mkdir_if_missing(save_gt_root)
        gt_file_name = opt.GT.split('/')[-1]
        save_gt_file_name = os.path.join(save_gt_root, gt_file_name)
        shutil.copy2(opt.GT, save_gt_file_name) # ground file copy
        '''
        # labels_with_ids per frame image
        self._gen_labels(opt.GT, seqinfo_file_name, data_name)

        self.image_files_listup(save_data_root, data_name)

        self.logger.info("seqinfo.ini file save path : {}".format(seqinfo_file_name))
        self.logger.info("label files save path : {}".format(os.path.join(self.data_root, 'created/labels_with_ids')))
        self.logger.info("config file save path : {}".format(data_cfg))
        self.logger.info("data images path list : {}".format(self.img_path_file_name))
        self.logger.info("Data create finish!")
