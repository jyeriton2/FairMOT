import os, sys

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
import torch_optimizer as optim
from .opts import opts
from .lib.models.model import create_model, load_model, save_model
#from models.data_parallel import DataParallel
from .jde import JointDataset
from .mot import MotTrainer

from log.logging_wrapper import LoggingWrapper

train_factory = {
        'mot':MotTrainer,
        }

def get_dataset(dataset, task):
    if task == 'mot':
        return JointDataset
    else:
        return None

class Train():
    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggingWrapper(__name__)
        self.logger.add_file_handler(opt.log_path, LoggingWrapper.DEBUG, None)
        self.logger.add_stream_handler(sys.stdout, LoggingWrapper.INFO, None)
        self.logger.info("Train module is initialized.")

        self.dataset = self._setting_data()

        self.logger.info("Model Generate...")
        self.model = create_model(self.opt.arch, self.opt.heads, self.opt.head_conv)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), self.opt.lr)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), opt.lr)
        self.optimizer = optim.RAdam(self.model.parameters(), opt.lr)
        # self.optimizer = optim.Yogi(self.model.parameters(), opt.lr)    # torch_optimizer library

    def _setting_data(self):
        opt = self.opt
        self.logger.info("Setting up data...")
        Dataset = get_dataset(opt.dataset, opt.task)
        if os.path.isfile(opt.data_cfg) is not True:
            self.logger.critical("Config file does not exists. check your config file path {}".format(opt.data_cfg))
            sys.exit()
        f = open(opt.data_cfg)
        data_config = json.load(f)
        trainset_paths = data_config['train']
        dataset_root = data_config['root']
        f.close()
        transforms = T.Compose([T.ToTensor()])
        try:
            dataset = Dataset(opt, dataset_root, trainset_paths, (1088, 608), augment=True, transforms=transforms)  # (1088, 608) , (512, 512)
        except:
            self.logger.critical("File (written train data paths) does not exists.")
            sys.exit()
        opt = opts().update_dataset_info_and_set_heads(opt, dataset)

        self.logger.info("total opt")
        self.logger.info(opt)
        self.logger.info("dataset summary")
        self.logger.info(dataset.tid_num)
        self.logger.info("total # indentities : {}".format(dataset.nID))
        self.logger.info("start index : {}".format(dataset.tid_start_index))

        return dataset

    def train(self):
        opt = self.opt
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test

        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
        opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

        self.logger.info("GPU VISIBLE DEVICES : {}".format(os.environ['CUDA_VISIBLE_DEVICES']))
        self.logger.info("Do not multi-gpu")

        start_epoch = 0
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,    #how many subprocesses to use for data loading
            pin_memory=True,
            drop_last=True
            )

        self.logger.info("Start training...")
        Trainer = train_factory[opt.task]
        trainer = Trainer(opt, self.model, self.optimizer)
        trainer.set_device(opt.gpus, opt.device)
        if opt.load_model != '':
            self.logger.info("loaded : {}".format(opt.load_model))
            self.model, self.optimizer, start_epoch = load_model(self.model,
                                                       opt.load_model,
                                                       trainer.optimizer,
                                                       opt.resume,
                                                       opt.lr,
                                                       opt.lr_step)
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        best = 1e10
        for epoch in range(start_epoch + 1, opt.num_epochs + 1):
            mark = epoch if opt.save_all else 'last'
            log_dict_train, _ = trainer.train(epoch, train_loader)
            self.logger.info("epoch: {} |".format(epoch))
            for k, v in log_dict_train.items():
                self.logger.info('train_{}, {:8f}'.format(k, v))
            if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, self.model, self.optimizer)
            else:
                save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, self.model, self.optimizer)
            if epoch in opt.lr_step:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, self.model, self.optimizer)
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                self.logger.info('Drop LR to {:8f}'.format(lr))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            if epoch % 5 == 0:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, self.model, self.optimizer)

        self.logger.info("Data train finish!")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'    # check gpu visible 
    opt = opts().parse()
    train = Train(opt)
    train.train()
