import os, sys

import cv2
import numpy as np
import torch
import torchvision

from .lib.multitracker import JDETracker
from .lib.tracking_utils import Timer

from . import jde as datasets
from .opts import opts

from log.logging_wrapper import LoggingWrapper

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)

class Track_video():
    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggingWrapper(__name__)
        self.logger.add_file_handler(opt.log_path, LoggingWrapper.DEBUG, None)
        self.logger.add_stream_handler(sys.stdout, LoggingWrapper.INFO, None)
        self.logger.info("Track module (video) is initialized.")

    def write_results(self, results):
        filename = self.result_filename
        data_type = self.data_type
        if data_type == 'mot':
            save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        elif data_type == 'kitti':
            save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
        else:
            raise ValueError(data_type)

        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids in results:
                if data_type == 'kitti':
                    frame_id -= 1
                for tlwh, track_id in zip(tlwhs, track_ids):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    f.write(line)
        self.logger.info("Save results to {}".format(filename))

    def eval_seq(self):
        opt = self.opt
        dataloader = self.dataloader
        frame_rate = self.frame_rate

        tracker = JDETracker(opt, frame_rate=frame_rate)
        timer = Timer()
        results = []
        frame_id = 0
        for path, img, img0 in dataloader:
            if frame_id % 20 == 0:
                print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

            # run tracking
            timer.tic()
            if opt.gpus[0] >= 0:
                blob = torch.from_numpy(img).cuda().unsqueeze(0)  # unsqueeze(0) : index 0 increase dimension
            else:
                blob = torch.from_numpy(img).unsqueeze(0)
            online_targets = tracker.update(blob, img0)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
            timer.toc()
            # save results
            results.append((frame_id + 1, online_tlwhs, online_ids))

            frame_id += 1
            # save results
        self.write_results(results)
        return frame_id, timer.average_time, timer.calls

    def track(self, data_root='./data/MOT20/train', det_root=None, seqs=('MOT20-05',), exp_name='demo'):
        opt = self.opt
        result_root = os.path.join(data_root, '..', 'results', exp_name)
        mkdir_if_missing(result_root)
        self.data_type = 'mot'

        # run tracking
        n_frame = 0
        timer_avgs, timer_calls = [], []
        for seq in seqs:
            self.logger.info("Start seq: {}".format(seq))
            if os.path.exists(os.path.join(data_root, seq, opt.input_video)) is not True:
                self.logger.critical("Test data do not exists. check your path {}.".format(os.path.join(data_root, seq, opt.input_video)))
                continue
            self.dataloader = datasets.LoadVideo(os.path.join(data_root, seq, opt.input_video), opt.img_size) # check LoadVideo class
            self.frame_rate = self.dataloader.frame_rate
            self.logger.info("Frame rate of the video: {}".format(self.frame_rate))
            self.logger.info("Length of the video: {:d} frames".format(self.dataloader.vn))
            self.result_filename = os.path.join(result_root, '{}.txt'.format(seq))
            self.logger.info("The result will be saved to {}".format(self.result_filename))

            nf, ta, tc = self.eval_seq()
            n_frame += nf
            timer_avgs.append(ta)
            timer_calls.append(tc)

        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        self.logger.info("Time elapsed: {:.2f} seconds, FPS: {:.2f}".format(all_time, 1.0 / avg_time))
        self.logger.info("Test video tracking finish!")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # check gpu visible 
    opt = opts().init()

    if opt.test_video:
        seqs_str = '''test_dynamite
                    '''
        data_root = os.path.join(opt.data_dir,'TEST_DANCE_PRACTICE/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    track = Track_video(opt)
    track.track(data_root=data_root,
                seqs=seqs,
                exp_name='dance_all_dla34')
