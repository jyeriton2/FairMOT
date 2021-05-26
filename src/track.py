import os, sys

import cv2
import numpy as np
import motmetrics as mm
import torch
import torchvision

from .lib.multitracker import JDETracker
from .lib.tracking_utils import Timer
from .lib.tracking_utils import Evaluator

from . import jde as datasets
from .opts import opts

from log.logging_wrapper import LoggingWrapper

def mkdir_if_missing(d):
    if not os.path.exists(d):
        os.makedirs(d)

class Track():
    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggingWrapper(__name__)
        self.logger.add_file_handler(opt.log_path, LoggingWrapper.DEBUG, None)
        self.logger.add_stream_handler(sys.stdout, LoggingWrapper.INFO, None)
        self.logger.info("Track module is initialized.")

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
                vertical = tlwh[2] / tlwh[3] > 1.6  # 1.12
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
        accs = []
        n_frame = 0
        timer_avgs, timer_calls = [], []
        for seq in seqs:
            self.logger.info("Start seq: {}".format(seq))
            if os.path.isdir(os.path.join(data_root, seq, 'img1')) is not True:
                self.logger.critical("Test data does not exists. check your path {}.".format(os.path.join(data_root, seq, 'img1')))
                continue
            self.dataloader = datasets.LoadImages(os.path.join(data_root, seq, 'img1'), opt.img_size)
            self.result_filename = os.path.join(result_root, '{}.txt'.format(seq))
            self.logger.info("The result will be saved to {}".format(self.result_filename))
            meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
            # seqinfo file MOT dataset check - (file name, image directory, frame rate, sequence length, image width, image height, image extansion)
            self.frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
            nf, ta, tc = self.eval_seq()
            n_frame += nf
            timer_avgs.append(ta)
            timer_calls.append(tc)

            self.logger.info('Evaluate seq: {}'.format(seq))
            evaluator = Evaluator(data_root, seq, self.data_type)
            accs.append(evaluator.eval_file(self.result_filename))

        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        self.logger.info("Time elapsed: {:.2f} seconds, FPS: {:.2f}".format(all_time, 1.0 / avg_time))
        # get summary
        metrics = mm.metrics.motchallenge_metrics
        mh = mm.metrics.create()
        summary = Evaluator.get_summary(accs, seqs, metrics)
        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)
        Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))
        self.logger.info("Test images tracking finish!")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # check gpu visible 
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    track = Track(opt)
    track.track(data_root=data_root,
                seqs=seqs,
                exp_name='MOT20_all_dla34')
