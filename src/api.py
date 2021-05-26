import os, sys, subprocess, time, datetime
import json
import jpype
import traceback
import requests

from flask import Flask
from flask import request
from flask_cors import CORS

from flask.views import MethodView

import cv2
import torch
import torchvision

from .lib.multitracker import JDETracker
from .lib.tracking_utils import Timer

from . import jde as datasets
from .opts import opts

class API():
    def __init__(self, opt):
        self.opt = opt
        tracker = JDETracker(opt)

        app = Flask(__name__)

        cors = CORS(app, resources={r"/*":{"origins": "*"}})
        
        prefix = os.path.join('/', self.opt.prefix)
        app.add_url_rule(prefix, view_func=DLWrapper.as_view('DLWrapper', opt=self.opt, tracker=tracker))

        app.run(host=self.opt.host,
                port=self.opt.port,
                threaded=True,
                debug=False)

class DLWrapper(MethodView):
    RESPONSE_CODE_SUCCESS = 1
    RESPONSE_CODE_INPUT_ERROR = 2
    RESPONSE_CODE_MODEL_ERROR = 3
    def __init__(self, opt, tracker):
        self.opt = opt
        self.timer = Timer()
        self.tracker = tracker

    def get(self):
        return json.dumps(
            {'status': self.RESPONSE_CODE_INPUT_ERROR, 'error_msg': 'request error. please use post method.'},
            ensure_ascii=False)

    def post(self):
        # jpype.attachThreadToJVM

        input_data = request.files['video']
        if input_data.filename.split('.')[-1] not in ['mp4','avi']:
            return json.dumps(
                {'status': self.RESPONSE_CODE_INPUT_ERROR, 'error_msg': 'Input data format is wrong. check your file.'},
                ensure_ascii=False)
        self.video_path = input_data.filename

        self.timer.clear()
        self.timer.tic()

        # print("API input video name : {}".format(self.video_path))   # for debug
        input_data.save(self.video_path)
        self.timer.toc()

        self.video_save_time = self.timer.diff

        if os.path.exists(self.video_path) is not True:
            return json.dumps(
                {'status': self.RESPONSE_CODE_MODEL_ERROR,
                 'error_msg': 'Problem occurs in API module source. The video has not been saved.'},
                ensure_ascii=False)


        result_dict=self.track()
        os.remove(self.video_path)

        return json.dumps(result_dict, ensure_ascii=False)

    def eval_seq(self):
        opt = self.opt
        dataloader = self.dataloader
        frame_rate = self.frame_rate

        self.tracker._stracks_reset(frame_rate=frame_rate)
        tracker = self.tracker

        results = []
        frame_id = 0
        for path, img, img0 in dataloader:
            buf_dict = {}
            if opt.gpus[0] >= 0:
                blob = torch.from_numpy(img).cuda().unsqueeze(0)    # unsqueeze(0) : index 0 increase dimension
            else:
                blob = torch.from_numpy(img).unsqueeze(0)
            online_targets = tracker.update(blob, img0)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:   # check opt.min_box_area
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

            buf_dict['frame'] = int(frame_id + 1)
            buf_dict['len_bbox'] = int(len(online_ids))
            buf_dict['bbox'] = []
            for tlwh, track_id in zip(online_tlwhs, online_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                bbox_buf = {}
                bbox_buf[str(int(track_id))] = dict(
                        left=x1,
                        top=y1,
                        width=w,
                        height=h
                        )   # [x1, y1, w, h]
                buf_dict['bbox'].append(bbox_buf)
            if len(buf_dict['bbox']) != 0:  # if bbox empty, do not write result json
                results.append(buf_dict)
            frame_id += 1

        return frame_id, results

    def track(self):
        opt = self.opt
        self.data_type = 'mot'

        # run tracking
        self.timer.tic()
        self.dataloader = datasets.LoadVideo(self.video_path, opt.img_size) # check your model trained image data size

        self.timer.toc()
        self.video_load_time = self.timer.diff
        self.frame_rate = self.dataloader.frame_rate
        # print("Frame rate of the video: {}".format(self.frame_rate))  # for debug
        # print("Length of the video: {:d}".format(self.dataloader.vn)) # for debug
        self.timer.tic()
        n_frame, result = self.eval_seq()
        self.timer.toc()

        if n_frame != self.dataloader.vn:
            return {'status': self.RESPONSE_CODE_MODEL_ERROR,
                    'error_msg': 'Problem occurs in API module source. Frame count wrong.'}

        self.elapsed_time = self.timer.total_time
        # print("video save time : {:.2f}, video load time : {:.2f}, total elapsed time :{:.2f}"
        #     .format(self.video_save_time, self.video_load_time, self.elapsed_time))
        # image load & model calculate time take a long time
        return {'status':self.RESPONSE_CODE_SUCCESS, 'elapsed_time': self.elapsed_time,'result': result}



if __name__ == '__main__':
    os.environ['CUDA_VISIBEL_DEVICES'] = '0'
    opt = opts().init()

    api = API(opt=opt)
