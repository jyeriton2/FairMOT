import os, sys
import json
import cv2
import numpy as np

from .opts import opts

from log.logging_wrapper import LoggingWrapper

class Visualize():
    def __init__(self, opt):
        self.opt = opt
        self.logger = LoggingWrapper(__name__)
        self.logger.add_file_handler(opt.log_path, LoggingWrapper.DEBUG, None)
        self.logger.add_stream_handler(sys.stdout, LoggingWrapper.INFO, None)
        self.logger.info("Visualize module is initialized.")

        bbox_color_json = open(os.path.join(os.path.dirname(__file__),'./bbox_color/bbox_color.json'),'r')
        self.bbox_color_buf = json.load(bbox_color_json)

        self.sw = opt.show_wait
        self.sr = opt.show_ratio

    def _img_show(self, show_name=None):
        if show_name is not None:
            show_name = show_name
        else:
            show_name = '?'

        img = cv2.resize(self.img, (0, 0), fx=self.sr, fy=self.sr, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(show_name, img)
        cv2.waitKey(self.sw)

    def read_bbox_txt(self, flag=0):
        '''
        flag = 0 : sort by frame , frag = 1 : sort by object id
        based on flag, output first key is (frame or object id)
        the value type in the first key is list
        that list is
        [object id, bbox left, bbox top, bbox width, bbox height, ?, ??, ???] (if flag = 0)
        [frame, bbox left, bbox top, bbox width, bbox height, ?, ??, ???] (if flag = 1)
        '''
        bbox_file_path = self.opt.GT
        if not os.path.isfile(bbox_file_path):
            self.logger.critical("Ground truth file does not exists. check your path {}".format(bbox_file_path))
            sys.exit()

        with open(bbox_file_path, 'r') as f:
            buf = f.read()
            buf = buf.split('\n')
            if buf[-1] == '':
                buf = buf[:-1]

        buf2 = []

        for i in buf:
            sbuf = i.split(',')
            buf2.append(sbuf)

        nframes = 0  # large number of frame
        nids = 0  # # of object id

        for i in buf2:
            if nframes < int(i[0]):
                nframes = int(i[0])
            if nids < int(i[1]):
                nids = int(i[1])

        if flag == 0:
            result = {}
            for i in buf2:
                if int(i[0]) not in result.keys():
                    result[int(i[0])] = [i[1:]]
                else:
                    result[int(i[0])].append(i[1:])
        elif flag == 1:
            result = {}
            for i in buf2:
                if int(i[1]) not in result.keys():
                    result[int(i[1])] = [i[0]] + i[2:]
                else:
                    result[int(i[1])].append([i[0]] + i[2:])

        return result

    def ltwh2ltbr(self, box):  # box : (left, top, width, height)
        result = []
        result.append(box[0])
        result.append(box[1])
        result.append(box[0] + box[2] - 1)
        result.append(box[1] + box[3] - 1)
        return result

    def one_image_with_one_bbox_draw(self, bbox, color_select):  # img: cv2 image, bbox: (left, top, width, height), color select
        ltbr = self.ltwh2ltbr(bbox)
        color = {0: 'black', 1: 'white', 2: 'red1', 3: 'red2', 4: 'yellow1', 5: 'green1', 6: 'green2', 7: 'blue1',
                 8: 'blue2', 9: 'blue3', 10: 'purple', 11: 'gray1', 12: 'blue4', 13: 'orange', 14: 'gray2',
                 15: 'yellow2', 16: 'blue5', 17: 'green3'}
        img = cv2.rectangle(self.img,
                            tuple(ltbr[:2]),
                            tuple(ltbr[2:]),
                            color=self.bbox_color_buf[color[color_select]],
                            thickness=3)
        return img

    def one_image_with_multiple_bbox_show(self):  # img: cv2 image, bbox: (object_id, left, top, width, height, ....)
        bboxes = self.gt[self.img_number]
        for bb in bboxes:
            # bb[0] : object id
            c = int(bb[0]) % 16  # not use black & white
            c += 2
            img = self.one_image_with_one_bbox_draw(list(map(int, map(float, bb[1:5]))), color_select=c)
            # use 'map' , first str to float list element value, second float to int list element value
        return img

    def visualize(self):
        opt = self.opt
        self.logger.info("Ground truth data struct :")
        self.logger.info("frame, obj_id, bb_left, bb_top, bb_width, bb_height, ?, ??, ???")
        gt = self.read_bbox_txt(0)

        if opt.show_task == 'one_shot':
            img_name = opt.show_test_name
            if os.path.isfile(img_name) is not True:
                self.logger.critical("Image file does not exists. check your path {}.".format(img_name))
                sys.exit()
            self.img = cv2.imread(img_name, cv2.IMREAD_COLOR)

            # print(list(map(int, map(float, gt[1][0][1:5]))))

            # self.gt = del_not_use_bbox(gt[1])
            # self.gt = select_class_bbox(gt[2190], 13)
            self.gt = gt
            self.img_number = int(img_name.split('/')[-1].split('.')[0])  # img naming = numbering

            if self.img_number not in self.gt.keys():  # image bbox does not exists
                self.logger.error("This image does not have bbox results")
                sys.exit()

            img_with_bboxes = self.one_image_with_multiple_bbox_show()

            if opt.show_image == True:
                self._img_show(show_name=str(self.img_number))
            if opt.save_image == True:
                img_save_location = opt.save_path
                if os.path.isdir(img_save_location) is not True:
                    os.mkdir(img_save_location)
                save_name = os.path.join(img_save_location, img_name.split('/')[-1])
                cv2.imwrite(save_name, img_with_bboxes)

        elif opt.show_task == 'video':
            video_file_path = opt.show_test_name
            if os.path.isfile(video_file_path) is not True:
                self.logger.critical("Video file does not exists. check your path {}.".format(video_file_path))
                sys.exit()

            cap = cv2.VideoCapture(video_file_path)

            self.img_number = 0
            while True:
                res, img0 = cap.read()
                self.img_number += 1
                if res is not True:
                    break

                # self.gt = del_not_use_bbox(gt[1])
                # self.gt = select_class_bbox(gt[2190], 13)
                self.gt = gt
                self.img = img0

                if self.img_number not in self.gt.keys():  # image bbox does not exists
                    self.logger.error("{} frame does not have bbox results".format(self.img_number))
                    if opt.save_image == True:
                        imgs_save_location = os.path.join(opt.save_path, 'images')
                        if os.path.isdir(imgs_save_location) is not True:
                            os.mkdir(imgs_save_location)
                        save_name = os.path.join(imgs_save_location, '{:06d}.jpg'.format(self.img_number))
                        cv2.imwrite(save_name, img0)
                    continue

                img_with_bboxes = self.one_image_with_multiple_bbox_show()

                if opt.show_image == True:
                    self._img_show(show_name='_test')
                if opt.save_image == True:
                    imgs_save_location = os.path.join(opt.save_path, 'images')
                    if os.path.isdir(imgs_save_location) is not True:
                        os.mkdir(imgs_save_location)
                    save_name = os.path.join(imgs_save_location, '{:06d}.jpg'.format(self.img_number))
                    cv2.imwrite(save_name, img_with_bboxes)

            if opt.save_video == True:
                if opt.save_image is not True:
                    self.logger.critical("Do not save images. if you want to make video, give option save image (--save_image)")
                    sys.exit()
                if os.path.isdir(os.path.join(opt.save_path, "images")) is not True:
                    self.logger.error("Result Images Directory not exists.")
                    sys.exit()
                save_name = os.path.join(opt.save_path, "video.mp4")
                framerate = int(round(cap.get(cv2.CAP_PROP_FPS)))
                cmd_str = 'ffmpeg -framerate {} -i {}/%06d.jpg -codec:v libx264 -profile:v baseline -preset slow -pix_fmt yuv420p -vf "scale=-1:1080, pad=ceil(iw/2)*2:ceil(ih/2)*2" -threads 0 -f mp4 "{}"'.format(framerate, os.path.join(opt.save_path, "images"), save_name)
                os.system(cmd_str)

        elif opt.show_task == 'images':
            imgs_location = opt.show_test_name
            if os.path.isdir(imgs_location) is not True:
                self.logger.critical("Images directory does not exists. check your path {}.".format(imgs_location))
                sys.exit()

            img_lists = os.listdir(imgs_location)
            img_lists = [file for file in img_lists if file.endswith(".jpg") or file.endswith(".png")]
            img_lists = sorted(img_lists)

            # print(img_lists)

            for i in img_lists:
                img_name = os.path.join(imgs_location, i)
                self.img = cv2.imread(img_name, cv2.IMREAD_COLOR)

                # self.gt = del_not_use_bbox(gt[1])
                # self.gt = select_class_bbox(gt[2190], 13)
                self.gt = gt
                self.img_number = int(i.split('.')[0])  # img naming = numbering

                if self.img_number not in self.gt.keys():  # image bbox does not exists
                    self.logger.error("{} frame does not have bbox results".format(self.img_number))
                    if opt.save_image == True:
                        imgs_save_location = os.path.join(opt.save_path, 'images')
                        if os.path.isdir(imgs_save_location) is not True:
                            os.mkdir(imgs_save_location)
                        save_name = os.path.join(imgs_save_location, i)
                        cv2.imwrite(save_name, self.img)
                    continue

                img_with_bboxes = self.one_image_with_multiple_bbox_show()

                if opt.show_image == True:
                    self._img_show(show_name='_test')
                if opt.save_image == True:
                    imgs_save_location = os.path.join(opt.save_path, 'images')
                    if os.path.isdir(imgs_save_location) is not True:
                        os.mkdir(imgs_save_location)
                    save_name = os.path.join(imgs_save_location, i)
                    cv2.imwrite(save_name, img_with_bboxes)

            if opt.save_video == True:
                if opt.save_image is not True:
                    self.logger.critical("Do not save images. if you want to make video, give option save image (--save_image)")
                    sys.exit()
                if os.path.isdir(os.path.join(opt.save_path, "images")) is not True:
                    self.logger.error("Result Images Directory not exists.")
                    sys.exit()
                save_name = os.path.join(opt.save_path, "video.mp4")
                framerate = 25
                cmd_str = 'ffmpeg -framerate {} -i {}/%06d.jpg -codec:v libx264 -profile:v baseline -preset slow -pix_fmt yuv420p -vf "scale=-1:1080, pad=ceil(iw/2)*2:ceil(ih/2)*2" -threads 0 -f mp4 "{}"'.format(framerate, os.path.join(opt.save_path, "images"), save_name)
                os.system(cmd_str)

        else:
            self.logger.critical("Wrong show task. Select [one_shot, video, images]")
            sys.exit()

        self.logger.info("Result Visualize finish!")


def del_not_use_bbox(bboxes):   # bbox: (object_id, left, top, width, height, ....) - mot dataset (frame, id, bb_left, bb_top, bb_width, bb_height, ?, ??, ???) index 6 {0,1}
    result = []
    for bb in bboxes:
        if int(bb[5]) == 1:
            result.append(bb)
    return result

def select_class_bbox(bboxes, nc=1):  # bbox: (object_id, left, top, width, height, ....) - mot dataset (frame, id, bb_left, bb_top, bb_width, bb_height, ?, ??, ???) index 7 {1,11,7}
    # nc: number of class
    # 1: person, 6: cart, 7: person obscured by an object, 11: ? background?, 13: noise
    result = []
    for bb in bboxes:
        if int(bb[6]) == nc:
            result.append(bb)

    return result
