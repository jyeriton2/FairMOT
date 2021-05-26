import os, sys
import json
import cv2
import numpy as np

def get_video_info(cap):
    fr = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    return fr, n, w, h

def check_video_file(cap, n_video):
    c = 0
    flag = 0
    while True:
        res, img0 = cap.read()
        c += 1
        if res is not True:
            check = c - 1
            if check == int(n_video):
                break
            else:
                flag = 1
                break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if flag == 1:
        return c
    else:
        return True

def write_seqinfo(save_path, video_name, frame_rate, n_video, video_width, video_height):
    with open(os.path.join(save_path,'seqinfo.ini'),'w') as f:
        f.write('[Sequence]\n')
        f.write('name={}\n'.format(video_name))
        f.write('imDir=img1\n')
        f.write('frameRate={}\n'.format(int(round(frame_rate))))
        f.write('seqLength={}\n'.format(int(n_video)))
        f.write('imWidth={}\n'.format(int(video_width)))
        f.write('imHeight={}\n'.format(int(video_height)))
        f.write('imExt=.jpg\n')

def video2images(save_path, cap):
    if os.path.isdir(save_path) is not True:
        os.mkdir(save_path)

    c = 0
    while True:
        res, img0 = cap.read()
        c += 1
        if res is not True:
            break
        img_save_name = os.path.join(save_path, '{:06d}.jpg'.format(c))
        cv2.imwrite(img_save_name, img0)

def get_images_seqinfo_file(video_name, save_path):
    cap = cv2.VideoCapture(video_name)
    frame_rate, n_video, video_width, video_height = get_video_info(cap)
    c = check_video_file(cap, n_video)
    if c is not True:
        return c
    
    frame_rate = int(round(frame_rate))
    write_seqinfo(save_path, video_name.split('/')[-1].split('.')[0], frame_rate=frame_rate, n_video=n_video, video_width=video_width, video_height=video_height)
    video2images(os.path.join(save_path, 'img1'), cap)
    cap.release()
    return True


if __name__ == '__main__':
    video_location = '/home/teddy/dataset_test'
    save_path = '/home/teddy/dataset_test'
    video_name = os.path.join(video_location, 'test_dynamite.mp4')
    cap = cv2.VideoCapture(video_name)
    frame_rate, n_video, video_width, video_height = get_video_info(cap)
    c = check_video_file(cap, n_video)
    if c is not True:
        print(c)
        print("opencv video read error")
        sys.exit()

    print("frame rate : {},\t total frames : {},\t video width : {},\t video height : {}".format(frame_rate, n_video, video_width, video_height))
    print(type(frame_rate), type(n_video), type(video_width), type(video_height))

    frame_rate = int(round(frame_rate))

    # write_seqinfo(save_path, video_name.split('/')[-1].split('.')[0], frame_rate=frame_rate, n_video=n_video, video_width=video_width, video_height=video_height)
    # video2images(save_path, cap)

    cap.release()
