import os
import sys

from log.logging_wrapper import LoggingWrapper
from src.opts import opts

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    opt = opts().parse()
    logger = LoggingWrapper(__name__)
    logger.add_file_handler(opt.log_path, LoggingWrapper.DEBUG, None)
    logger.add_stream_handler(sys.stdout, LoggingWrapper.DEBUG, None)
    logger.info("Main parameters are initialized.")

    current_path = os.getcwd()
    logger.info("source current path : {}".format(current_path))

    logger.info("Operation parameter is [%s]" % opt.op)
    if opt.op == 'c' or opt.op == 'create':
        from src.create import Create
        create = Create(opt)
        create.create()

    elif opt.op == 't' or opt.op == 'train':
        from src.train import Train
        train = Train(opt)
        train.train()

    elif opt.op == 'T' or opt.op == 'test':
        opt = opts().init()
        if opt.test_mot20:
            seqs_str = '''MOT20-04
                          MOT20-06
                          MOT20-07
                          MOT20-08
                          '''
            data_root = os.path.join(opt.data_dir, 'MOT20/test')
            seqs = [seq.strip() for seq in seqs_str.split()]
        
            from src.track import Track
            track = Track(opt)
            track.track(data_root=data_root, seqs=seqs, exp_name='teddy_all_dla34')
        elif opt.test_mot17:
            seqs_str = '''MOT17-09-SDP
                          MOT17-10-SDP'''
            data_root = os.path.join(opt.data_dir, 'MOT17/test')
            seqs = [seq.strip() for seq in seqs_str.split()]

            from src.track import Track
            track = Track(opt)
            track.track(data_root=data_root, seqs=seqs, exp_name='teddy_all_dla34')
        elif opt.test_video:
            seqs_str = '''test_sample
                          '''
            data_root = os.path.join(opt.data_dir, 'TEST_DANCE_PRACTICE/test')
            seqs = [seq.strip() for seq in seqs_str.split()]

            from src.video_track import Track_video
            track = Track_video(opt)
            track.track(data_root=data_root, seqs=seqs, exp_name='test_sample_dla34')

        else:
            logger.critical("Not defined test dataset. check your option.")
            sys.exit()

    elif opt.op == 's' or opt.op == 'show':
        from src.visualization import Visualize
        show = Visualize(opt)
        show.visualize()

    elif opt.op == 'a' or opt.op == 'api':
        opt = opts().init()
        from src.api import API
        api = API(opt)

    else:
        logger.critical("Not defined operation : %s" % opt.op)
        sys.exit()

    logger.info("Operation [%s] done." % opt.op)
