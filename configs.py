import numpy as np
import os, sys
import glob
import argparse
from methods import backbone
import logging

model_dict = dict(
    Conv4=backbone.Conv4)


def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--save_dir', default='./record')
    parser.add_argument('--data_dir', default='./filelists/')
    parser.add_argument('--log_path', default='logs/')
    parser.add_argument('--dataset', default='miniImagenet', help='miniImagenet')
    parser.add_argument('--model', default='Conv4', help='model: Conv4')
    parser.add_argument('--method', default='matchingnet',
                        help='matchingnet')
    parser.add_argument('--train_n_way', default=5, type=int,
                        help='class num to classify for training')
    parser.add_argument('--test_n_way', default=5, type=int,
                        help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot', default=5, type=int,
                        help='number of labeled data in each class, same as n_support')
    parser.add_argument('--train_aug', default=False, type=bool,
                        help='perform data augmentation or not during training ')  # still required for save_features.py and test.py to find the model path correctly

    if script == 'train':
        parser.add_argument('--num_classes', default=200, type=int,
                            help='total number of classes in softmax, only used in baseline')  # make it larger than the maximum label value in base class
        parser.add_argument('--save_freq', default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=-1, type=int,
                            help='Stopping epoch')
        parser.add_argument('--resume', default=True, type=bool, help='continue from previous trained model with largest epoch')
    elif script == 'save_features':
        parser.add_argument('--split', default='novel',
                            help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int, help='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split', default='novel',
                            help='base/val/novel')  # default novel, but you can also test base/val class accuracy if you want
        parser.add_argument('--save_iter', default=-1, type=int,
                            help='saved feature from the model trained in x epoch, use the best model if x is -1')
    else:
        raise ValueError('Unknown script')

    return parser.parse_args()


def get_assigned_file(checkpoint_dir, num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file


def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file


def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)


def get_logger(filename, logger_name=None):
    """set logging file and format
    Args:
        filename: str, full path of the logger file to write
        logger_name: str, the logger name, e.g., 'master_logger', 'local_logger'
    Return:
        logger: python logger
    """
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt="%m%d %I:%M:%S %p")
    # different name is needed when creating multiple logger in one process
    logger = logging.getLogger(logger_name)
    fh = logging.FileHandler(os.path.join(filename))
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    return logger

