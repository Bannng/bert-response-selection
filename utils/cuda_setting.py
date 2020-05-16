__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import torch as T


def get_device_setting():
    return T.device('cuda') if T.cuda.is_available() else T.device('cpu')
