import time
import numpy as np
import torch

class TicToc:
    def __init__(self):
        self.start_time = None
    def tic(self):
        self.start_time = time.time()

    def toc(self):
        temp_time = self.start_time
        self.start_time = time.time()
        return (time.time() - temp_time)*1e3



def stamp2seconds(msg):
    # 访问时间戳
    timestamp = msg.header.stamp
    
    # 转换为秒
    return timestamp.secs + timestamp.nsecs * 1e-9

def stamp2seconds_header(msg):
    # 访问时间戳
    timestamp = msg.stamp
    # 转换为秒
    return timestamp.secs + timestamp.nsecs * 1e-9



def inverse_Tmatrix(T):
    assert( T.shape == (4,4))
    R = T[:3 , :3] ; t = T[:3 , 3:]
    T_inv = torch.eye(4).to("cuda")
    T_inv[:3,  :3] = R.T
    T_inv[:3 , 3:] = R.T@(-t)
    return T_inv