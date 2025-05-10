from train_network import load_data
import os
import torch
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from os.path import join, abspath
from utils import fs

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy_original'  # debug
setup_path = join(data_root, 'setups', setup_name)

cam_scene, cam_train, cam_valid, prj_train, prj_valid, im_mask,\
mask_corners, setup_info = load_data(data_root, setup_name, input_size=None, compensation=False)

torch.set_printoptions(profile='full',linewidth=100000)
print(im_mask)
fs(im_mask, title='direct_light_mask')