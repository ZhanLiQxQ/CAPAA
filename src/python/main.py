"""
Set up ProCams and capture data
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from os.path import join, abspath
import cv2 as cv
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import classifier
import utils as ut
import matplotlib.pyplot as plt
from classifier import query_multi_classifiers
from img_proc import checkerboard, create_gray_pattern, center_crop as cc
from projector_based_attack_all import run_projector_based_attack, summarize_single_universal_attacker, project_capture_real_attack, \
    summarize_all_attackers, get_attacker_cfg,get_attacker_all_cfg
from projector_based_attack_cam import summarize_single_attacker,summarize_single_attacker_for_UAP_task
# from projector_based_attack_all import run_projector_based_attack_universal


# %% (1) [local] Setup configs, e.g., projector and camera sizes, and sync delay params, etc.
setup_info = DictConfig(dict(
    prj_screen_sz      = (800 , 600),   # projector screen resolution (i.e., set in the OS)
    prj_im_sz          = (256 , 256),   # projector input image resolution, the image will be scaled to match the prj_screen_sz by plt
    prj_offset         = (2560,   0),   # an offset to move the projector plt figure to the correct screen (check your OS display setting)
    cam_raw_sz         = (1280, 720),   # the size of camera's direct output frame
    cam_crop_sz        = (960 , 720),   # a size used to center crop camera output frame, cam_crop_sz <= cam_raw_sz
    cam_im_sz          = (320 , 240),   # a size used to resize center cropped camera output frame, cam_im_sz <= cam_crop_sz, and should keep aspect ratio
    classifier_crop_sz = (240 , 240),   # a size used to center crop resize cam image for square classifier input, classifier_crop_sz <= cam_im_sz
    prj_brightness     = 0.5,           # brightness (0~1 float) of the projector gray image for scene illumination.

    # adjust the two params below according to your ProCams latency, until the projected and captured numbers images are correctly matched
    delay_frames = 10,  # how many frames to drop before we capture the correct one, increase it when ProCams are not in sync
    delay_time = 0.02,  # a delay time (s) between the project and capture operations for software sync, increase it when ProCams are not in sync
    )
)

# Check projector and camera FOV (the camera should see the full projector FOV);
# focus (fixed and sharp, autofocus and image stabilization off); aspect ratio;
# exposure (lowest ISO, larger F-number and proper shutter speed); white balance (fixed); flickering (anti on) etc.
# Make sure when projector brightness=0.5, the cam-captured image is correctly exposed and correctly classified by the classifiers.

# create projector window with different brightnesses, and check whether the exposures are correct
prj_fig = list()
for brightness in [0, setup_info['prj_brightness'], 1.0]:
    prj_fig.append(ut.init_prj_window(*setup_info['prj_screen_sz'], brightness, setup_info['prj_offset']))

# live preview the camera frames, make sure the square preview looks correct: the camera is in focus; the object is centered;
# when brightness=0.5 the illumination looks reasonable;

print('Previewing the camera, make sure everything looks good and press "q" to exit...')
cam = ut.init_cam(setup_info['cam_raw_sz'])
ut.preview_cam(setup_info['cam_raw_sz'], (min(setup_info['cam_raw_sz']), min(setup_info['cam_raw_sz'])),cam)
# plt.close('all')

# %% (2) [local] Check whether the projector and the camera are in sync by capturing the numbers data
data_root      = abspath(join(os.getcwd(), '../../data'))
setup_name     = 'sync_test'
setup_path     = join(data_root  , 'setups/'    , setup_name)

# project and capture, then save the images to cam_cap_path
phase          = 'numbers'
prj_input_path = join(data_root , 'prj_share', phase)
cam_cap_path   = join(setup_path, 'cam/raw'  , phase)
# ut.project_capture_data(prj_input_path, cam_cap_path, setup_info,cam)

# %% (3) [local] Check if the object is correctly classified by *all* three classifiers, if not, go back to step (1) and adjust your ProCams and scene
print('Recognizing the camera-captured scene using the three classifiers...')
ut.print_sys_info()

# set which GPUs to use
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ut.set_torch_reproducibility(False)
ut.reset_rng_seeds(0)

# load ImageNet label
imagenet_labels = classifier.load_imagenet_labels(join(data_root, 'imagenet1000_clsidx_to_labels.txt'))

# project a gray image and capture the scene
prj = ut.init_prj_window(*setup_info['prj_screen_sz'], setup_info['prj_brightness'], setup_info['prj_offset'])
# cam = ut.init_cam(setup_info['cam_raw_sz'])
# %% (3.2)
# drop frames (avoid the wrong frame) and capture
for j in range(0, 100):
    _, im_cam = cam.read()
im_cam = cc(torch.Tensor(cv.cvtColor(cv.resize(cc(im_cam, setup_info['cam_crop_sz']), setup_info['cam_im_sz'], interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)).permute(2, 0, 1) / 255, setup_info['classifier_crop_sz'])
ut.fs(im_cam, title='camera-captured scene1')

# check results of each classifier
classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
ut.fs(im_cam, title='camera-captured scene2')
query_multi_classifiers(im_cam, setup_info['classifier_crop_sz'], classifier_names, imagenet_labels, device, device_ids)
ut.fs(im_cam, title='camera-captured scene3')

# %% (4) [local] Project and capture images needed for adversarial attack and PCNet/CompenNet++ training
data_root  = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy'  # rename it to describe your scene (better to use imageNet labels)
setup_path = join(data_root, 'setups', setup_name)
ut.make_setup_subdirs(setup_path)
OmegaConf.save(setup_info, join(setup_path, 'setup_info.yml'))

for phase in ['ref', 'cb', 'train', 'test']:
    if phase in ['ref', 'cb']:
        prj_input_path = join(setup_path, 'prj/raw', phase)
    else:
        prj_input_path = join(data_root, 'prj_share', phase)
    cam_cap_path = join(setup_path, 'cam/raw', phase)

    # pure color images for reference
    if phase == 'ref':
        ut.save_imgs(np.ones((1, *setup_info['prj_im_sz'], 3), dtype = np.uint8) * 255 * 0                           , prj_input_path, idx = 0) # black
        ut.save_imgs(np.ones((1, *setup_info['prj_im_sz'], 3), dtype = np.uint8) * 255 * setup_info['prj_brightness'], prj_input_path, idx = 1) # gray
        ut.save_imgs(np.ones((1, *setup_info['prj_im_sz'], 3), dtype = np.uint8) * 255 * 1                           , prj_input_path, idx = 2) # white

    # two-shifted checkerboard patterns to separate direct and indirect illuminations
    if phase == 'cb':
        num_squares = 32  # num squares per half image
        cb_sz = setup_info['prj_im_sz'][1] // (num_squares * 2)
        ut.save_imgs((checkerboard(cb_sz, num_squares) > 0.5).astype('uint8')[None, ..., None] * 255, prj_input_path, idx=0)
        ut.save_imgs((checkerboard(cb_sz, num_squares) < 0.5).astype('uint8')[None, ..., None] * 255, prj_input_path, idx=1)

    # project and capture the scene with various projections
    ut.project_capture_data(prj_input_path, cam_cap_path, setup_info,cam)

    if phase == 'ref':
        # double check if the scene image is correctly classified by all classifiers
        cam_scene = ut.torch_imread(join(setup_path, 'cam/raw/ref/img_0002.png'))
        classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
        pred_labels, _ = query_multi_classifiers(cam_scene, setup_info['classifier_crop_sz'], classifier_names, imagenet_labels, device, device_ids)
        assert pred_labels.count(pred_labels[0]) == len(classifier_names), 'Classifiers made different predictions!'

print(f'Finished data capturing, you may want to transfer the data to the server for SPAA/PerC-AL+CompenNet++ training and simulated attack')

# %% (5.1) [server] Train PCNet and perform SPAA (you may want to transfer data to the server and train/attack from that machine)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy'  # debug
setup_path = join(data_root, 'setups', setup_name)

attacker_name = ['original SPAA','CAM','USPAA','all']
# get the attack configs
# PCNet + attention
spaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True, d_threshes = [2, 3, 4, 5])

# start CAM-SPAA attack
spaa_cfg = run_projector_based_attack(spaa_cfg)

print(f'Finish SPAA attack, you may want to transfer the data in {join(setup_path, "prj/adv")} to the local machine for real projector-based attacks')

# %% 【CAM, original】(5.2) [local]  Project and capture SPAA generated adversarial projections, then summarize the attack results
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy'  # debug
setup_path = join(data_root, 'setups', setup_name)

# TODO: attacker_name = ['original SPAA','CAM']
attacker_name = ['original SPAA','CAM']

# get the attack configs
spaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)
project_capture_real_attack(spaa_cfg,cam)


# %% 【CAM, original】(5.3) [server] 【CAM, original】 summarize the attack
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
# device_ids = [0,1]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy'  # debug
setup_path = join(data_root, 'setups', setup_name)
# universal_dynamic_CAM is not tested because its performance is not better
# attacker_name = ['original SPAA','CAM','USPAA','all','universal_dynamic_CAM']
attacker_name = ['original SPAA','CAM']
spaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)

# summarize the result
spaa_ret = summarize_single_attacker(attacker_name, data_root=spaa_cfg.data_root, setup_list=spaa_cfg.setup_list,
                                     device=spaa_cfg.device, device_ids=spaa_cfg.device_ids, pose = 'original')

spaa_ret_USPAA = summarize_single_attacker_for_UAP_task(attacker_name, data_root=spaa_cfg.data_root, setup_list=spaa_cfg.setup_list,
                                     device=spaa_cfg.device, device_ids=spaa_cfg.device_ids)


# %% 【USPAA , all】 (5.2) [local] Project and capture SPAA generated adversarial projections, then summarize the attack results
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = ['teddy']  # debug
attacker_name = ['USPAA','all']

# get the attack configs
spaa_cfg = get_attacker_all_cfg(attacker_name=attacker_name, data_root=data_root, setup_list=setup_name, device_ids=device_ids, plot_on=True)
project_capture_real_attack(spaa_cfg,cam)

# %% 【USPAA , all】(5.3)  [server] summarize the attack
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy'  # debug
setup_path = join(data_root, 'setups', setup_name)
attacker_name = ['original SPAA','CAM','USPAA','all']

spaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)

spaa_ret = summarize_single_universal_attacker(attacker_name=['all','USPAA'], data_root=spaa_cfg.data_root, setup_list=spaa_cfg.setup_list,
                                     device=spaa_cfg.device, device_ids=spaa_cfg.device_ids, pose = 'original')

# %% [changed pose]

# %% [changed pose]【CAM, original】(6.2) [local] Project and capture SPAA generated adversarial projections, then summarize the attack results
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'basketball'  # debug
setup_path = join(data_root, 'setups', setup_name)
attacker_name = ['original SPAA','CAM']
# get the attack configs
spaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)
project_capture_real_attack(spaa_cfg,cam)

# %% [changed pose]【CAM, original】(6.3) [server] 【CAM, original】 summarize the attack
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
# device_ids = [0,1]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'basketball'  # debug
setup_path = join(data_root, 'setups', setup_name)
attacker_name = ['original SPAA','CAM','USPAA','all','universal_dynamic_CAM']

spaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)

spaa_ret = summarize_single_attacker(attacker_name=['original SPAA','CAM'], data_root=spaa_cfg.data_root, setup_list=spaa_cfg.setup_list,
                                     device=spaa_cfg.device, device_ids=spaa_cfg.device_ids, pose = 'changed')

spaa_ret_USPAA = summarize_single_attacker_for_UAP_task(attacker_name=['original SPAA','CAM'], data_root=spaa_cfg.data_root, setup_list=spaa_cfg.setup_list,
                                     device=spaa_cfg.device, device_ids=spaa_cfg.device_ids)

# %% [changed pose]【USPAA , all】 (6.2) [local] Project and capture SPAA generated adversarial projections, then summarize the attack results
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = ['basketball']  # debug
# setup_path = join(data_root, 'setups', setup_name)
attacker_name = ['USPAA','all','universal_dynamic_CAM']

# get the attack configs
spaa_cfg = get_attacker_all_cfg(attacker_name=attacker_name, data_root=data_root, setup_list=setup_name, device_ids=device_ids, plot_on=True)
project_capture_real_attack(spaa_cfg,cam)

# %% [changed pose]【USPAA , all】(6.3)  [server] summarize the attack
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
# device_ids = [0,1]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'basketball'  # debug
setup_path = join(data_root, 'setups', setup_name)
attacker_name = ['original SPAA','CAM','USPAA','all','universal_dynamic_CAM']

spaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)

# spaa_ret = summarize_single_universal_attacker(attacker_name=['USPAA','all','universal_dynamic_CAM'], data_root=spaa_cfg.data_root, setup_list=spaa_cfg.setup_list,
#                                      device=spaa_cfg.device, device_ids=spaa_cfg.device_ids)
spaa_ret = summarize_single_universal_attacker(attacker_name=['USPAA','all'], data_root=spaa_cfg.data_root, setup_list=spaa_cfg.setup_list,
                                     device=spaa_cfg.device, device_ids=spaa_cfg.device_ids, pose = 'changed')
# %% (7.1) [end!]

# %% (8) [local or server] Summarize results of all projector-based attackers (SPAA, PerC-AL+CompenNet++ and One-pixel DE) on all setups
data_root = abspath(join(os.getcwd(), '../../data'))
setup_list = [
'basketball_original','basketball_3_60','basketball_3_75','basketball_3_105','basketball_3_120','basketball_zoomin5mm','basketball_zoomout5mm',
'backpack_original','backpack_3_60','backpack_3_75',
    'backpack_3_105',
    'backpack_3_120',
'backpack_zoomin5mm','backpack_zoomout5mm',
'envelope_original','envelope_3_60','envelope_3_75','envelope_3_105','envelope_3_120','envelope_zoomin5mm','envelope_zoomout5mm',
'lotion_original','lotion_3_60','lotion_3_75','lotion_3_105','lotion_3_120','lotion_zoomin5mm','lotion_zoomout5mm',
'packet_original','packet_3_60','packet_3_75','packet_3_105','packet_3_120','packet_zoomin5mm','packet_zoomout5mm',
'paper_towel_original','paper_towel_3_60','paper_towel_3_75','paper_towel_3_105','paper_towel_3_120','paper_towel_zoomin5mm','paper_towel_zoomout5mm',
'sunscreen_original','sunscreen_3_60','sunscreen_3_75','sunscreen_3_105','sunscreen_3_120','sunscreen_zoomin5mm','sunscreen_zoomout5mm',
'teddy_original','teddy_3_60','teddy_3_75','teddy_3_105','teddy_3_120','teddy_zoomin5mm','teddy_zoomout5mm',
'crock pot_original','crock pot_3_60','crock pot_3_75','crock pot_3_105','crock pot_3_120','crock pot_zoomin5mm','crock pot_zoomout5mm',
'coffee mug_original','coffee mug_3_60','coffee mug_3_75','coffee mug_3_105','coffee mug_3_120','coffee mug_zoomin5mm','coffee mug_zoomout5mm',
]

attacker_names = ['original SPAA','CAM','USPAA','all']
all_ret, pivot_table = summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs=False)
# all_ret, pivot_table = summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs=True)

print(f'\n------------------ Pivot table of {len(setup_list)} setups in {data_root} ------------------')
print(pivot_table.to_string(index=True, float_format='%.4f'))
