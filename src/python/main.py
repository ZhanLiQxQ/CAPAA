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
from classifier import query_multi_classifiers
from img_proc import checkerboard, center_crop as cc
from projector_based_attack_all import run_projector_based_attack, summarize_single_universal_attacker, project_capture_real_attack, \
    summarize_all_attackers, get_attacker_cfg, get_attacker_all_cfg
from projector_based_attack_cam import summarize_single_attacker, summarize_single_attacker_for_UAP_task

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
#
# create projector window with different brightnesses, and check whether the exposures are correct
prj_fig = list()
for brightness in [0, setup_info['prj_brightness'], 1.0]:
    prj_fig.append(ut.init_prj_window(*setup_info['prj_screen_sz'], brightness, setup_info['prj_offset']))

# live preview the camera frames, make sure the square preview looks correct: the camera is in focus; the object is centered;
# when brightness=0.5 the illumination looks reasonable;

print('Previewing the camera, make sure everything looks good and press "q" to exit...')
cam = ut.init_cam(setup_info['cam_raw_sz'])
ut.preview_cam(setup_info['cam_raw_sz'], (min(setup_info['cam_raw_sz']), min(setup_info['cam_raw_sz'])),cam)

# %% (2) [local] Check whether the projector and the camera are in sync by capturing the numbers data
data_root      = abspath(join(os.getcwd(), '../../data'))
setup_name     = 'sync_test'
setup_path     = join(data_root  , 'setups/'    , setup_name)

# project and capture, then save the images to cam_cap_path
phase          = 'numbers'
prj_input_path = join(data_root , 'prj_share', phase)
cam_cap_path   = join(setup_path, 'cam/raw'  , phase)

ut.project_capture_data(prj_input_path, cam_cap_path, setup_info,cam)

# %% (3) [local] Check if the object is correctly classified by *all* three classifiers, if not, go back to step (1) and adjust your ProCams and scene
print('Recognizing the camera-captured scene using the given classifiers...')
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

# %% (3.2)
# drop frames (avoid the wrong frame) and capture
for j in range(0, 100):
    _, im_cam = cam.read()
im_cam = cc(torch.Tensor(cv.cvtColor(cv.resize(cc(im_cam, setup_info['cam_crop_sz']), setup_info['cam_im_sz'], interpolation=cv.INTER_AREA), cv.COLOR_BGR2RGB)).permute(2, 0, 1) / 255, setup_info['classifier_crop_sz'])
print("Close the window to continue...")
ut.fs(im_cam, title='camera-captured scene(close it to continue)')


# check results of each classifier, please make sure the scene is correctly classified by *all* classifiers
# 'vit_b_16' was not used in the main paper, but the extra experiments in supplementary included it for completeness
classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
query_multi_classifiers(im_cam, setup_info['classifier_crop_sz'], classifier_names, imagenet_labels, device, device_ids)

# %% (4) [local] Project and capture images needed for adversarial attack and PCNet training
data_root  = abspath(join(os.getcwd(), '../../data'))

setup_name = 'teddy'  # Rename to describe your scene (preferably using ImageNet labels).
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

print(f'Finished data capturing, you may want to transfer the data to the server for CAPAA training and simulated attack')

# %% (5.1) [server] Train PCNet and perform CAPAA (transfer data to server and run training/attacks there if needed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]

data_root      = abspath(join(os.getcwd(), '../../data'))

setup_name = 'teddy'  # Must be consistent with the setup name used in data capturing.
setup_path = join(data_root, 'setups', setup_name)

attacker_name = ['SPAA','CAPAA (classifier-specific)','CAPAA (without attention)','CAPAA']
# get the attack configs
capaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True, d_threshes = [2,3,4,5])

# Generate adversarial patterns and save them to 'setup_path/prj/adv'.
capaa_cfg = run_projector_based_attack(capaa_cfg)

print(f'Finished generating adversarial projections. Transfer data in {join(setup_path, "prj/adv")} to local machine for real-world projector-based attacks.')

# %% 【CAPAA (classifier-specific), SPAA】(5.2) [local] Project and capture adversarial projections generated by 'SPAA' and 'CAPAA (classifier-specific)'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy'  # consistent with the setup name above
setup_path = join(data_root, 'setups', setup_name)
classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
attacker_name = ['SPAA','CAPAA (classifier-specific)']

# get the attack configs
capaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)
project_capture_real_attack(capaa_cfg,cam)
print(f'Finish real projector-based attacks for SPAA and CAPAA (classifier-specific), you may want to transfer the data in {join(setup_path, "cam/raw/adv")} to the server for attack summarization.')

# %% 【CAPAA (classifier-specific), SPAA】(5.3) [server] summarize the attack of 'SPAA' and 'CAPAA (classifier-specific)'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
# device_ids = [0,1]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy'  # debug
setup_path = join(data_root, 'setups', setup_name)
classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
attacker_name = ['SPAA','CAPAA (classifier-specific)']

capaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)

# summarize the result against specific targeted classifier
capaa_ret = summarize_single_attacker(attacker_name, data_root=capaa_cfg.data_root, setup_list=capaa_cfg.setup_list,
                                     device=capaa_cfg.device, device_ids=capaa_cfg.device_ids, pose = 'original', classifier_names = classifier_names)
# test the attack performance of generated projections against all classifiers (for classifier-agnostic adversarial attack)
capaa_ret_classifier_agnostic = summarize_single_attacker_for_UAP_task(attacker_name, data_root=capaa_cfg.data_root, setup_list=capaa_cfg.setup_list,
                                     device=capaa_cfg.device, device_ids=capaa_cfg.device_ids)


# %% 【CAPAA (without attention), CAPAA】 (5.4) [local] Project and capture adversarial projections generated by 'CAPAA (w/o attention)' and 'CAPAA'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = ['teddy']  # debug
attacker_name = ['CAPAA (without attention)','CAPAA']

# get the attack configs
capaa_cfg = get_attacker_all_cfg(attacker_name=attacker_name, data_root=data_root, setup_list=setup_name, device_ids=device_ids, plot_on=True)
project_capture_real_attack(capaa_cfg,cam)
print(f'Finish real projector-based attacks for CAPAA (w/o attention) and CAPAA, you may want to transfer the data in {join(setup_path, "cam/raw/adv")} to the server for attack summarization.')

# %% 【CAPAA (without attention), CAPAA】(5.5) [server] summarize the attack of 'CAPAA (w/o attention)' and 'CAPAA'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy'  # debug
setup_path = join(data_root, 'setups', setup_name)
classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
attacker_name = ['CAPAA (without attention)','CAPAA']

capaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)

# summarize the result against all classifiers (for classifier-agnostic adversarial attack task)
capaa_ret = summarize_single_universal_attacker(attacker_name=attacker_name, data_root=capaa_cfg.data_root, setup_list=capaa_cfg.setup_list,
                                     device=capaa_cfg.device, device_ids=capaa_cfg.device_ids, pose = 'original', classifier_names=classifier_names)

print(f'Finish attacks under the original camera pose, you may want to change the camera capture pose')
# %% [changed pose] (6.1) [local] Now you may change the camera capture pose.
# Manually adjust the camera pose (angle, distance, zoom) for robustness testing.

# %% [changed pose](6.2) [local] Project and capture adversarial projections generated by 'SPAA' and 'CAPAA (classifier-specific)'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy'  # debug
setup_path = join(data_root, 'setups', setup_name)
attacker_name = ['SPAA','CAPAA (classifier-specific)']
# get the attack configs
capaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)
project_capture_real_attack(capaa_cfg,cam)

# %% [changed pose]【CAPAA (classifier-specific), SPAA】(6.3) summarize the attack of 'SPAA' and 'CAPAA (classifier-specific)'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
# device_ids = [0,1]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'teddy'  # debug
setup_path = join(data_root, 'setups', setup_name)
classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
attacker_name = ['SPAA','CAPAA (classifier-specific)']

capaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)

# The 'pose' parameter here should reflect the new camera pose data being analyzed.
capaa_ret = summarize_single_attacker(attacker_name = attacker_name, data_root=capaa_cfg.data_root, setup_list=capaa_cfg.setup_list,
                                     device=capaa_cfg.device, device_ids=capaa_cfg.device_ids, pose = 'changed', classifier_names=classifier_names)

capaa_ret_classifier_agnostic = summarize_single_attacker_for_UAP_task(attacker_name = attacker_name, data_root=capaa_cfg.data_root, setup_list=capaa_cfg.setup_list,
                                     device=capaa_cfg.device, device_ids=capaa_cfg.device_ids, classifier_names= classifier_names)

# %% [changed pose]【CAPAA (classifier-specific), CAPAA】 (6.4) [local] Project and capture adversarial projections generated by 'CAPAA (w/o attention)' and 'CAPAA'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = ['basketball']  # debug
# setup_path = join(data_root, 'setups', setup_name)

attacker_name = ['CAPAA (without attention)','CAPAA']

# get the attack configs
capaa_cfg = get_attacker_all_cfg(attacker_name=attacker_name, data_root=data_root, setup_list=setup_name, device_ids=device_ids, plot_on=True)
project_capture_real_attack(capaa_cfg,cam)

# %% [changed pose] (6.5)  [server] summarize the attack of 'CAPAA (w/o attention)' and 'CAPAA'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
# device_ids = [0,1]
data_root = abspath(join(os.getcwd(), '../../data'))
setup_name = 'basketball'  # debug
setup_path = join(data_root, 'setups', setup_name)
classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
attacker_name = ['CAPAA (without attention)','CAPAA']
capaa_cfg = get_attacker_cfg(attacker_name = attacker_name, data_root=data_root, setup_list=[setup_name], device_ids=device_ids, plot_on=True)

capaa_ret = summarize_single_universal_attacker(attacker_name=attacker_name, data_root=capaa_cfg.data_root, setup_list=capaa_cfg.setup_list,
                                     device=capaa_cfg.device, device_ids=capaa_cfg.device_ids, pose = 'changed',classifier_names=classifier_names)

print(f'Finish attacks under the changed camera pose, you may want to change the camera capture pose and repeat the steps from (6.1) to (6.5) for further experiments.')

# %% (7) [local] Further experiments: vary camera pose or setup.
# Repeat steps (6.1) to (6.5) with different camera poses, or try entirely new object setups.

# %% (8) [local or server] Summarize results of all projector-based attackers on all setups
data_root = abspath(join(os.getcwd(), '../../data'))
# Example list of setups, including original and varied poses.
# Naming convention might be 'object_original', 'object_posevariant1', 'object_posevariant2', etc.
setup_list = [
'basketball_original','basketball_3_60','basketball_3_75','basketball_3_105','basketball_3_120','basketball_zoomin5mm','basketball_zoomout5mm',
# ... (other setups) ...
'teddy_original','teddy_3_60','teddy_3_75','teddy_3_105','teddy_3_120','teddy_zoomin5mm','teddy_zoomout5mm',
'coffee mug_original','coffee mug_3_60','coffee mug_3_75','coffee mug_3_105','coffee mug_3_120','coffee mug_zoomin5mm','coffee mug_zoomout5mm',
]

attacker_names = ['SPAA','CAPAA (classifier-specific)','CAPAA (without attention)','CAPAA']

# Set recreate_stats_and_imgs to True if you want to re-generate plots and stats (time-consuming), False to load existing.
all_ret, pivot_table = summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs=False)
# all_ret, pivot_table = summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs=True)

print(f'\n------------------ Pivot table of {len(setup_list)} setups in {data_root} ------------------')
print(pivot_table.to_string(index=True, float_format='%.4f'))
