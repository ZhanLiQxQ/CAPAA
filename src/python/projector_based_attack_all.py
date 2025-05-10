"""
Useful functions for projector-based adversarial attack
"""
import os
from os.path import join, abspath
import numpy as np
import cv2 as cv
import pandas as pd
import torch
from torchvision import models
from torchvision import transforms
from omegaconf import DictConfig
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.utils import make_grid
from train_network import load_setup_info, train_eval_pcnet, train_eval_compennet_pp, get_model_train_cfg, ssim_fun
from img_proc import resize, insert_text, expand_boarder, expand_4d, center_crop as cc, invertGrid
import utils as ut
from utils import calc_img_dists
from perc_al.differential_color_functions import rgb2lab_diff, ciede2000_diff,deltaE
import itertools
from classifier import Classifier, load_imagenet_labels
from one_pixel_attacker import ProjectorOnePixelAttacker
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from grad_cam import GradCAM,GradCAMpp,resize_to_original
import math
from projector_based_attack_cam import spaa,attack_results_change_pose,attack_results,summarize_single_attacker_for_UAP_task
from vit_model import vit_base_patch16_224

class LargeMarginSoftmaxLoss(nn.Module):
    def __init__(self, embedding_size, class_num, margin=4):
        super(LargeMarginSoftmaxLoss, self).__init__()
        self.margin = margin
        self.class_num = class_num
        self.embedding_size = embedding_size

        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # 归一化权重和输入
        normed_weight = F.normalize(self.weight, p=2, dim=1)
        normed_input = F.normalize(input, p=2, dim=1)

        # 计算余弦相似度
        cosine_similarity = torch.mm(normed_input, normed_weight.t())

        # 选择目标类别的余弦相似度
        target_logit = cosine_similarity[torch.arange(0, input.size(0)), label].view(-1, 1)

        # 计算margin倍的角度
        sin_target = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_margin_target = target_logit * math.cos(self.margin) - sin_target * math.sin(self.margin)

        # 增加margin
        one_hot = torch.zeros_like(cosine_similarity)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = cosine_similarity * (1 - one_hot) + cos_margin_target * one_hot

        # 计算交叉熵损失
        loss = F.cross_entropy(output, label)
        return loss

def max_mask(a, b, c):
    max_values, _ = torch.max(torch.stack([a, b, c]), dim=0)
    mask = torch.eq(a, max_values)
    return mask.detach().cpu().numpy()

class ReshapeTransform:
    def __init__(self, model):
        input_size = model.patch_embed.img_size
        patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        result = x[:, 1:, :].reshape(x.size(0),
                                     self.h,
                                     self.w,
                                     x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def run_projector_based_attack(cfg):
    attacker_name_list = cfg.attacker_name

    # assert attacker_name in ['original SPAA','CAM','USPAA','all'], f'{attacker_name} not supported!'
    # assert (attacker_name != 'One-pixel_DE') or (len(cfg.setup_list) == 1), f'{attacker_name} does not support attacking multiple setups simultaneously!'

    # set PyTorch device to GPU
    device = torch.device(cfg.device)
    ut.reset_rng_seeds(0)

    for setup_name in cfg.setup_list:
        for attacker_name in attacker_name_list:
            print(f'\nPerforming [{attacker_name}] attack on [{setup_name}]')

            # load setup info and images
            setup_path = join(cfg.data_root, 'setups', setup_name)
            setup_info = load_setup_info(setup_path)
            cp_sz = setup_info.classifier_crop_sz
            cam_scene  = cc(ut.torch_imread(join(setup_path, 'cam/raw/ref/img_0002.png')), setup_info.cam_im_sz[::-1]) # cam-captured scene (Is), ref/img_0002

            # ImageNet and targeted attack labels
            imagenet_labels = load_imagenet_labels(join(cfg.data_root, 'imagenet1000_clsidx_to_labels.txt'))
            target_labels   = load_imagenet_labels(join(cfg.data_root, 'imagenet10_clsidx_to_labels.txt'))  # targeted attack labels

            dl_based = attacker_name in ['original SPAA','CAM','USPAA','all','universal_dynamic_CAM']
            if dl_based:
                cam_scene = cam_scene.to(device)

                # 只有第一次训练PCNet除此之外都是loading PCNet
                if attacker_name == 'original SPAA':
                    #for debug
                    # TODO: change True to False
                    load_pretrained = True
                else:
                    load_pretrained = True
                # train or load PCNet/CompenNet++ model
                model_cfg = get_model_train_cfg(model_list=None, data_root=cfg.data_root, setup_list=[setup_name], device_ids=cfg.device_ids,
                                                load_pretrained=load_pretrained, plot_on=cfg.plot_on)
                model_cfg.model_list = ['PCNet']
                # model_cfg.max_iters = 100 # debug
                model, model_ret, model_cfg, fine_grid = train_eval_pcnet(model_cfg)

                # elif attacker_name == 'PerC-AL+CompenNet++':
                #     model_cfg.model_list = ['CompenNet++']
                #     # model_cfg.max_iters = 100 # debug
                #     model, model_ret, model_cfg = train_eval_compennet_pp(model_cfg)

                # set to evaluation mode
                model.eval()

                # fix model weights
                for param in model.parameters():
                    param.requires_grad = False
            else:
                # Nichols & Jasper's projector-based One-pixel DE attacker
                one_pixel_de = ProjectorOnePixelAttacker(imagenet_labels, setup_info)
                im_prj_org = setup_info['prj_brightness'] * torch.ones(3, *setup_info['prj_im_sz'])
                one_pixel_de.im_cam_org = cam_scene
                model_cfg = None  # no deep learning-based models

            attacker_cfg_str = to_attacker_cfg_str(attacker_name)[0]
            cfg.model_cfg = model_cfg

            # we perform n = 10 targeted attacks and 1 untargeted attack
            n = 10
            target_labels = dict(itertools.islice(target_labels.items(), n))
            target_idx    = list(target_labels.keys())

            # takes 42.5s for each loss + thresh (all 3 classifiers). In total 42.5*4*6/60=17min for 4 loss, 6 thresh, 3 classifiers
            for stealth_loss in cfg.stealth_losses:
                for d_thr in cfg.d_threshes:
                    #判断是否是universal
                    if attacker_name in ['USPAA','all','universal_dynamic_CAM']:
                        # 判断是否需要CAM
                        if attacker_name == 'USPAA':
                            attention_use = False
                        else:
                            attention_use = True

                        attack_ret_folder = join(attacker_cfg_str, stealth_loss, str(d_thr), "classifier_all")
                        cam_raw_adv_path = join(setup_path, 'cam/raw/adv', attack_ret_folder)
                        cam_infer_adv_path = join(setup_path, 'cam/infer/adv', attack_ret_folder)
                        prj_adv_path = join(setup_path, 'prj/adv', attack_ret_folder)
                        warped_prj_adv_path = join(setup_path, 'prj/warped_adv', attack_ret_folder)

                        print("Classifier names:", cfg.classifier_names)
                        print("Number of classifier names:", len(cfg.classifier_names))


                        # get the true label of the current scene
                        classifier0 = Classifier(cfg.classifier_names[0], device, cfg.device_ids, fix_params=True, sort_results=dl_based)
                        classifier1 = Classifier(cfg.classifier_names[1], device, cfg.device_ids, fix_params=True,
                                                 sort_results=dl_based)
                        classifier2 = Classifier(cfg.classifier_names[2], device, cfg.device_ids, fix_params=True,
                                                 sort_results=dl_based)
                        classifier3 = Classifier(cfg.classifier_names[3], device, cfg.device_ids, fix_params=True,
                                                 sort_results=dl_based)
                        with torch.no_grad():
                            _, p0, pred_idx0 = classifier0(cam_scene, cp_sz)
                            _, p1, pred_idx1 = classifier1(cam_scene, cp_sz)
                            _, p2, pred_idx2 = classifier2(cam_scene, cp_sz)
                            _, p3, pred_idx3 = classifier3(cam_scene, cp_sz)
                        true_idx0 = pred_idx0[
                            0, 0] if dl_based else p0.argmax()  # true label index of the scene given by the current classifier
                        true_label0 = imagenet_labels[true_idx0]

                        true_idx1 = pred_idx1[
                            0, 0] if dl_based else p1.argmax()  # true label index of the scene given by the current classifier
                        true_label1 = imagenet_labels[true_idx1]

                        true_idx2 = pred_idx2[
                            0, 0] if dl_based else p2.argmax()  # true label index of the scene given by the current classifier
                        true_label2 = imagenet_labels[true_idx2]

                        true_idx3 = pred_idx3[
                            0, 0] if dl_based else p3.argmax()  # true label index of the scene given by the current classifier
                        true_label3 = imagenet_labels[true_idx3]
                        print(
                            f'\n-------------------- [{attacker_name}] attacking [{"classifier_all"}], original prediction: ({true_label0}, '
                            f'p={p0.max():.2f}), Loss: [{stealth_loss}], d_thr: [{d_thr}] --------')
                        print(
                            f'\n-------------------- [{attacker_name}] attacking [{"classifier_all"}], original prediction: ({true_label1}, '
                            f'p={p1.max():.2f}), Loss: [{stealth_loss}], d_thr: [{d_thr}] --------')
                        print(
                            f'\n-------------------- [{attacker_name}] attacking [{"classifier_all"}], original prediction: ({true_label2}, '
                            f'p={p2.max():.2f}), Loss: [{stealth_loss}], d_thr: [{d_thr}] --------')

                        print(
                            f'\n-------------------- [{attacker_name}] attacking [{"classifier_all"}], original prediction: ({true_label3}, '
                            f'p={p3.max():.2f}), Loss: [{stealth_loss}], d_thr: [{d_thr}] --------')

                        # untargeted attack
                        targeted_attack = False
                        print(f'[Untargeted] attacking [{"classifier_all"}]...')
                        # 判断是否是动态CAM
                        if attacker_name == "universal_dynamic_CAM":
                            cam_infer_adv_untar, prj_adv_untar,warped_prj_untar = spaa_universal_dynamic_CAM(model, classifier0, classifier1,
                                                                                classifier2, imagenet_labels,
                                                                                target_idx, [true_idx0],
                                                                                [true_idx1], [true_idx2],
                                                                                targeted_attack, cam_scene, d_thr,
                                                                                stealth_loss,
                                                                                cfg.device, setup_info, fine_grid,
                                                                                attention_use)


                        else:
                            cam_infer_adv_untar, prj_adv_untar,warped_prj_untar = spaa_universal(model, classifier0, classifier1,
                                                                                classifier2, classifier3, imagenet_labels,
                                                                                target_idx, [true_idx0],
                                                                                [true_idx1], [true_idx2],[true_idx3],
                                                                                targeted_attack, cam_scene, d_thr,
                                                                                stealth_loss,
                                                                                cfg.device, setup_info, fine_grid,
                                                                                attention_use)

                        # targeted attack (batched)
                        targeted_attack = True
                        v = 7  # we only show one adversarial target in the console, v is the index
                        print(
                            f'\n[ Targeted ] attacking [{"classifier_all"}], target: ({imagenet_labels[target_idx[v]]})...')

                        if attacker_name == "universal_dynamic_CAM":
                            cam_infer_adv_tar, prj_adv_tar,warped_prj_tar = spaa_universal_dynamic_CAM(model, classifier0, classifier1,
                                                                            classifier2, imagenet_labels, target_idx,
                                                                            [true_idx0],
                                                                            [true_idx1], [true_idx2],
                                                                            targeted_attack, cam_scene, d_thr,
                                                                            stealth_loss,
                                                                            cfg.device, setup_info, fine_grid,
                                                                            attention_use)
                        else:
                            cam_infer_adv_tar, prj_adv_tar,warped_prj_tar  = spaa_universal(model, classifier0, classifier1,
                                                                            classifier2, classifier3, imagenet_labels, target_idx,
                                                                            [true_idx0],
                                                                            [true_idx1], [true_idx2],[true_idx3],
                                                                            targeted_attack, cam_scene, d_thr,
                                                                            stealth_loss,
                                                                            cfg.device, setup_info, fine_grid,
                                                                            attention_use)

                        # save adversarial projections and inferred/real cam-captured ones
                        if dl_based:
                            ut.save_imgs(expand_4d(torch.cat((cam_infer_adv_tar, cam_infer_adv_untar), 0)), cam_infer_adv_path)
                            ut.save_imgs(expand_4d(torch.cat((prj_adv_tar, prj_adv_untar), 0)), prj_adv_path)
                            ut.save_imgs(expand_4d(torch.cat((warped_prj_tar, warped_prj_untar), 0)),
                                     warped_prj_adv_path)
                        # else:
                        #     ut.save_imgs(expand_4d(cam_raw_adv_untar), cam_raw_adv_path, idx=n)
                        #     ut.save_imgs(expand_4d(prj_adv_untar), prj_adv_path, idx=n)

                    else:
                        for classifier_name in cfg.classifier_names:
                            # 判断是否需要CAM
                            if attacker_name == 'CAM':
                                attention_use = True
                            else:
                                attention_use = False
                            attack_ret_folder = join(attacker_cfg_str, stealth_loss, str(d_thr), classifier_name)
                            cam_raw_adv_path = join(setup_path, 'cam/raw/adv', attack_ret_folder)
                            cam_infer_adv_path = join(setup_path, 'cam/infer/adv', attack_ret_folder)
                            prj_adv_path = join(setup_path, 'prj/adv', attack_ret_folder)
                            warped_prj_adv_path = join(setup_path, 'prj/warped_adv', attack_ret_folder)

                            # get the true label of the current scene
                            classifier = Classifier(classifier_name, device, cfg.device_ids, fix_params=True,
                                                    sort_results=dl_based)
                            with torch.no_grad():
                                _, p, pred_idx = classifier(cam_scene, cp_sz)
                            true_idx = pred_idx[
                                0, 0] if dl_based else p.argmax()  # true label index of the scene given by the current classifier
                            true_label = imagenet_labels[true_idx]

                            print(
                                f'\n-------------------- [{attacker_name}] attacking [{classifier_name}], original prediction: ({true_label}, '
                                f'p={p.max():.2f}), Loss: [{stealth_loss}], d_thr: [{d_thr}] --------')

                            # untargeted attack
                            targeted_attack = False
                            print(f'[Untargeted] attacking [{classifier_name}]...')

                            cam_infer_adv_untar, prj_adv_untar, warped_prj_untar = spaa(model, classifier, imagenet_labels,
                                                                      [true_idx], targeted_attack, cam_scene, d_thr,
                                                                      stealth_loss, cfg.device, setup_info,
                                                                      fine_grid, attention_use)

                            # targeted attack (batched)
                            targeted_attack = True
                            v = 7  # we only show one adversarial target in the console, v is the index

                            print(
                                f'\n[ Targeted ] attacking [{classifier_name}], target: ({imagenet_labels[target_idx[v]]})...')
                            cam_infer_adv_tar, prj_adv_tar, warped_prj_tar = spaa(model, classifier, imagenet_labels, target_idx,
                                                                  targeted_attack, cam_scene, d_thr, stealth_loss,
                                                                  cfg.device, setup_info, fine_grid,
                                                                  attention_use)
                            ut.save_imgs(expand_4d(torch.cat((cam_infer_adv_tar, cam_infer_adv_untar), 0)),
                                         cam_infer_adv_path)
                            ut.save_imgs(expand_4d(torch.cat((prj_adv_tar, prj_adv_untar), 0)), prj_adv_path)
                            ut.save_imgs(expand_4d(torch.cat((warped_prj_tar, warped_prj_untar), 0)),
                                         warped_prj_adv_path)

            if dl_based:
                print(f'\nThe next step is to project and capture [{attacker_name}] generated adversarial projections in {join(setup_path, "prj/adv", attacker_cfg_str)}')
            else:
                print(f'\nThe next step is to inspect the camera-captured adversarial projections in {join(setup_path, "cam/raw/adv", attacker_cfg_str)}')
    return cfg


def project_capture_real_attack(cfg, cam):
    attacker_name_list = cfg.attacker_name
    for attacker_name in attacker_name_list:

        # assert attacker_name in ['SPAA',
        #                          'PerC-AL+CompenNet++'], f'{attacker_name} not supported, One-pixel_DE does not use this function!'
        # assert len(
        #     cfg.setup_list) == 1, f'The current attacker cfg contains multiple/or no setup_names in setup_list, it should have exactly one setup_name!'

        setup_path = join(cfg.data_root, 'setups', cfg.setup_list[0])
        setup_info = load_setup_info(setup_path)

        for stealth_loss in cfg.stealth_losses:
            for d_thr in cfg.d_threshes:
                for classifier_name in cfg.classifier_names:
                    attacker_cfg_str = to_attacker_cfg_str(attacker_name)[0]
                    # 这里是结果储存的文件名，可以看到是在后边加loss种类，d_thr，分类器种类
                    attack_ret_folder = join(attacker_cfg_str, stealth_loss, str(d_thr), classifier_name)
                    prj_input_path = join(setup_path, 'prj/adv', attack_ret_folder)
                    cam_cap_path = join(setup_path, 'cam/raw/adv', attack_ret_folder)
                    ut.project_capture_data(prj_input_path, cam_cap_path, setup_info, cam)


def get_attacker_cfg(attacker_name, data_root, setup_list, device_ids=[0], plot_on=True,d_threshes = [2, 3, 4, 5]):
    # default projector-based attacker configs
    cfg_default = DictConfig({})
    cfg_default.attacker_name = attacker_name
    # cfg_default.classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
    cfg_default.classifier_names = ['inception_v3', 'resnet18', 'vgg16']
    cfg_default.data_root = data_root
    cfg_default.setup_list = setup_list
    cfg_default.device = 'cuda'
    cfg_default.device_ids = device_ids
    cfg_default.plot_on = plot_on

    # cfg_default.stealth_losses = ['caml2', 'camdE', 'camdE_caml2' , 'camdE_caml2_prjl2']
    cfg_default.stealth_losses = ['camdE_caml2']
    cfg_default.d_threshes = d_threshes

    return cfg_default

def get_attacker_all_cfg(attacker_name, data_root, setup_list, device_ids=[0], load_pretrained=False, plot_on=True):
    # default projector-based attacker configs
    cfg_default = DictConfig({})
    cfg_default.attacker_name = attacker_name
    cfg_default.classifier_names = ['classifier_all']
    cfg_default.data_root = data_root
    cfg_default.setup_list = setup_list
    cfg_default.device = 'cuda'
    cfg_default.device_ids = device_ids
    cfg_default.load_pretrained = load_pretrained
    cfg_default.plot_on = plot_on

    # cfg_default.stealth_losses = ['caml2', 'camdE', 'camdE_caml2' , 'camdE_caml2_prjl2']
    cfg_default.stealth_losses = ['camdE_caml2']
    cfg_default.d_threshes = [2, 3, 4, 5]

    return cfg_default


def to_attacker_cfg_str(attacker_name):
    # assert attacker_name in ['SPAA', 'PerC-AL+CompenNet++', 'One-pixel_DE'], f'{attacker_name} not supported!'

    model_cfg = get_model_train_cfg(model_list=['PCNet'], single=True)
    model_cfg_str = f'{model_cfg.model_name}_{model_cfg.loss}_{model_cfg.num_train}_{model_cfg.batch_size}_{model_cfg.max_iters}'
    attacker_cfg_str = f'{attacker_name}_{model_cfg_str}'

    return attacker_cfg_str, model_cfg_str

def preprocess_image(
    img: np.ndarray, mean=[
        0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def reshape_transform(tensor, height=14, width=14):
    # 去掉cls token
    result = tensor[:, 1:, :].reshape(tensor.size(0),
    height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def spaa_universal(pcnet, classifier0, classifier1, classifier2, classifier3, imagenet_labels,  target_idx, target_idx0, target_idx1, target_idx2, target_idx3, targeted, cam_scene, d_thr, stealth_loss, device, setup_info, fine_grid = None, attention_use = False):
    # device = torch.device(device)
    print("Device configuration:", device)
    if targeted:
        num_target = len(target_idx)
    else:
        num_target = len(target_idx0)
    cp_sz = setup_info['classifier_crop_sz']

    # camera-captured scene image used for attacks
    cam_scene_batch = cam_scene.expand(num_target, -1, -1, -1).to(device)

    # projector input image
    im_gray = setup_info['prj_brightness'] * torch.ones(num_target, 3, *setup_info['prj_im_sz']).to(
        device)  # TODO: cam_train.mean() may be better?
    prj_adv = im_gray.clone()
    prj_adv.requires_grad = True

    # [debug] we perform batched targeted attacks, and we only show one adversarial target in console, v is the index
    v = 7 if targeted else 0

    # learning rates
    adv_lr = 2  # SPAA Algorithm 1's \beta_1: step size in minimizing adversarial loss
    col_lr = 1  # SPAA Algorithm 1's \beta_2: step size in minimizing stealthiness loss

    # 自建参数
    weight_init = 1 / 4  # weight for 4 classifier to linear combination
    weight_init_0 = (1 / 4)
    weight_init_1 = (1 / 4)
    weight_init_2 = (1 / 4)
    weight_init_3 = (1 / 4)

    # loss weights, lower adv_w or larger color loss weights reduce success rates but make prj_adv more imperceptible
    # SPAA Eq. 9: adv_w=1, prjl2_w=0, caml2_w=1, camdE_w=0, other combinations are shown in supplementary Sec. 3.2. Different stealthiness loss functions
    adv_w = 1  # adversarial loss weights
    adv_f = 0.1 # 反均方误差 adversarial loss weights
    prjl2_w = 0.1 if 'prjl2' in stealth_loss else 0  # projector input image l2 loss weights, SPAA paper prjl2_w=0
    caml2_w = 1 if 'caml2' in stealth_loss else 0  # camera captured image l2 loss weights
    camdE_w = 1 if 'camdE' in stealth_loss else 0  # camera captured image deltaE loss weights

    # SPAA Algorithm 1's pthr: threshold for adversarial confidence
    # lower it when SPAA has a good quality, otherwise increase (can get lower class_loss)
    p_thresh = 0.9  # if too high, the attack may not reach it and the output prj_adv is not perturbed, thus is all gray

    # iterative refine the input, SPAA Algorithm 1's K:number of iterations
    iters = 200  # TODO: improve it, we can early stop when attack requirements are met


    prj_adv_best = prj_adv.clone()
    cam_infer_best = cam_scene.repeat(prj_adv_best.shape[0], 1, 1, 1)
    col_loss_best = 1e6 * torch.ones(prj_adv_best.shape[0]).to(device)

    # CAM_attention_threshold 的制作

    '''grad-CAM
    model_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    target_layers_vgg16 = [model_vgg16.features]

    model_inception_v3 = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    target_layers_inception_v3 = [model_inception_v3.Mixed_7c]

    model_resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    target_layers_resnet18 = [model_resnet18.layer4]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trans_cam_scene = cam_scene.permute(1, 2, 0).cpu().numpy()
    trans_cam_scene = data_transform(trans_cam_scene)
    trans_cam_scene_batch = torch.unsqueeze(trans_cam_scene, dim=0)

    # 分别获取三种分类器的cam函数
    cam_vgg16 = GradCAM(model=model_vgg16, target_layers=target_layers_vgg16, use_cuda=True)
    cam_inception_v3 = GradCAM(model=model_inception_v3, target_layers=target_layers_inception_v3, use_cuda=True)
    cam_resnet18 = GradCAM(model=model_resnet18, target_layers=target_layers_resnet18, use_cuda=True)
    
    # 分别获取三种分类器的预测idx
    raw_score0, p0, idx0 = classifier0(cam_scene, cp_sz)
    raw_score1, p1, idx1 = classifier1(cam_scene, cp_sz)
    raw_score2, p2, idx2 = classifier2(cam_scene, cp_sz)
    pred_idx0 = idx0[0][0].tolist()
    pred_idx1 = idx1[0][0].tolist()
    pred_idx2 = idx2[0][0].tolist()

    # 利用raw_score的倒数作为权重
    # 获取当前物体的分别关于三个分类器的CAM
    if classifier0.name == 'vgg16':
        grayscale_cam0 = cam_vgg16(input_tensor=trans_cam_scene_batch, target_category=pred_idx0)  # (1 * 240 *320)
    elif classifier0.name == 'inception_v3':
        grayscale_cam0 = cam_inception_v3(input_tensor=trans_cam_scene_batch,
                                          target_category=pred_idx0)  # (1 * 240 *320)
    else:
        grayscale_cam0 = cam_resnet18(input_tensor=trans_cam_scene_batch, target_category=pred_idx0)  # (1 * 240 *320)

    if classifier1.name == 'vgg16':
        grayscale_cam1 = cam_vgg16(input_tensor=trans_cam_scene_batch, target_category=pred_idx1)  # (1 * 240 *320)
    elif classifier1.name == 'inception_v3':
        grayscale_cam1 = cam_inception_v3(input_tensor=trans_cam_scene_batch,
                                          target_category=pred_idx1)  # (1 * 240 *320)
    else:
        grayscale_cam1 = cam_resnet18(input_tensor=trans_cam_scene_batch, target_category=pred_idx1)  # (1 * 240 *320)

    if classifier2.name == 'vgg16':
        grayscale_cam2 = cam_vgg16(input_tensor=trans_cam_scene_batch, target_category=pred_idx2)  # (1 * 240 *320)
    elif classifier2.name == 'inception_v3':
        grayscale_cam2 = cam_inception_v3(input_tensor=trans_cam_scene_batch,
                                          target_category=pred_idx2)  # (1 * 240 *320)
    else:
        grayscale_cam2 = cam_resnet18(input_tensor=trans_cam_scene_batch, target_category=pred_idx2)  # (1 * 240 *320)

    
    
    
    '''




    model_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    target_layer_vgg16 = model_vgg16.features
    model_vgg16.eval()

    model_inception_v3 = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    target_layer_inception_v3 = model_inception_v3.Mixed_7c
    model_inception_v3.eval()

    model_resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    target_layer_resnet18 = model_resnet18.layer4
    model_resnet18.eval()

    # 加载预训练的 ViT 模型
    # model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
    model_vit = vit_base_patch16_224()
    model_vit.eval()

    model_vit.load_state_dict(torch.load("./vit_base_patch16_224.pth", map_location=device))
    # self.model.load_state_dict(load_state_dict_from_url(pretrained_model_url))
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trans_cam_scene = cam_scene.permute(1, 2, 0).cpu().numpy()
    trans_cam_scene = data_transform(trans_cam_scene)
    trans_cam_scene_batch = torch.unsqueeze(trans_cam_scene, dim=0)
    # for gradcam for vits, only resize to (224,224)
    resized_vit_trans_cam_scene_batch = resize(trans_cam_scene_batch, size=(224, 224)) # 1* 3* 224 * 224

    # 分别获取三种分类器的grad-campp函数
    cam_vgg16 = GradCAMpp(model_vgg16, target_layer_vgg16)
    cam_inception_v3 = GradCAMpp(model_inception_v3, target_layer_inception_v3)
    cam_resnet18 = GradCAMpp(model_resnet18, target_layer_resnet18)
    # 创建 GradCAM 对象
    cam_vit = GradCAM(model=model_vit,
                  target_layers=[model_vit.blocks[-1].norm1],
                  # 这里的target_layer要看模型情况，
                  # 比如还有可能是：target_layers = [model.blocks[-1].ffn.norm]
                  use_cuda=torch.cuda.is_available(),
                  reshape_transform=ReshapeTransform(model_vit))


    # 利用raw_score的倒数作为权重
    # 获取当前物体的分别关于三个分类器的CAM
    if classifier0.name == 'vgg16':
        grayscale_cam0 , _ = cam_vgg16(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    elif classifier0.name == 'inception_v3':
        grayscale_cam0 , _ = cam_inception_v3(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    elif classifier0.name == 'resnet18':
        grayscale_cam0 , _ = cam_resnet18(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    else:
        grayscale_cam0 = cam_vit(resized_vit_trans_cam_scene_batch)  # (1, 224,224)


    if classifier1.name == 'vgg16':
        grayscale_cam1 , _ = cam_vgg16(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    elif classifier1.name == 'inception_v3':
        grayscale_cam1 , _ = cam_inception_v3(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    elif classifier1.name == 'resnet18':
        grayscale_cam1 , _ = cam_resnet18(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    else:
        grayscale_cam1 = cam_vit(resized_vit_trans_cam_scene_batch)  # (1,3, 224,224)



    if classifier2.name == 'vgg16':
        grayscale_cam2 , _ = cam_vgg16(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    elif classifier2.name == 'inception_v3':
        grayscale_cam2 , _ = cam_inception_v3(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    elif classifier2.name == 'resnet18':
        grayscale_cam2, _ = cam_resnet18(trans_cam_scene_batch)  # (1, 1, 7, 10)
    else:
        grayscale_cam2 = cam_vit(resized_vit_trans_cam_scene_batch)  # (1,3, 224,224)


    if classifier3.name == 'vgg16':
        grayscale_cam3, _ = cam_vgg16(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    elif classifier3.name == 'inception_v3':
        grayscale_cam3, _ = cam_inception_v3(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    elif classifier3.name == 'resnet18':
        grayscale_cam3, _ = cam_resnet18(trans_cam_scene_batch)  # (1 * 3* 240 *320)
    else:
        grayscale_cam3 = cam_vit(resized_vit_trans_cam_scene_batch)  # (1,224,224)
    # 0到1
    grayscale_cam0 = resize_to_original(trans_cam_scene_batch, grayscale_cam0)
    # 0到1
    grayscale_cam1 = resize_to_original(trans_cam_scene_batch, grayscale_cam1)
    # 0到1
    grayscale_cam2 = resize_to_original(trans_cam_scene_batch, grayscale_cam2) # tensor (3, 240, 320)

    grayscale_cam3 = torch.from_numpy(grayscale_cam3).unsqueeze(0)  # tensor (1, 224, 224)
    # Note: vits GradCAM is (1,224,224), however, we should make it to 240 * 320
    grayscale_cam3 = resize_to_original(trans_cam_scene_batch, grayscale_cam3)

    grayscale_cam = (grayscale_cam0 + grayscale_cam1 + grayscale_cam2 + grayscale_cam3 ) / 4
    # (3, 240,320)
    # # 均值归一化
    # mean_val = grayscale_cam.mean(dim=(1, 2), keepdim=True)  # 沿空间维度计算均值，保持维度
    #
    # # 归一化到均值为 1
    # grayscale_cam = grayscale_cam / mean_val

    # grayscale_cam = torch.stack([torch.tensor(grayscale_cam)] * 3, dim=0)  # (3 * 240 * 320)

    # grayscale_cam = torch.stack([torch.from_numpy(grayscale_cam[0, :])] * 3, dim=0).clone().detach()
    CAM_attention = grayscale_cam.expand(num_target, -1, -1, -1).to(device)  # untargeted:(1 * 3 * 240 * 320),targeted:(10 * 3 * 240 * 320)


    # TODO：inverse_warping CAM
    # prj2cam_grid = pcnet.module.get_warping_grid()
    prj2cam_grid = fine_grid[0, :].unsqueeze(0)  # (1,240,320,2)
    cam2prj_grid = torch.Tensor(invertGrid(prj2cam_grid.detach().cpu(), setup_info['prj_im_sz']))[
        None]  # warps desired cam-captured image to the prj space

    # desired cam-captured image warped to the prj image space
    prj_cmp_init = F.grid_sample(CAM_attention.detach().cpu(),
                                    cam2prj_grid.expand(CAM_attention.shape[0], -1, -1, -1), align_corners=True)

    # desired cam-captured image fov warped to prj image space
    prj_mask = F.grid_sample(pcnet.module.get_mask().float()[None, None].detach().cpu(), cam2prj_grid,
                                align_corners=True)

    prj_CAM_attention = (prj_cmp_init * prj_mask).to(device)
    prj_CAM_attention = prj_CAM_attention[0, :]

    for i in range(0, iters):
        cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)
        raw_score0, p0, idx0 = classifier0(cam_infer, cp_sz)
        raw_score1, p1, idx1 = classifier1(cam_infer, cp_sz)
        raw_score2, p2, idx2 = classifier2(cam_infer, cp_sz)
        raw_score3, p3, idx3 = classifier3(cam_infer, cp_sz)


        pp0 = F.softmax(raw_score0, dim=1)
        pp1 = F.softmax(raw_score1, dim=1)
        pp2 = F.softmax(raw_score2, dim=1)
        pp3 = F.softmax(raw_score2, dim=1)

        ppp0 = F.softmax(raw_score0 / (i // 40 + 1), dim=1)
        ppp1 = F.softmax(raw_score1 / (i // 40 + 1), dim=1)
        ppp2 = F.softmax(raw_score2 / (i // 40 + 1), dim=1)
        ppp3 = F.softmax(raw_score2 / (i // 40 + 1), dim=1)

        # adversarial loss
        if targeted:
            adv_loss = weight_init_0 * adv_w * (- torch.log(ppp0[torch.arange(num_target), target_idx])).mean()\
                       + weight_init_1 * adv_w * (- torch.log(ppp1[torch.arange(num_target), target_idx])).mean()\
                       + weight_init_2 * adv_w * (- torch.log(ppp2[torch.arange(num_target), target_idx])).mean() \
                       + weight_init_3 * adv_w * (- torch.log(ppp3[torch.arange(num_target), target_idx])).mean()

        else:
            adv_loss = weight_init_0 * adv_w * (raw_score0[torch.arange(num_target), target_idx0]).mean() \
                       + weight_init_1 * adv_w * (raw_score1[torch.arange(num_target), target_idx1]).mean() \
                       + weight_init_2 * adv_w * (raw_score2[torch.arange(num_target), target_idx2]).mean() \
                       + weight_init_2 * adv_w * (raw_score3[torch.arange(num_target), target_idx3]).mean()

        prjl2 = torch.norm(im_gray - prj_adv, dim=1).mean(1).mean(1)  # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch = prjl2_w * prjl2

        # stealthiness loss: cam-captured image should look like cam_scene (L2 loss)
        caml2 = torch.norm(cam_scene_batch - cam_infer, dim=1).mean(1).mean(
            1)  # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch += caml2_w * caml2

        # stealthiness loss: cam-captured image should look like cam_scene (CIE deltaE 2000 loss)
        camdE = ciede2000_diff(rgb2lab_diff(cam_infer, device), rgb2lab_diff(cam_scene_batch, device), device).mean(
            1).mean(1)
        col_loss_batch += camdE_w * camdE

        # average stealthiness (color) losses
        col_loss = col_loss_batch.mean()

        # mask adversarial confidences that are higher than p_thresh
        mask_high_conf0 = p0[:, 0] > p_thresh
        mask_high_conf1 = p1[:, 0] > p_thresh
        mask_high_conf2 = p2[:, 0] > p_thresh
        mask_high_conf3 = p3[:, 0] > p_thresh
        mask_high_pert = (camdE > d_thr).detach().cpu().numpy()

        # alternating between the adversarial loss and the stealthiness (color) loss
        if targeted:
            mask_succ_adv0 = idx0[:, 0] == target_idx
            mask_succ_adv1 = idx1[:, 0] == target_idx
            mask_succ_adv2 = idx2[:, 0] == target_idx
            mask_succ_adv3 = idx3[:, 0] == target_idx
            mask_best_adv0 = mask_succ_adv0 & mask_high_conf0 & mask_high_pert
            mask_best_adv1 = mask_succ_adv1 & mask_high_conf1 & mask_high_pert
            mask_best_adv2 = mask_succ_adv2 & mask_high_conf2 & mask_high_pert
            mask_best_adv3 = mask_succ_adv3 & mask_high_conf3 & mask_high_pert

            mask_succ_adv = (mask_succ_adv0 & mask_succ_adv1 & mask_succ_adv2 & mask_succ_adv3)# 选出针对三个分类器都分类成功的图片
            mask_best_adv = (mask_best_adv0 & mask_best_adv1 & mask_best_adv2 & mask_best_adv3)# 选出针对三个分类器都分类成功且有高置信度的图片
        else:
            mask_succ_adv0 = idx0[:, 0] != target_idx0
            mask_succ_adv1 = idx1[:, 0] != target_idx1
            mask_succ_adv2 = idx2[:, 0] != target_idx2
            mask_succ_adv3 = idx3[:, 0] != target_idx3

            mask_best_adv0 = mask_succ_adv0 & mask_high_pert
            mask_best_adv1 = mask_succ_adv1 & mask_high_pert
            mask_best_adv2 = mask_succ_adv2 & mask_high_pert
            mask_best_adv3 = mask_succ_adv3 & mask_high_pert
            mask_succ_adv = (mask_succ_adv0 & mask_succ_adv1 & mask_succ_adv2 &  mask_best_adv3)
            mask_best_adv = (mask_best_adv0 & mask_best_adv1 & mask_best_adv3 & mask_best_adv2)

        adv_loss.backward(retain_graph=True)
        adv_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        # 利用CAM attention
        if attention_use:
            # update unsuccessfully attacked samples using class_loss gradient by lr_class*g/||g||
            prj_adv.data[~mask_best_adv] -= adv_lr * prj_CAM_attention * (
                        adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(
                3, 0, 1, 2)[~mask_best_adv]
        else:

            # update unsuccessfully attacked samples using class_loss gradient by lr_class*g/||g||
            prj_adv.data[~mask_best_adv] -= adv_lr * (
                        adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(
                3, 0, 1, 2)[~mask_best_adv]

        # if successfully attacked, perturb image toward color_grad
        col_loss.backward()
        col_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        if attention_use:
            # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
            prj_adv.data[mask_best_adv] -= col_lr * prj_CAM_attention * (
                        col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(
                3, 0, 1, 2)[mask_best_adv]
        else:
            # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
            prj_adv.data[mask_best_adv] -= col_lr * (
                        col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(
                3, 0, 1, 2)[mask_best_adv]

        # keep the best (smallest color loss and successfully attacked ones)
        mask_best_color = (col_loss_batch < col_loss_best).detach().cpu().numpy()
        mask_best = mask_best_color * mask_best_adv
        col_loss_best[mask_best] = col_loss_batch.data[mask_best].clone()

        # make sure successful adversarial attacks first
        prj_adv_best[mask_succ_adv] = prj_adv[mask_succ_adv].clone()
        cam_infer_best[mask_succ_adv] = cam_infer[mask_succ_adv].clone()

        # then try to set the best
        prj_adv_best[mask_best] = prj_adv[mask_best].clone()
        cam_infer_best[mask_best] = cam_infer[mask_best].clone()

        if i % 30 == 0 or i == iters - 1:
            v = 7 if targeted else 0
            print(f'adv_loss = {adv_loss.item():<9.4f} |'
                f'| col_loss = {col_loss.item():<9.4f} | prjl2 = {prjl2.mean() * 255:<9.4f} '
                f'| caml2 = {caml2.mean() * 255:<9.4f} | camdE = {camdE.mean():<9.4f} | p = {torch.log(pp0[v, target_idx0[0]]):.4f} '
                f'| y = p = {torch.log(pp1[v, target_idx1[0]]):.4f} (p = {torch.log(pp2[v, target_idx2[0]]):.4f}) (p = {torch.log(pp3[v, target_idx3[0]]):.4f})')










    # TODO: camdE or caml2 强制提高隐秘性，将camdE大于d_thr的强制提高隐匿性
    while torch.any(camdE > d_thr):
        col_lr = 0.2

        cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)
        raw_score0, p0, idx0 = classifier0(cam_infer, cp_sz)
        raw_score1, p1, idx1 = classifier1(cam_infer, cp_sz)
        raw_score2, p2, idx2 = classifier2(cam_infer, cp_sz)


        pp0 = F.softmax(raw_score0, dim=1)
        pp1 = F.softmax(raw_score1, dim=1)
        pp2 = F.softmax(raw_score2, dim=1)
        pp3 = F.softmax(raw_score2, dim=1)


        # if targeted:
        #     adv_loss = adv_loss + math.exp(-alpha * (raw_score0[torch.arange(num_target), target_idx]).mean()) + math.exp(-alpha * (raw_score1[torch.arange(num_target), target_idx]).mean()) + math.exp(-alpha * (raw_score2[torch.arange(num_target), target_idx]).mean())
        # stealthiness loss: prj adversarial pattern should look like im_gray (not used in SPAA)
        prjl2 = torch.norm(im_gray - prj_adv, dim=1).mean(1).mean(1)  # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch = prjl2_w * prjl2

        # stealthiness loss: cam-captured image should look like cam_scene (L2 loss)
        caml2 = torch.norm(cam_scene_batch - cam_infer, dim=1).mean(1).mean(
            1)  # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch += caml2_w * caml2

        # stealthiness loss: cam-captured image should look like cam_scene (CIE deltaE 2000 loss)
        camdE = ciede2000_diff(rgb2lab_diff(cam_infer, device), rgb2lab_diff(cam_scene_batch, device), device).mean(
            1).mean(1)
        col_loss_batch += camdE_w * camdE

        # average stealthiness (color) losses
        col_loss = col_loss_batch.mean()

        mask_high_pert = (camdE > d_thr).detach().cpu().numpy()

        # if successfully attacked, perturb image toward color_grad
        col_loss.backward()
        col_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        if attention_use:
            # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
            prj_adv.data[mask_high_pert] -= col_lr * prj_CAM_attention * (
                    col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(
                3, 0, 1, 2)[mask_high_pert]
        else:
            # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
            prj_adv.data[mask_high_pert] -= col_lr * (
                    col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(
                3, 0, 1, 2)[mask_high_pert]


        # then try to set the best
        prj_adv_best = prj_adv.clone()
        cam_infer_best = cam_infer.clone()

        if i % 1 == 0 :
            v = 7 if targeted else 0
            print(f'adv_loss = {adv_loss.item():<9.4f} |'
                f'| col_loss = {col_loss.item():<9.4f} | prjl2 = {prjl2.mean() * 255:<9.4f} '
                f'| caml2 = {caml2.mean() * 255:<9.4f} | camdE = {camdE.mean():<9.4f} | p = {torch.log(pp0[v, target_idx0[0]]):.4f} '
                f'| y = p = {torch.log(pp1[v, target_idx1[0]]):.4f} (p = {torch.log(pp2[v, target_idx2[0]]):.4f})')

    # clamp to [0, 1]
    prj_adv_best = torch.clamp(prj_adv_best, 0, 1)  # this inplace opt cannot be used in the for loops above

    # warp the projection from projector space to camera space
    warped_prj = F.grid_sample(prj_adv_best, fine_grid[0,:].unsqueeze(0).expand(prj_adv_best.shape[0], -1, -1, -1),align_corners=True)
    warped_prj_mask = pcnet.module.get_mask().float()[None, None]
    # warped_prj_mask[warped_prj_mask<0.5] = 0.5

    # threshold = torch.tensor(setup_info['prj_brightness']).expand_as(warped_prj).to(warped_prj.device)
    # gray_mask = warped_prj == threshold
    # # 在通道维度上逐像素求与，以检查是否所有通道都等于0.5
    # all_equal = gray_mask.all(dim=0)
    # all_equal = all_equal.unsqueeze(0)
    # # 只保留不是灰色圖的部分
    # warped_prj = warped_prj * ~all_equal
    # mask直射光掩码
    warped_prj = warped_prj * warped_prj_mask

    return cam_infer_best, prj_adv_best, warped_prj

def spaa_universal_dynamic_CAM(pcnet, classifier0, classifier1, classifier2, imagenet_labels,  target_idx, target_idx0, target_idx1, target_idx2, targeted, cam_scene, d_thr, stealth_loss, device, setup_info, fine_grid = None, attention_use = False):
    device = torch.device(device)
    if targeted:
        num_target = len(target_idx)
    else:
        num_target = len(target_idx0)
    cp_sz = setup_info['classifier_crop_sz']

    # camera-captured scene image used for attacks
    cam_scene_batch = cam_scene.expand(num_target, -1, -1, -1).to(device)

    # projector input image
    im_gray = setup_info['prj_brightness'] * torch.ones(num_target, 3, *setup_info['prj_im_sz']).to(
        device)  # TODO: cam_train.mean() may be better?
    prj_adv = im_gray.clone()
    prj_adv.requires_grad = True

    # [debug] we perform batched targeted attacks, and we only show one adversarial target in console, v is the index
    v = 7 if targeted else 0

    # learning rates
    adv_lr = 2  # SPAA Algorithm 1's \beta_1: step size in minimizing adversarial loss
    col_lr = 1  # SPAA Algorithm 1's \beta_2: step size in minimizing stealthiness loss

    # 自建参数
    weight_init = 1/3  # weight for three classifier to linear combination
    weight_init_0 = (1 / 3)
    weight_init_1 = (1 / 3)
    weight_init_2 = (1 / 3)

    # loss weights, lower adv_w or larger color loss weights reduce success rates but make prj_adv more imperceptible
    # SPAA Eq. 9: adv_w=1, prjl2_w=0, caml2_w=1, camdE_w=0, other combinations are shown in supplementary Sec. 3.2. Different stealthiness loss functions
    adv_w = 1  # adversarial loss weights
    adv_f = 0.1 # 反均方误差 adversarial loss weights
    prjl2_w = 0.1 if 'prjl2' in stealth_loss else 0  # projector input image l2 loss weights, SPAA paper prjl2_w=0
    caml2_w = 1 if 'caml2' in stealth_loss else 0  # camera captured image l2 loss weights
    camdE_w = 1 if 'camdE' in stealth_loss else 0  # camera captured image deltaE loss weights

    # SPAA Algorithm 1's pthr: threshold for adversarial confidence
    # lower it when SPAA has a good quality, otherwise increase (can get lower class_loss)
    p_thresh = 0.9  # if too high, the attack may not reach it and the output prj_adv is not perturbed, thus is all gray

    # iterative refine the input, SPAA Algorithm 1's K:number of iterations
    iters = 200  # TODO: improve it, we can early stop when attack requirements are met


    prj_adv_best = prj_adv.clone()
    cam_infer_best = cam_scene.repeat(prj_adv_best.shape[0], 1, 1, 1)
    col_loss_best = 1e6 * torch.ones(prj_adv_best.shape[0]).to(device)

    # CAM_attention_threshold 的制作
    '''
        grad-CAM
        
        model_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    target_layers_vgg16 = [model_vgg16.features]

    model_inception_v3 = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    target_layers_inception_v3 = [model_inception_v3.Mixed_7c]

    model_resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    target_layers_resnet18 = [model_resnet18.layer4]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trans_cam_scene = cam_scene.permute(1, 2, 0).cpu().numpy()
    trans_cam_scene = data_transform(trans_cam_scene)
    trans_cam_scene_batch = torch.unsqueeze(trans_cam_scene, dim=0)

    # 分别获取三种分类器的cam函数
    cam_vgg16 = GradCAM(model=model_vgg16, target_layers=target_layers_vgg16, use_cuda=True)
    cam_inception_v3 = GradCAM(model=model_inception_v3, target_layers=target_layers_inception_v3, use_cuda=True)
    cam_resnet18 = GradCAM(model=model_resnet18, target_layers=target_layers_resnet18, use_cuda=True)

    # 分别获取三种分类器的预测idx
    raw_score0, p0, idx0 = classifier0(cam_scene, cp_sz)
    raw_score1, p1, idx1 = classifier1(cam_scene, cp_sz)
    raw_score2, p2, idx2 = classifier2(cam_scene, cp_sz)
    pred_idx0 = idx0[0][0].tolist()
    pred_idx1 = idx1[0][0].tolist()
    pred_idx2 = idx2[0][0].tolist()

    # 利用raw_score的倒数作为权重
    # 获取当前物体的分别关于三个分类器的CAM
    if classifier0.name == 'vgg16':
        grayscale_cam0 = cam_vgg16(input_tensor=trans_cam_scene_batch, target_category=pred_idx0)  # (1 * 240 *320)
    elif classifier0.name == 'inception_v3':
        grayscale_cam0 = cam_inception_v3(input_tensor=trans_cam_scene_batch,
                                          target_category=pred_idx0)  # (1 * 240 *320)
    else:
        grayscale_cam0 = cam_resnet18(input_tensor=trans_cam_scene_batch, target_category=pred_idx0)  # (1 * 240 *320)

    if classifier1.name == 'vgg16':
        grayscale_cam1 = cam_vgg16(input_tensor=trans_cam_scene_batch, target_category=pred_idx1)  # (1 * 240 *320)
    elif classifier1.name == 'inception_v3':
        grayscale_cam1 = cam_inception_v3(input_tensor=trans_cam_scene_batch,
                                          target_category=pred_idx1)  # (1 * 240 *320)
    else:
        grayscale_cam1 = cam_resnet18(input_tensor=trans_cam_scene_batch, target_category=pred_idx1)  # (1 * 240 *320)

    if classifier2.name == 'vgg16':
        grayscale_cam2 = cam_vgg16(input_tensor=trans_cam_scene_batch, target_category=pred_idx2)  # (1 * 240 *320)
    elif classifier2.name == 'inception_v3':
        grayscale_cam2 = cam_inception_v3(input_tensor=trans_cam_scene_batch,
                                          target_category=pred_idx2)  # (1 * 240 *320)
    else:
        grayscale_cam2 = cam_resnet18(input_tensor=trans_cam_scene_batch, target_category=pred_idx2)  # (1 * 240 *320)

    '''


    # grad-CAMpp
    model_vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    target_layer_vgg16 = model_vgg16.features
    model_vgg16.eval()

    model_inception_v3 = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    target_layer_inception_v3 = model_inception_v3.Mixed_7c
    model_inception_v3.eval()

    model_resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    target_layer_resnet18 = model_resnet18.layer4
    model_resnet18.eval()

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trans_cam_scene = cam_scene.permute(1, 2, 0).cpu().numpy()

    trans_cam_scene = data_transform(trans_cam_scene)
    # reshape 4D tensor (N, C, H, W)
    trans_cam_scene_batch = trans_cam_scene.unsqueeze(0)

    # 分别获取三种分类器的cam函数
    cam_vgg16 = GradCAMpp(model_vgg16, target_layer_vgg16)
    cam_inception_v3 = GradCAMpp(model_inception_v3, target_layer_inception_v3)
    cam_resnet18 = GradCAMpp(model_resnet18, target_layer_resnet18)


    # 利用raw_score的倒数作为权重
    # 获取当前物体的分别关于三个分类器的CAM
    if classifier0.name == 'vgg16':
        grayscale_cam0, _  = cam_vgg16(trans_cam_scene_batch)  # (1 * 3 * 240 *320)
    elif classifier0.name == 'inception_v3':
        grayscale_cam0, _  = cam_inception_v3(trans_cam_scene_batch)  # (1 * 3 * 240 *320)
    else:
        grayscale_cam0, _  = cam_resnet18(trans_cam_scene_batch)  # (1 * 3 * 240 *320)

    if classifier1.name == 'vgg16':
        grayscale_cam1, _  = cam_vgg16(trans_cam_scene_batch)  # (1 * 240 *320)
    elif classifier1.name == 'inception_v3':
        grayscale_cam1, _  = cam_inception_v3(trans_cam_scene_batch)  # (1 * 240 *320)
    else:
        grayscale_cam1, _  = cam_resnet18(trans_cam_scene_batch)  # (1 * 240 *320)

    if classifier2.name == 'vgg16':
        grayscale_cam2, _  = cam_vgg16(trans_cam_scene_batch)  # (1 * 240 *320)
    elif classifier2.name == 'inception_v3':
        grayscale_cam2, _  = cam_inception_v3(trans_cam_scene_batch)  # (1 * 240 *320)
    else:
        grayscale_cam2, _  = cam_resnet18(trans_cam_scene_batch)  # (1 * 240 *320)

    # 0到1
    grayscale_cam0 = resize_to_original(trans_cam_scene_batch, grayscale_cam0)
    # 0到1
    grayscale_cam1 = resize_to_original(trans_cam_scene_batch, grayscale_cam1)
    # 0到1
    grayscale_cam2 = resize_to_original(trans_cam_scene_batch, grayscale_cam2)

    from torchvision.utils import save_image
    # cam_raw_adv_path = join(setup_path, 'cam/raw/adv', attack_ret_folder)
    # save_image(grayscale_cam1,)
    # grayscale_cam0 = torch.stack([torch.tensor(grayscale_cam0[0, :])] * 3, dim=0)  # (3 * 240 * 320)
    # grayscale_cam0 = torch.stack([torch.from_numpy(grayscale_cam0[0, :])] * 3, dim=0).clone().detach()
    CAM_attention0 = grayscale_cam0.expand(num_target, -1, -1, -1).to(device)  # untargeted:(1 * 3 * 240 * 320),targeted:(10 * 3 * 240 * 320)

    # grayscale_cam1 = torch.stack([torch.tensor(grayscale_cam1[0, :])] * 3, dim=0)  # (3 * 240 * 320)
    # grayscale_cam1 = torch.stack([torch.from_numpy(grayscale_cam0[0, :])] * 3, dim=0).clone().detach()
    CAM_attention1 = grayscale_cam1.expand(num_target, -1, -1, -1).to(device)  # untargeted:(1 * 3 * 240 * 320),targeted:(10 * 3 * 240 * 320)

    # grayscale_cam2 = torch.stack([torch.tensor(grayscale_cam2[0, :])] * 3, dim=0)  # (3 * 240 * 320)
    # grayscale_cam2 = torch.stack([torch.from_numpy(grayscale_cam0[0, :])] * 3, dim=0).clone().detach()
    CAM_attention2 = grayscale_cam2.expand(num_target, -1, -1, -1).to(device)  # untargeted:(1 * 3 * 240 * 320),targeted:(10 * 3 * 240 * 320)

    # TODO：inverse_warping CAM
    # prj2cam_grid = pcnet.module.get_warping_grid()
    prj2cam_grid = fine_grid[0, :].unsqueeze(0)  # (1,240,320,2)
    cam2prj_grid = torch.Tensor(invertGrid(prj2cam_grid.detach().cpu(), setup_info['prj_im_sz']))[
        None]  # warps desired cam-captured image to the prj space

    # desired cam-captured image warped to the prj image space
    # 分别生成3张prj_attention
    prj_cmp_init0 = F.grid_sample(CAM_attention0.detach().cpu(),cam2prj_grid.expand(CAM_attention0.shape[0], -1, -1, -1), align_corners=True)
    prj_cmp_init1 = F.grid_sample(CAM_attention1.detach().cpu(),cam2prj_grid.expand(CAM_attention1.shape[0], -1, -1, -1), align_corners=True)
    prj_cmp_init2 = F.grid_sample(CAM_attention2.detach().cpu(),cam2prj_grid.expand(CAM_attention2.shape[0], -1, -1, -1), align_corners=True)
    # desired cam-captured image fov warped to prj image space
    prj_mask = F.grid_sample(pcnet.module.get_mask().float()[None, None].detach().cpu(), cam2prj_grid,
                                align_corners=True)

    prj_CAM_attention0 = (prj_cmp_init0 * prj_mask).to(device)
    # save_image(prj_CAM_attention0, "prj_CAM_attention0.png")
    prj_CAM_attention0 = prj_CAM_attention0[0, :]
    prj_CAM_attention1 = (prj_cmp_init1 * prj_mask).to(device)
    # save_image(prj_CAM_attention1, "prj_CAM_attention1.png")
    prj_CAM_attention1 = prj_CAM_attention1[0, :]
    prj_CAM_attention2 = (prj_cmp_init2 * prj_mask).to(device)
    # save_image(prj_CAM_attention2, "prj_CAM_attention2.png")
    prj_CAM_attention2 = prj_CAM_attention2[0, :]


    # 初始prj_CAM_attention
    prj_CAM_attention = (prj_CAM_attention0 + prj_CAM_attention1 + prj_CAM_attention2) / 3
    # prj_CAM_attention_save = torch.unsqueeze(prj_CAM_attention,dim = 0)
    # save_image(prj_CAM_attention_save, "prj_CAM_attention.png")
    for i in range(0, iters):
        cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)
        raw_score0, p0, idx0 = classifier0(cam_infer, cp_sz)
        raw_score1, p1, idx1 = classifier1(cam_infer, cp_sz)
        raw_score2, p2, idx2 = classifier2(cam_infer, cp_sz)


        pp0 = F.softmax(raw_score0, dim=1)
        pp1 = F.softmax(raw_score1, dim=1)
        pp2 = F.softmax(raw_score2, dim=1)

        ppp0 = F.softmax(raw_score0 / (i // 40 + 1), dim=1)
        ppp1 = F.softmax(raw_score1 / (i // 40 + 1), dim=1)
        ppp2 = F.softmax(raw_score2 / (i // 40 + 1), dim=1)

        adv_loss_untar0 = adv_w * (- torch.log(ppp0[torch.arange(num_target), target_idx])).mean()
        adv_loss_untar1 = adv_w * (- torch.log(ppp1[torch.arange(num_target), target_idx])).mean()
        adv_loss_untar2 = adv_w * (- torch.log(ppp2[torch.arange(num_target), target_idx])).mean()

        adv_loss_tar0 = adv_w * (raw_score0[torch.arange(num_target), target_idx0]).mean()
        adv_loss_tar1 = adv_w * (raw_score1[torch.arange(num_target), target_idx1]).mean()
        adv_loss_tar2 = adv_w * (raw_score2[torch.arange(num_target), target_idx2]).mean()

        # adversarial loss
        if targeted:
            adv_loss = weight_init_0 * adv_loss_untar0 + weight_init_1 * adv_loss_untar1 + weight_init_2 * adv_loss_untar2

        else:
            adv_loss = weight_init_0 * adv_loss_tar0 + weight_init_1 * adv_loss_tar1 + weight_init_2 * adv_loss_tar2

        prjl2 = torch.norm(im_gray - prj_adv, dim=1).mean(1).mean(1)  # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch = prjl2_w * prjl2

        # stealthiness loss: cam-captured image should look like cam_scene (L2 loss)
        caml2 = torch.norm(cam_scene_batch - cam_infer, dim=1).mean(1).mean(
            1)  # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch += caml2_w * caml2

        # stealthiness loss: cam-captured image should look like cam_scene (CIE deltaE 2000 loss)
        camdE = ciede2000_diff(rgb2lab_diff(cam_infer, device), rgb2lab_diff(cam_scene_batch, device), device).mean(
            1).mean(1)
        col_loss_batch += camdE_w * camdE

        # average stealthiness (color) losses
        col_loss = col_loss_batch.mean()

        # mask adversarial confidences that are higher than p_thresh
        mask_high_conf0 = p0[:, 0] > p_thresh
        mask_high_conf1 = p1[:, 0] > p_thresh
        mask_high_conf2 = p2[:, 0] > p_thresh


        mask_high_pert = (camdE > d_thr).detach().cpu().numpy()

        # alternating between the adversarial loss and the stealthiness (color) loss
        if targeted:
            # 判断最大的loss是否是classifer0
            mask_min_conf0 = max_mask(adv_loss_untar0, adv_loss_untar1, adv_loss_untar2)
            # 判断最大的loss是否是classifer1
            mask_min_conf1 = max_mask(adv_loss_untar1, adv_loss_untar0, adv_loss_untar2)
            # 判断最大的loss是否是classifer2
            mask_min_conf2 = max_mask(adv_loss_untar2, adv_loss_untar1, adv_loss_untar0)

            mask_succ_adv0 = idx0[:, 0] != target_idx0
            mask_succ_adv1 = idx1[:, 0] != target_idx1
            mask_succ_adv2 = idx2[:, 0] != target_idx2
            mask_best_adv0 = mask_succ_adv0 & mask_high_pert & mask_high_conf0
            mask_best_adv1 = mask_succ_adv1 & mask_high_pert & mask_high_conf1
            mask_best_adv2 = mask_succ_adv2 & mask_high_pert & mask_high_conf2
            mask_succ_adv = (mask_succ_adv0 & mask_succ_adv1 & mask_succ_adv2)  # 选出三个分类器都能成功攻击的
            mask_best_adv = (mask_best_adv0 & mask_best_adv1 & mask_best_adv2)  # 选出三个分类器都能成功攻击并且有较高置信度的
            # 将每张没有攻击成功的图片按照最低置信度的classifier种类进行分类
            mask_min_adv0 = (~mask_best_adv & mask_min_conf0)
            mask_min_adv1 = (~mask_best_adv & mask_min_conf1)
            mask_min_adv2 = (~mask_best_adv & mask_min_conf2)
        else:
            # alternating between the adversarial loss and the stealthiness (color) loss
            # 判断最大的loss是否是classifer0
            mask_min_conf0 = max_mask(adv_loss_untar0, adv_loss_untar1, adv_loss_untar2)
            # 判断最大的loss是否是classifer1
            mask_min_conf1 = max_mask(adv_loss_untar1, adv_loss_untar0, adv_loss_untar2)
            # 判断最大的loss是否是classifer2
            mask_min_conf2 = max_mask(adv_loss_untar2, adv_loss_untar1, adv_loss_untar0)

            mask_succ_adv0 = idx0[:, 0] != target_idx0
            mask_succ_adv1 = idx1[:, 0] != target_idx1
            mask_succ_adv2 = idx2[:, 0] != target_idx2

            mask_best_adv0 = mask_succ_adv0 & mask_high_pert
            mask_best_adv1 = mask_succ_adv1 & mask_high_pert
            mask_best_adv2 = mask_succ_adv2 & mask_high_pert
            mask_succ_adv = (mask_succ_adv0 & mask_succ_adv1 & mask_succ_adv2)  # 选出三个分类器都能成功攻击的
            mask_best_adv = (mask_best_adv0 & mask_best_adv1 & mask_best_adv2)  # 选出三个分类器都能成功攻击并且有较高置信度的
            # 将每张没有攻击成功的图片按照最高col_loss的classifier种类进行分类
            mask_min_adv0 = (~mask_best_adv & mask_min_conf0)
            mask_min_adv1 = (~mask_best_adv & mask_min_conf1)
            mask_min_adv2 = (~mask_best_adv & mask_min_conf2)



        adv_loss.backward(retain_graph=True)
        adv_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        # 初始化先用平均三个CAM的
        if i == 0:
            # 利用CAM attention
            if attention_use:
                # update unsuccessfully attacked samples using class_loss gradient by lr_class*g/||g||
                prj_adv.data[~mask_best_adv] -= adv_lr * prj_CAM_attention * (
                            adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(
                    3, 0, 1, 2)[~mask_best_adv]
            else:

                # update unsuccessfully attacked samples using class_loss gradient by lr_class*g/||g||
                prj_adv.data[~mask_best_adv] -= adv_lr * (
                            adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(
                    3, 0, 1, 2)[~mask_best_adv]

            # if successfully attacked, perturb image toward color_grad
            col_loss.backward()
            col_grad = prj_adv.grad.clone()
            prj_adv.grad.zero_()

            if attention_use:
                # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
                prj_adv.data[mask_best_adv] -= col_lr * prj_CAM_attention * (
                            col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(
                    3, 0, 1, 2)[mask_best_adv]
            else:
                # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
                prj_adv.data[mask_best_adv] -= col_lr * (
                            col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(
                    3, 0, 1, 2)[mask_best_adv]

        # 根据生成的infer动态调整三个CAM的比重，将置信度最低的classifier的CAM比重调高
        else:
            prj_CAM_mask0 = (prj_CAM_attention0 * 3 + prj_CAM_attention1 + prj_CAM_attention2) / 4
            prj_CAM_mask1 = (prj_CAM_attention0 + prj_CAM_attention1 * 3 + prj_CAM_attention2) / 4
            prj_CAM_mask2 = (prj_CAM_attention0 + prj_CAM_attention1 + prj_CAM_attention2 * 3) / 4

            # 利用CAM attention
            if attention_use:
                # update unsuccessfully attacked samples using class_loss gradient by lr_class*g/||g||
                prj_adv.data[mask_min_adv0] -= adv_lr * prj_CAM_mask0 * (
                        adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(
                    3, 0, 1, 2)[mask_min_adv0]
                prj_adv.data[mask_min_adv1] -= adv_lr * prj_CAM_mask1 * (
                        adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(
                    3, 0, 1, 2)[mask_min_adv1]
                prj_adv.data[mask_min_adv2] -= adv_lr * prj_CAM_mask2 * (
                        adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(
                    3, 0, 1, 2)[mask_min_adv2]
            else:

                # update unsuccessfully attacked samples using class_loss gradient by lr_class*g/||g||
                prj_adv.data[~mask_best_adv] -= adv_lr * (
                        adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(
                    3, 0, 1, 2)[~mask_best_adv]

            # if successfully attacked, perturb image toward color_grad
            col_loss.backward()
            col_grad = prj_adv.grad.clone()
            prj_adv.grad.zero_()

            if attention_use:
                # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
                prj_adv.data[mask_best_adv] -= col_lr * prj_CAM_attention * (col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(3, 0, 1, 2)[mask_best_adv]

            else:
                # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
                prj_adv.data[mask_best_adv] -= col_lr * (
                        col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(
                    3, 0, 1, 2)[mask_best_adv]

        # keep the best (smallest color loss and successfully attacked ones)
        mask_best_color = (col_loss_batch < col_loss_best).detach().cpu().numpy()
        mask_best = mask_best_color * mask_best_adv
        col_loss_best[mask_best] = col_loss_batch.data[mask_best].clone()

        # make sure successful adversarial attacks first
        prj_adv_best[mask_succ_adv] = prj_adv[mask_succ_adv].clone()
        cam_infer_best[mask_succ_adv] = cam_infer[mask_succ_adv].clone()

        # then try to set the best
        prj_adv_best[mask_best] = prj_adv[mask_best].clone()
        cam_infer_best[mask_best] = cam_infer[mask_best].clone()

        if i % 30 == 0 or i == iters - 1:
            v = 7 if targeted else 0
            print(f'adv_loss = {adv_loss.item():<9.4f} |'
                f'| col_loss = {col_loss.item():<9.4f} | prjl2 = {prjl2.mean() * 255:<9.4f} '
                f'| caml2 = {caml2.mean() * 255:<9.4f} | camdE = {camdE.mean():<9.4f} | p = {p0[v, 0]:.4f}'
                f'| y = {idx0[v, 0]:3d} ({imagenet_labels[idx0[v, 0].item()]})'
                f'| p = {torch.log(pp0[v, target_idx0[0]]):.4f} | y = p = {torch.log(pp1[v, target_idx1[0]]):.4f} (p = {torch.log(pp2[v, target_idx2[0]]):.4f})'
            )


    # TODO: camdE or caml2 强制提高隐秘性，将camdE大于d_thr的强制提高隐匿性
    while torch.any(camdE > d_thr):
        # while camdE.mean() > d_thr:
        col_lr = 0.2
        # TODO: cam_scene_batch * mask
        cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)

        # stealthiness loss: prj adversarial pattern should look like im_gray (not used in SPAA)
        prjl2 = torch.norm(im_gray - prj_adv, dim=1).mean(1).mean(1)  # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch = prjl2_w * prjl2

        # stealthiness loss: cam-captured image should look like cam_scene (L2 loss)
        caml2 = torch.norm(cam_scene_batch - cam_infer, dim=1).mean(1).mean(
            1)  # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch += caml2_w * caml2

        # stealthiness loss: cam-captured image should look like cam_scene (CIE deltaE 2000 loss)
        camdE = ciede2000_diff(rgb2lab_diff(cam_infer, device), rgb2lab_diff(cam_scene_batch, device), device).mean(
            1).mean(1)
        col_loss_batch += camdE_w * camdE

        # average stealthiness (color) losses
        col_loss = col_loss_batch.mean()

        # TODO:d_threshold

        # mask_high_pert = (caml2 * 255 > d_thr).detach().cpu().numpy()
        mask_high_pert = (camdE > d_thr).detach().cpu().numpy()

        # if successfully attacked, perturb image toward color_grad
        col_loss.backward()
        col_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        if attention_use:
            # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
            prj_adv.data[mask_high_pert] -= col_lr * (
                    col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1),
                                                              dim=1)).permute(3, 0, 1, 2)[mask_high_pert]
        else:
            # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
            prj_adv.data[mask_high_pert] -= col_lr * (
                    col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1),
                                                              dim=1)).permute(3, 0, 1, 2)[mask_high_pert]

        # then try to set the best
        prj_adv_best = prj_adv.clone()
        cam_infer_best = cam_infer.clone()

        if i % 1 == 0:
                v = 7 if targeted else 0
                if targeted:
                    print(f'adv_loss = {adv_loss.item():<9.4f} '
                        f'| col_loss = {col_loss.item():<9.4f} | prjl2 = {prjl2.mean() * 255:<9.4f} '
                        f'| caml2 = {caml2.mean() * 255:<9.4f} | camdE = {camdE[0]:<9.4f} |camdE = {camdE[1]:<9.4f} |camdE = {camdE[2]:<9.4f} |camdE = {camdE[3]:<9.4f} |camdE = {camdE[4]:<9.4f} |camdE = {camdE[5]:<9.4f} |camdE = {camdE[6]:<9.4f} |camdE = {camdE[7]:<9.4f} |camdE = {camdE[8]:<9.4f}|camdE = {camdE[9]:<9.4f}| p = {p0[v, 0]:.4f}'
                        f'| y = {idx0[v, 0]:3d} ({imagenet_labels[idx0[v, 0].item()]})'
                        f'| p = {torch.log(pp0[v, target_idx0[0]]):.4f} | y = p = {torch.log(pp1[v, target_idx1[0]]):.4f} (p = {torch.log(pp2[v, target_idx2[0]]):.4f})'
                    )
                else:
                    print(
                        f'adv_loss = {adv_loss.item():<9.4f} | col_loss = {col_loss.item():<9.4f} | prjl2 = {prjl2.mean() * 255:<9.4f} '
                        f'| caml2 = {caml2.mean() * 255:<9.4f} | camdE = {camdE.mean():<9.4f} | p = {p0[v, 0]:.4f} '
                     f'| y = {idx0[v, 0]:3d} ({imagenet_labels[idx0[v, 0].item()]})'
                        f'| p = {torch.log(pp0[v, target_idx0[0]]):.4f} | y = p = {torch.log(pp1[v, target_idx1[0]]):.4f} (p = {torch.log(pp2[v, target_idx2[0]]):.4f})'
                    )


    # clamp to [0, 1]
    prj_adv_best = torch.clamp(prj_adv_best, 0, 1)  # this inplace opt cannot be used in the for loops above


    # warp the projection from projector space to camera space
    warped_prj = F.grid_sample(prj_adv_best, fine_grid[0,:].unsqueeze(0).expand(prj_adv_best.shape[0], -1, -1, -1), align_corners=True)
    warped_prj_mask = pcnet.module.get_mask().float()[None, None]
    # warped_prj_mask[warped_prj_mask<0.5] = 0.5

    # threshold = torch.tensor(setup_info['prj_brightness']).expand_as(warped_prj).to(warped_prj.device)
    # gray_mask = warped_prj == threshold
    # # 在通道维度上逐像素求与，以检查是否所有通道都等于0.5
    # all_equal = gray_mask.all(dim=0)
    # all_equal = all_equal.unsqueeze(0)
    # # 只保留不是灰色圖的部分
    # warped_prj = warped_prj * ~all_equal


    # mask直射光掩码
    warped_prj = warped_prj * warped_prj_mask
    return cam_infer_best, prj_adv_best, warped_prj



def perc_al_compennet_pp(compennet_pp, classifier, imgnet_labels, target_idx, targeted, cam_scene, d_thr, device,
                         setup_info):
    # PerC-AL+CompenNet++. A two step based attacker.
    device = torch.device(device)
    num_target = len(target_idx)
    cp_sz = setup_info['classifier_crop_sz']

    # camera-captured scene image used for attacks
    cam_scene_batch = cam_scene.expand(num_target, -1, -1, -1)

    # 1. Digital attack using PerC-AL;

    data_root = abspath(join(os.getcwd(), '../../data'))

    file_perturbation = os.path.join(data_root, 'universal.npy')

    cam_infer_best_temp = np.load(file_perturbation).transpose(0, 3, 1, 2) / 256 + 0.5

    cam_infer_best = torch.tensor(cam_infer_best_temp, dtype=torch.float32)

    # 2. Use CompenNet++ to compensate digital adversarial images

    prj_adv_best = compennet_pp(cam_infer_best, cam_scene_batch)

    return cam_infer_best, prj_adv_best



def summarize_single_universal_attacker(attacker_name, data_root, setup_list, device='cuda', device_ids=[0], pose = 'original'):
    # given the attacker_name and setup_list, summarize all attacks, create stats.txt/xls and montages
    # assert attacker_name in ['SPAA', 'PerC-AL+CompenNet++', 'One-pixel_DE'], f'{attacker_name} not supported!'
    attacker_name_list = attacker_name
    # set PyTorch device to GPU
    device = torch.device(device)




    # ImageNet and targeted attack labels
    imagenet_labels = load_imagenet_labels(join(data_root, 'imagenet1000_clsidx_to_labels.txt'))
    target_labels   = load_imagenet_labels(join(data_root, 'imagenet10_clsidx_to_labels.txt'))  # targeted attack labels

    # we perform n = 10 targeted attacks and 1 untargeted attack
    n = 10  # modify n for debug
    target_labels = dict(itertools.islice(target_labels.items(), n))
    target_idx    = list(target_labels.keys())

    # attack results table
    phase   = ['Valid', 'prj', 'infer', 'real']
    metrics = ['PSNR', 'RMSE', 'SSIM', 'L2', 'Linf', 'dE']
    columns = ['Setup', 'Attacker', 'Stealth_loss', 'd_thr', 'Classifier', 'T.top-1_infer', 'T.top-5_infer', 'T.top-1_real', 'T.top-5_real',
               'U.top-1_infer', 'U.top-1_real'] + [phase[0] + '_' + y for y in metrics] +\
              ['T.' + x + '_' + y for x in phase[1:] for y in metrics] + ['U.' + x + '_' + y for x in phase[1:] for y in metrics] + \
              ['All.' + x + '_' + y for x in phase[1:] for y in metrics]

    # stealth_losses = ['caml2', 'camdE', 'camdE_caml2', 'camdE_caml2_prjl2', '-']
    stealth_losses   = ['camdE_caml2']
    d_threshes       = [2, 3, 4, 5]
    # classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
    classifier_names = ['inception_v3', 'resnet18', 'vgg16']
    for setup_name in setup_list:
        for attacker_name in attacker_name_list:
            dl_based = attacker_name in ['original SPAA', 'CAM', 'USPAA', 'all']
            attacker_cfg_str, model_cfg_str = to_attacker_cfg_str(attacker_name)
            setup_path = join(data_root, 'setups', setup_name)
            print(f'\nCalculating stats of [{attacker_name}] on [{setup_path}]')
            table = pd.DataFrame(columns=columns)

            # load setup info and images
            setup_info = load_setup_info(setup_path)
            cp_sz = setup_info['classifier_crop_sz']

            # projector illumination
            im_gray = setup_info['prj_brightness'] * torch.ones(1, 3, *setup_info['prj_im_sz']).to(device)

            # load training and validation data
            cam_scene  = ut.torch_imread(join(setup_path, 'cam/raw/ref/img_0002.png')).to(device)

            # calc validation metrics
            im_infer  = cc(ut.torch_imread_mt(join(setup_path, 'cam/infer/test', model_cfg_str)), cp_sz).to(device)
            im_gt     = cc(ut.torch_imread_mt(join(setup_path, 'cam/raw/test')), cp_sz).to(device)
            valid_ret = calc_img_dists(im_infer, im_gt)

            classifier_train = 'classifier_all'

            for stealth_loss in stealth_losses:
                for d_thr in d_threshes:
                    attack_ret_folder = join(attacker_cfg_str, stealth_loss, str(d_thr), classifier_train)
                    prj_adv_path = join(setup_path, 'prj/adv', attack_ret_folder)
                    cam_infer_path = join(setup_path, 'cam/infer/adv', attack_ret_folder)
                    cam_real_path = join(setup_path, 'cam/raw/adv', attack_ret_folder)
                    warped_prj_adv_path = join(setup_path, 'prj/warped_adv', attack_ret_folder)

                    prj_adv = ut.torch_imread_mt(prj_adv_path).to(device)
                    cam_real = ut.torch_imread_mt(cam_real_path).to(device)
                    cam_infer = ut.torch_imread_mt(cam_infer_path).to(device) if dl_based else cam_real
                    warped_prj_adv = ut.torch_imread_mt(warped_prj_adv_path).to(device)

                    ret = {}  # classification result dict

                    for classifier_name in classifier_names:
                        if classifier_name == 'vit_b_16':
                            print('----------------vit_b_16-----------------')


                        # check whether all images are captured for results summary
                        dirs_to_check = [prj_adv_path, cam_real_path]
                        skip = False
                        if dl_based:
                            dirs_to_check.append(cam_infer_path)

                        for img_dir in dirs_to_check:
                            if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
                                print(f'No such folder/images: {img_dir}\n'
                                      f'Maybe [{attacker_name}] has no [{join(stealth_loss, str(d_thr), classifier_train)}] attack cfg, or you forget to project and capture.\n')
                                skip = True
                                break
                        if skip:
                            break

                        with torch.no_grad():
                            classifier = Classifier(classifier_name, device, device_ids, fix_params=True, sort_results=True)
                            ret['scene' + classifier_name] = classifier(cam_scene, cp_sz)
                            ret['infer' + classifier_name] = classifier(cam_infer, cp_sz)
                            ret['real' + classifier_name] = classifier(cam_real, cp_sz)


                        # Targeted: top-1 and top-5 success rate
                        # inferred attacks
                        t1_infer = np.count_nonzero(ret['infer' + classifier_name][2][:n, 0] == target_idx) / n
                        t5_infer = np.count_nonzero([target_idx[i] in ret['infer'+ classifier_name][2][i, :5] for i in range(n)]) / n

                        # real camera-captured attacks
                        t1_real = np.count_nonzero(ret['real'+ classifier_name][2][:n, 0] == target_idx) / n
                        t5_real = np.count_nonzero([target_idx[i] in ret['real'+ classifier_name][2][i, :5] for i in range(n)]) / n

                        # Untargeted: top-1 success rate
                        true_idx = ret['scene'+ classifier_name][2][0, 0]
                        t1_untar_infer = np.count_nonzero(ret['infer'+ classifier_name][2][n, 0] != true_idx)
                        t1_untar_real = np.count_nonzero(ret['real'+ classifier_name][2][n, 0] != true_idx)

                        # calc image similarity metrics
                        table.loc[len(table)] = [
                            setup_name, attacker_cfg_str, stealth_loss, d_thr, classifier_name, t1_infer, t5_infer, t1_real,
                            t5_real, t1_untar_infer, t1_untar_real,
                            # model infer vs GT on the validation data
                            *valid_ret,

                            # targeted [0, n-1]
                            *calc_img_dists(prj_adv[:n], im_gray.expand_as(prj_adv[:n])),  # prj adv
                            *calc_img_dists(cc(cam_infer[:n], cp_sz),
                                            cc(cam_scene, cp_sz).expand_as(cc(cam_infer[:n], cp_sz))),  # cam infer
                            *calc_img_dists(cc(cam_real[:n], cp_sz),
                                            cc(cam_scene, cp_sz).expand_as(cc(cam_real[:n], cp_sz))),
                            # cam real (the last four columns of SPAA paper Table 1 and supplementary Table 4)

                            # untargeted [n]
                            *calc_img_dists(prj_adv[n, None], im_gray.expand_as(prj_adv[n, None])),  # prj adv
                            *calc_img_dists(cc(cam_infer[n, None], cp_sz),
                                            cc(cam_scene, cp_sz).expand_as(cc(cam_infer[n, None], cp_sz))),  # cam infer
                            *calc_img_dists(cc(cam_real[n, None], cp_sz),
                                            cc(cam_scene, cp_sz).expand_as(cc(cam_real[n, None], cp_sz))),  # cam real

                            # both targeted and untargeted [0, n].
                            *calc_img_dists(prj_adv, im_gray.expand_as(prj_adv)),  # prj adv
                            *calc_img_dists(cc(cam_infer, cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_infer, cp_sz))),
                            # cam infer
                            *calc_img_dists(cc(cam_real, cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_real, cp_sz)))
                            # cam real
                            # !!! The original SPAA paper showed targeted [0, n-1] stealthiness metrics, and missed untargeted [n] by mistake (although it does not change the paper's conclusion).
                            # Future works use the mean of both targeted and untargeted as the last four columns of main paper Table 1 and supplementary Table 4.
                        ]

                    # create the result montage as shown in SPAA main paper Figs. 4-5 and supplementary
                    im_montage = []

                    if pose == 'original':
                        for t in range(n + 1):  # when t = n + 1, it is the untargeted attack result
                            im_montage.append(
                                attack_results(ret, t, imagenet_labels, im_gray, prj_adv, warped_prj_adv, cam_scene,
                                               cam_infer, cam_real, setup_info['prj_im_sz'], cp_sz,
                                               classifier_names))
                    else:
                        for t in range(n + 1):  # when t = n + 1, it is the untargeted attack result
                            im_montage.append(attack_results_change_pose(ret, t, imagenet_labels, im_gray, prj_adv,
                                                                         warped_prj_adv, cam_scene, cam_infer,
                                                                         cam_real, setup_info['prj_im_sz'], cp_sz,
                                                                         classifier_names))

                    # [debug] show montage in visdom
                    # vfs(torch.stack(im_montage, 0), ncol=1, title=attacker_cfg_str + '_' + stealth_loss + '_' + str(d_thr) + '_' + classifier_name)

                    # save montage
                    montage_path = join(setup_path, 'ret', attack_ret_folder)
                    ut.save_imgs(torch.stack(im_montage, 0), montage_path)

            # print results
            print(f'\n-------------------- [{attacker_name}] results on [{setup_name}] --------------------')
            print(table.to_string(index=False, float_format='%.4f'))
            print('-------------------------------------- End of result table ---------------------------\n')
            # print(table.filter(regex = 'top-[0-9]_', axis = 1).to_string(index = False, float_format = '%.2f'))  # columns that only contain success rates
            # print(table.filter(regex = '_L2'    , axis = 1).to_string(index = False, float_format = '%.4f'))  # columns that only contain L2
            # print(table.filter(regex = '_dE'    , axis = 1).to_string(index = False, float_format = '%.4f'))  # columns that only contain dE

            # save stats to files
            ret_path = join(setup_path, 'ret', attacker_cfg_str)
            if not os.path.exists(ret_path): os.makedirs(ret_path)
            table.to_csv(join(ret_path, 'stats.txt'), index=False, float_format='%.4f', sep='\t')
            table.to_excel(join(ret_path, 'stats.xlsx'), float_format='%.4f', index=False)
    return table


def summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs=False):
    """
    given attacker_names and setup_list, summarize all attacks
    :param attacker_names:
    :param data_root:
    :param setup_list:
    :param recreate_stats_and_imgs: when False, only gather all existing stats.txt of all setups and create a pivot table [setup/pivot_table_all.xlsx]
    :return:
    """
    table = []

    for setup_name in tqdm(setup_list):
        setup_path = join(data_root, 'setups', setup_name)
        for attacker_name in attacker_names:
            attacker_cfg_str = to_attacker_cfg_str(attacker_name)[0]
            ret_path = join(setup_path, 'ret', attacker_cfg_str)
            print(f'Gathering stats of {ret_path}')

            if 'original' in setup_name:
                pose = 'original'
            else:
                pose = 'changed'
            # (time-consuming) recreate stats.txt, stats.xls and images in [ret] folder for each setup
            if attacker_name in ['original SPAA', 'CAM']:
                if recreate_stats_and_imgs:
                    summarize_single_attacker_for_UAP_task(attacker_name=[attacker_name], data_root=data_root, setup_list=[setup_name], device='cuda', device_ids=[0],pose=pose)
                table.append(pd.read_csv(join(ret_path, 'universal_stats.txt'), index_col=None, header=0, sep='\t'))
            else:
                if recreate_stats_and_imgs:
                    summarize_single_universal_attacker(attacker_name=[attacker_name], data_root=data_root, setup_list=[setup_name], device='cuda', device_ids=[0],pose=pose)
                table.append(pd.read_csv(join(ret_path, 'stats.txt'), index_col=None, header=0, sep='\t'))

    table = pd.concat(table, axis=0, ignore_index=True)

    # pivot_table is supplementary Table 2, and SPAA paper's Table 1 is its subset
    pivot_table = pd.pivot_table(table,
                                 # values=['T.top-1_real', 'T.top-5_real', 'U.top-1_real', 'T.real_L2', 'T.real_Linf', 'T.real_dE', 'T.real_SSIM'],
                                 values=['T.top-1_real', 'T.top-5_real', 'U.top-1_real', 'U.real_L2', 'U.real_Linf',
                                         'U.real_dE', 'U.real_SSIM', 'All.real_L2', 'All.real_Linf', 'All.real_dE',
                                         'All.real_SSIM'],
                                 index=['Attacker', 'd_thr', 'Stealth_loss', 'Classifier'], aggfunc=np.mean)
    pivot_table = pivot_table.sort_index(level=[0, 1], ascending=[False, True])  # to match SPAA Table order

    # save tables
    table.to_csv(join(data_root, 'setups/stats_all.txt'), index=False, float_format='%.4f', sep='\t')
    table.to_excel(join(data_root, 'setups/stats_all.xlsx'), float_format='%.4f', index=False)
    pivot_table.to_excel(join(data_root, 'setups/pivot_table_all.xlsx'), float_format='%.4f', index=True)

    return table, pivot_table
