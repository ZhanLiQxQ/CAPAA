"""
Useful functions for projector-based adversarial attack
"""
from os.path import join
import cv2 as cv
import pandas as pd
from torchvision import models
from torchvision import transforms
from torchvision.utils import make_grid
from train_network import load_setup_info, get_model_train_cfg
from img_proc import resize, insert_text, expand_boarder, expand_4d, center_crop as cc, invertGrid
import utils as ut
from utils import calc_img_dists
from differential_color_functions import rgb2lab_diff, ciede2000_diff,deltaE

import itertools
from classifier import Classifier, load_imagenet_labels
import torch.nn.functional as F

import os
import numpy as np
import torch
from grad_cam import GradCAM,GradCAMpp,resize_to_original
from vit_model import vit_base_patch16_224



def to_attacker_cfg_str(attacker_name):

    model_cfg        = get_model_train_cfg(model_list=['PCNet'], single=True)
    model_cfg_str    = f'{model_cfg.model_name}_{model_cfg.loss}_{model_cfg.num_train}_{model_cfg.batch_size}_{model_cfg.max_iters}'
    attacker_cfg_str = f'{attacker_name}_{model_cfg_str}'
    return attacker_cfg_str, model_cfg_str


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


def spaa_and_classifier_specific_capaa(pcnet, classifier, imagenet_labels, target_idx, targeted, cam_scene, d_thr, stealth_loss, device, setup_info, fine_grid = None, attention_use = False):
    """
    Stealthy Projector-based Adversarial Attack (SPAA) and CAPAA (Classifier-specific)
    :param pcnet:
    :param classifier:
    :param imagenet_labels:
    :param target_idx:
    :param targeted:
    :param cam_scene:
    :param d_thr: SPAA Algorithm 1's d_thr: threshold for L2 perturbation size
    :param stealth_loss:
    :param device:
    :param setup_info:
    :return:
    """
    device = torch.device(device)
    num_target = len(target_idx)
    cp_sz = setup_info['classifier_crop_sz']

    # camera-captured scene image used for attacks
    cam_scene_batch = cam_scene.expand(num_target, -1, -1, -1).to(device)# (3,240,320) -> (1,3,240,320)

    # projector input image
    im_gray = setup_info['prj_brightness'] * torch.ones(num_target, 3, *setup_info['prj_im_sz']).to(device)  # TODO: cam_train.mean() may be better?
    prj_adv = im_gray.clone()
    prj_adv.requires_grad = True

    # [debug] we perform batched targeted attacks, and we only show one adversarial target in console, v is the index
    v = 7 if targeted else 0

    # learning rates
    adv_lr = 2  # CAPAA Algorithm 1's \beta_1: step size in minimizing adversarial loss
    col_lr = 1  # CAPAA Algorithm 1's \beta_2: step size in minimizing stealthiness loss

    # loss weights, lower adv_w or larger color loss weights reduce success rates but make prj_adv more imperceptible
    adv_w   = 1                                # adversarial loss weights
    prjl2_w = 0.1 if 'prjl2' in stealth_loss else 0    # projector input image l2 loss weights, CAPAA paper prjl2_w=0
    caml2_w = 1   if 'caml2' in stealth_loss else 0    # camera captured image l2 loss weights
    camdE_w = 1   if 'camdE' in stealth_loss else 0    # camera captured image deltaE loss weights

    # CAPAA Algorithm 1's pthr: threshold for adversarial confidence
    # lower it when CAPAA has a good quality, otherwise increase (can get lower class_loss)
    p_thresh = 0.9  # if too high, the attack may not reach it and the output prj_adv is not perturbed, thus is all gray

    # iterative refine the input, CAPAA Algorithm 1's K:number of iterations
    iters = 200  # TODO: improve it, we can early stop when attack requirements are met

    prj_adv_best   = prj_adv.clone()
    cam_infer_best = cam_scene.repeat(prj_adv_best.shape[0], 1, 1, 1) # (3,240,320) -> (batch,3,240,320)
    col_loss_best  = 1e6 * torch.ones(prj_adv_best.shape[0]).to(device)


    if classifier.name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        target_layer = model.features

    if classifier.name == 'inception_v3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        target_layer = model.Mixed_7c

    if classifier.name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        target_layer = model.layer4

    if classifier.name == 'vit_b_16':
        # model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        model = vit_base_patch16_224()
        model.eval()
        model.load_state_dict(torch.load("./vit_base_patch16_224.pth", map_location=device))

    model.eval()

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trans_cam_scene = cam_scene.permute(1, 2, 0).cpu().numpy()
    trans_cam_scene = data_transform(trans_cam_scene)
    trans_cam_scene_batch = torch.unsqueeze(trans_cam_scene, dim=0)



    if classifier.name == 'vit_b_16':
        cam_vit = GradCAM(model=model,
                          target_layers=[model.blocks[-1].norm1],
                          use_cuda=torch.cuda.is_available(),
                          reshape_transform=ReshapeTransform(model))
        # for gradcam for vits, only resize to (224,224)
        resized_vit_trans_cam_scene_batch = resize(trans_cam_scene_batch, size=(224, 224))  # 1* 3* 224 * 224
        grayscale_cam = cam_vit(resized_vit_trans_cam_scene_batch)  # (1,224,224)
        grayscale_cam = torch.from_numpy(grayscale_cam).unsqueeze(0)  # tensor (1, 224, 224)
    else:
        gradcampp = GradCAMpp(model, target_layer)
        grayscale_cam, idx = gradcampp(trans_cam_scene_batch)

    # 0到1

    grayscale_cam = resize_to_original(trans_cam_scene_batch, grayscale_cam)
    # grayscale_cam = torch.stack([torch.from_numpy(grayscale_cam[0, :])] * 3, dim=0).clone().detach()

    CAM_attention = grayscale_cam.expand(num_target, -1, -1, -1).to(device)  # untargeted:(1 * 3 * 240 * 320),targeted:(10 * 3 * 240 * 320)

    # inverse_warping CAM
    prj2cam_grid = fine_grid[0,:].unsqueeze(0) # (1,240,320,2)
    cam2prj_grid = torch.Tensor(invertGrid(prj2cam_grid.detach().cpu(), setup_info['prj_im_sz']))[None]  # warps desired cam-captured image to the prj space

    # desired cam-captured image warped to the prj image space
    prj_cmp_init = F.grid_sample(CAM_attention.detach().cpu(), cam2prj_grid.expand(CAM_attention.shape[0], -1, -1, -1), align_corners=True)

    # desired cam-captured image fov warped to prj image space
    prj_mask = F.grid_sample(pcnet.module.get_mask().float()[None, None].detach().cpu(), cam2prj_grid, align_corners=True)

    prj_CAM_attention = (prj_cmp_init * prj_mask).to(device)
    prj_CAM_attention = prj_CAM_attention[0, :]

    for i in range(0, iters):

        cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)

        raw_score, p, idx = classifier(cam_infer, cp_sz)

        # adversarial loss
        if targeted:
            adv_loss = adv_w * (-raw_score[torch.arange(num_target), target_idx]).mean()
        else:
            adv_loss = adv_w * (raw_score[torch.arange(num_target) , target_idx]).mean()

        # stealthiness loss: prj adversarial pattern should look like im_gray (not used in CAPAA)
        prjl2           = torch.norm(im_gray - prj_adv, dim=1).mean(1).mean(1)            # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch  = prjl2_w * prjl2

        # stealthiness loss: cam-captured image should look like cam_scene (L2 loss)
        caml2           = torch.norm(cam_scene_batch - cam_infer, dim=1).mean(1).mean(1)  # mean L2 norm, consistent with Zhao_CVPR_20
        col_loss_batch += caml2_w * caml2

        # stealthiness loss: cam-captured image should look like cam_scene (CIE deltaE 2000 loss)
        camdE           = ciede2000_diff(rgb2lab_diff(cam_infer, device), rgb2lab_diff(cam_scene_batch, device), device).mean(1).mean(1)
        col_loss_batch += camdE_w * camdE

        # average stealthiness (color) losses
        col_loss        = col_loss_batch.mean()

        # mask adversarial confidences that are higher than p_thresh
        mask_high_conf = p[:, 0].detach().cpu().numpy() > p_thresh

        # mask_high_pert = (caml2 * 255 > d_thr).detach().cpu().numpy()
        mask_high_pert = (camdE > d_thr).detach().cpu().numpy()

        # alternating between the adversarial loss and the stealthiness (color) loss
        if targeted:
            mask_succ_adv = idx[:, 0] == target_idx
            mask_best_adv = mask_succ_adv & mask_high_conf & mask_high_pert
        else:
            mask_succ_adv = idx[:, 0] != target_idx
            mask_best_adv = mask_succ_adv & mask_high_pert

        # if not successfully attacked, perturb prj_adv toward class_grad
        adv_loss.backward(retain_graph=True)
        adv_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        if attention_use:
            # update unsuccessfully attacked samples using class_loss gradient by lr_class*g/||g||
            prj_adv.data[~mask_best_adv] -= adv_lr * prj_CAM_attention * (adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(3, 0, 1, 2)[~mask_best_adv]
        else:

            # update unsuccessfully attacked samples using class_loss gradient by lr_class*g/||g||
            prj_adv.data[~mask_best_adv] -= adv_lr * (adv_grad.permute(1, 2, 3, 0) / torch.norm(adv_grad.view(adv_grad.shape[0], -1), dim=1)).permute(3, 0, 1, 2)[~mask_best_adv]

        # if successfully attacked, perturb image toward color_grad
        col_loss.backward()
        col_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        if attention_use:
            # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
            prj_adv.data[mask_best_adv] -= col_lr * prj_CAM_attention * (col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(3, 0, 1, 2)[mask_best_adv]
        else:
            # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
            prj_adv.data[mask_best_adv] -= col_lr * (col_grad.permute(1, 2, 3, 0) / torch.norm(col_grad.view(col_grad.shape[0], -1), dim=1)).permute(3, 0, 1, 2)[mask_best_adv]

        # keep the best (smallest color loss and successfully attacked ones)
        mask_best_color               = (col_loss_batch < col_loss_best).detach().cpu().numpy()
        mask_best                     = mask_best_color * mask_best_adv
        col_loss_best[mask_best]      = col_loss_batch.data[mask_best].clone()

        # make sure successful adversarial attacks first
        prj_adv_best[mask_succ_adv]   = prj_adv[mask_succ_adv].clone()
        cam_infer_best[mask_succ_adv] = cam_infer[mask_succ_adv].clone()

        # then try to set the best
        prj_adv_best[mask_best]       = prj_adv[mask_best].clone()
        cam_infer_best[mask_best]     = cam_infer[mask_best].clone()

        if i % 30 == 0 or i == iters - 1:
            # lr *= 0.2 # drop lr
            print(f'adv_loss = {adv_loss.item():<9.4f} | col_loss = {col_loss.item():<9.4f} | prjl2 = {prjl2.mean() * 255:<9.4f} '
                  f'| caml2 = {caml2.mean() * 255:<9.4f} | camdE = {camdE.mean():<9.4f} | p = {p[v, 0]:.4f} '
                  f'| y = {idx[v, 0]:3d} ({imagenet_labels[idx[v, 0].item()]})'
                 )
            # n += 1
    # clamp to [0, 1]

    #  For projection whose camdE is greater than d_thr, its stealthiness will be forcibly enhanced.
    while torch.any(camdE > d_thr):
    # while camdE.mean() > d_thr:
        col_lr = 0.2
        cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)

        raw_score, p, idx = classifier(cam_infer, cp_sz)

        # stealthiness loss: prj adversarial pattern should look like im_gray (not used in CAPAA)
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

        if i % 30 == 0:
            # lr *= 0.2 # drop lr
            print(
                f'adv_loss = {adv_loss.item():<9.4f} | col_loss = {col_loss.item():<9.4f} | prjl2 = {prjl2.mean() * 255:<9.4f} '
                f'| caml2 = {caml2.mean() * 255:<9.4f} | camdE = {camdE.mean():<9.4f} | p = {p[v, 0]:.4f} '
                f'| y = {idx[v, 0]:3d} ({imagenet_labels[idx[v, 0].item()]})'
            )



    prj_adv_best = torch.clamp(prj_adv_best, 0, 1)  # this inplace opt cannot be used in the for loops above
    # warp the projection from projector space to camera space
    warped_prj = F.grid_sample(prj_adv_best, fine_grid[0,:].unsqueeze(0).expand(prj_adv_best.shape[0], -1, -1, -1), align_corners=True)
    warped_prj_mask = pcnet.module.get_mask().float()[None, None]

    warped_prj = warped_prj * warped_prj_mask
    return cam_infer_best, prj_adv_best, warped_prj



def attack_results_change_pose(ret, t, imgnet_labels, im_gray, prj_adv, warped_prj, cam_scene, cam_infer, cam_real, prj_im_sz, cp_sz,attacked_names = None):
    # compute projector-based attack stats and create a result montage
    with torch.no_grad():
        # crop

        cam_scene_cp = cc(cam_scene.squeeze(), cp_sz)
        cam_real_t_cp = cc(cam_real[t], cp_sz)


        # resize to prj_im_sz
        cam_scene_cp_rz = resize(cam_scene_cp, tuple(prj_im_sz))
        cam_real_t_cp_rz = resize(cam_real_t_cp, tuple(prj_im_sz))



        # calculate normalized perturbation for pseudocolor visualization
        cam_real_diff = torch.abs(cam_real_t_cp_rz - cam_scene_cp_rz)
        cam_real_diff = (cam_real_diff - cam_real_diff.min()) / (cam_real_diff.max() - cam_real_diff.min())

        # to pseudo color
        cam_real_diff_color = cv.applyColorMap(np.uint8(cam_real_diff.cpu().numpy().mean(0) * 255), cv.COLORMAP_JET)
        cam_real_diff_color = (
                torch.Tensor(cv.cvtColor(cam_real_diff_color, cv.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255).to(
            cam_scene.device)


        # create result montage
        im = make_grid(
            torch.stack((cam_scene_cp_rz, prj_adv[t], cam_real_t_cp_rz,
                         cam_real_diff_color), 0),
            nrow=4, padding=5,
            pad_value=1)

        # calculate stats on cropped image
        prj_l2 = ut.l2_norm(prj_adv[t], im_gray)/255.0  # mean L2 norm, consistent with Zhao_CVPR_20
        real_dE = deltaE(cam_real_t_cp, cam_scene_cp)

        # For CAPAA and CAPAA (w/o attention)
        if attacked_names == None:

            im = expand_boarder(im, (0, 26, 0, 0))
            im = insert_text(im,f'{imgnet_labels[ret["scene"][2][0, 0].item()]} ({ret["scene"][1][0, 0]:.2f})', (5, 14), 14)
            im = insert_text(im,f'{imgnet_labels[ret["real"][2][t, 0].item()]} ({ret["real"][1][t, 0]:.2f})', (540, 14), 14)
        else:
            # For SPAA and CAPAA (classifier-specific)
            im = expand_boarder(im, (0, 54, 0, 0))
            i = 0
            for attacked_name in attacked_names:
                i += 1
                im = insert_text(im,
                                 f'{attacked_name + " : "+ imgnet_labels[ret["scene" + attacked_name][2][0, 0].item()]} (p={ret["scene" + attacked_name][1][0, 0].item():.2f})',
                                 (5, 14 * i), 13)
                im = insert_text(im,
                                 f'{attacked_name + " : "+ imgnet_labels[ret["real" + attacked_name][2][t, 0].item()]} (p={ret["real" + attacked_name][1][t, 0].item():.2f})',
                                 (530, 14 * i), 13)

        im = insert_text(im, f'Cam-captured scene under changed pose ({t})', (0, 0), 14)

        im = insert_text(im, 'Model inferred adversarial projection', (290, 0), 14)
        im = insert_text(im, f'L2={prj_l2:.2f}', (370, 28), 14)



        im = insert_text(im, 'Real cam-captured projection', (550, 0), 14)
        # im = insertText(im, 'SSIM:{:.2f}'.format(real_ssim), (975, 14), 14)
        # im = insert_text(im, 'L2={:.2f}'.format(real_dE), (980, 14), 14)
        im = insert_text(im, f'dE={real_dE:.2f}', (730, 28), 14)
        im = insert_text(im, 'Normalized difference of 5th-1st', (820, 0), 14)
        # im = insert_text(im, 'Normalized difference of 5th-3rd', (1590, 0), 14)
        # vfs(im)

    return im


def attack_results(ret, t, imgnet_labels, im_gray, prj_adv, warped_prj, cam_scene, cam_infer, cam_real, prj_im_sz, cp_sz,attacked_names = None):
    # compute projector-based attack stats and create a result montage
    with torch.no_grad():
        # crop
        # cp_sz = classifier_crop_sz = (240, 240)
        cam_scene_cp = cc(cam_scene.squeeze(), cp_sz)
        cam_real_t_cp = cc(cam_real[t], cp_sz)
        cam_infer_t_cp = cc(cam_infer[t], cp_sz)

        warped_prj_t_cp = cc(warped_prj[t], cp_sz)

        # resize to prj_im_sz
        cam_scene_cp_rz = resize(cam_scene_cp, tuple(prj_im_sz))
        cam_real_t_cp_rz = resize(cam_real_t_cp, tuple(prj_im_sz))
        cam_infer_t_cp_rz = resize(cam_infer_t_cp, tuple(prj_im_sz))
        warped_prj_t_cp_rz = resize(warped_prj_t_cp,tuple(prj_im_sz))


        # calculate normalized perturbation for pseudocolor visualization
        cam_real_diff = torch.abs(cam_real_t_cp_rz - cam_scene_cp_rz)
        cam_real_diff = (cam_real_diff - cam_real_diff.min()) / (cam_real_diff.max() - cam_real_diff.min())

        # to pseudo color
        cam_real_diff_color = cv.applyColorMap(np.uint8(cam_real_diff.cpu().numpy().mean(0) * 255), cv.COLORMAP_JET)
        cam_real_diff_color = (
                    torch.Tensor(cv.cvtColor(cam_real_diff_color, cv.COLOR_BGR2RGB).transpose((2, 0, 1))) / 255).to(
            cam_scene.device)



        # create result montage
        im = make_grid(
            torch.stack((cam_scene_cp_rz, prj_adv[t], warped_prj_t_cp_rz, cam_infer_t_cp_rz, cam_real_t_cp_rz, cam_real_diff_color), 0),
            nrow=6, padding=5,
            pad_value=1)


        # calculate stats on cropped image
        prj_l2  = ut.l2_norm(prj_adv[t]    , im_gray) / 255.0  # mean L2 norm, consistent with Zhao_CVPR_20
        pred_dE = deltaE(cam_infer_t_cp,cam_scene_cp)
        # real_l2 = ut.l2_norm(cam_real_t_cp , cam_scene_cp)
        # real_dE = ciede2000_diff(rgb2lab_diff(expand_4d(cam_real_t_cp),device = 'cuda' if torch.cuda.is_available() else 'cpu'), rgb2lab_diff(expand_4d(cam_scene_cp),device = 'cuda' if torch.cuda.is_available() else 'cpu'), device = 'cuda' if torch.cuda.is_available() else 'cpu').mean(1).mean(1)
        real_dE = deltaE(cam_real_t_cp,cam_scene_cp)

        # For CAPAA and CAPAA (w/o attention)
        if attacked_names == None:

            im = expand_boarder(im, (0, 26, 0, 0))
            im = insert_text(im,f'{imgnet_labels[ret["scene"][2][0, 0].item()]} ({ret["scene"][1][0, 0]:.2f})',(5, 14 ), 14)
            im = insert_text(im,f'{imgnet_labels[ret["infer"][2][t, 0].item()]} ({ret["infer"][1][t, 0]:.2f})',(790, 14), 14)
            im = insert_text(im,f'{imgnet_labels[ret["real"][2][t, 0].item()]} ({ret["real"][1][t, 0]:.2f})',(1050, 14), 14)
        else:
            # For SPAA and CAPAA (classifier-specific)
            im = expand_boarder(im, (0, 54, 0, 0))
            i = 0
            for attacked_name in attacked_names:
                i += 1
                im = insert_text(im,
                                 f'{attacked_name + " : "+ imgnet_labels[ret["scene" + attacked_name][2][0, 0].item()]} (p={ret["scene" + attacked_name][1][0, 0].item():.2f})',
                                 (5, 14 * i), 13)
                im = insert_text(im,
                                 f'{attacked_name + " : "+ imgnet_labels[ret["infer" + attacked_name][2][t, 0].item()]} (p={ret["infer" + attacked_name][1][t, 0].item():.2f})',
                                 (790, 14 * i), 13)
                im = insert_text(im,
                                 f'{attacked_name + " : "+ imgnet_labels[ret["real" + attacked_name][2][t, 0].item()]} (p={ret["real" + attacked_name][1][t, 0].item():.2f})',
                                 (1050, 14 * i), 13)

        im = insert_text(im, f'Cam-captured scene under original pose ({t})', (0, 0), 14)
        im = insert_text(im, 'Inferred cam-captured prj (original pose)', (790, 0), 14)
        im = insert_text(im, 'Real cam-captured projection', (1080, 0), 14)

        im = insert_text(im, 'Model inferred adversarial projection', (285, 0), 14)
        im = insert_text(im, f'L2={prj_l2:.2f}', (370, 28), 14)

        im = insert_text(im, 'Inferred geometric distortion of projection', (530, 0), 14)

        im = insert_text(im, f'dE={pred_dE:.2f}', (980, 28), 14)


        im = insert_text(im, f'dE={real_dE:.2f}', (1240, 28), 14)
        im = insert_text(im, 'Normalized difference of 5th-1st', (1330, 0), 14)
        im = insert_text(im, 'Normalized difference of 5th-3rd', (1590, 0), 14)
        # vfs(im)

    return im

def summarize_single_attacker(attacker_name, data_root, setup_list, device='cuda', device_ids=[0], pose = 'original',classifier_names = ['inception_v3', 'resnet18', 'vgg16']):
    # given the attacker_name and setup_list, summarize all attacks, create stats.txt/xls and montages

    # set PyTorch device to GPU
    device = torch.device(device)
    attacker_name_list = attacker_name


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
    d_threshes = [2, 3, 4, 5]


    for setup_name in setup_list:
        for attacker_name in attacker_name_list:
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


            for stealth_loss in stealth_losses:
                for d_thr in d_threshes:
                    for classifier_name in classifier_names:
                        attack_ret_folder = join(attacker_cfg_str, stealth_loss, str(d_thr), classifier_name)
                        prj_adv_path   = join(setup_path, 'prj/adv'      , attack_ret_folder)
                        cam_infer_path = join(setup_path, 'cam/infer/adv', attack_ret_folder)
                        cam_real_path  = join(setup_path, 'cam/raw/adv'  , attack_ret_folder)
                        warped_prj_adv_path = join(setup_path, 'prj/warped_adv', attack_ret_folder)

                        # check whether all images are captured for results summary
                        dirs_to_check = [prj_adv_path, cam_real_path]
                        skip = False
                        dirs_to_check.append(cam_infer_path)

                        for img_dir in dirs_to_check:
                            if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
                                print(f'No such folder/images: {img_dir}\n'
                                      f'Maybe [{attacker_name}] has no [{join(stealth_loss, str(d_thr), classifier_name)}] attack cfg, or you forget to project and capture.\n')
                                skip = True
                                break
                        if skip:
                            break

                        prj_adv   = ut.torch_imread_mt(prj_adv_path).to(device)
                        cam_real  = ut.torch_imread_mt(cam_real_path).to(device)
                        cam_infer = ut.torch_imread_mt(cam_infer_path).to(device)
                        warped_prj_adv = ut.torch_imread_mt(warped_prj_adv_path).to(device)

                        ret = {}  # classification result dict
                        with torch.no_grad():
                            classifier = Classifier(classifier_name, device, device_ids, fix_params=True)
                            ret['scene'] = classifier(cam_scene, cp_sz)
                            ret['infer'] = classifier(cam_infer, cp_sz)
                            ret['real'] = classifier(cam_real , cp_sz)

                        # create the result montage as shown in CAPAA main paper Figs. 4-5 and supplementary
                        im_montage = []
                        if pose == 'original':
                            for t in range(n + 1):  # when t = n + 1, it is the untargeted attack result
                                im_montage.append(attack_results(ret, t, imagenet_labels, im_gray, prj_adv, warped_prj_adv, cam_scene, cam_infer, cam_real, setup_info['prj_im_sz'], cp_sz))
                        else:
                            for t in range(n + 1):  # when t = n + 1, it is the untargeted attack result
                                im_montage.append(attack_results_change_pose(ret, t, imagenet_labels, im_gray, prj_adv,
                                                                             warped_prj_adv, cam_scene, cam_infer,
                                                                             cam_real, setup_info['prj_im_sz'], cp_sz))
                        # [debug] show montage in visdom
                        # vfs(torch.stack(im_montage, 0), ncol=1, title=attacker_cfg_str + '_' + stealth_loss + '_' + str(d_thr) + '_' + classifier_name)

                        # save montage
                        montage_path = join(setup_path, 'ret', attack_ret_folder)
                        ut.save_imgs(torch.stack(im_montage, 0), montage_path)

                        # Targeted: top-1 and top-5 success rate
                        # inferred attacks
                        t1_infer = np.count_nonzero(ret['infer'][2][:n, 0] == target_idx) / n
                        t5_infer = np.count_nonzero([target_idx[i] in ret['infer'][2][i, :5] for i in range(n)]) / n

                        # real camera-captured attacks
                        t1_real  = np.count_nonzero(ret['real'][2][:n, 0] == target_idx) / n
                        t5_real  = np.count_nonzero([target_idx[i] in ret['real'][2][i, :5] for i in range(n)]) / n

                        # Untargeted: top-1 success rate
                        true_idx = ret['scene'][2][0, 0]
                        t1_untar_infer = np.count_nonzero(ret['infer'][2][n, 0] != true_idx)
                        t1_untar_real  = np.count_nonzero(ret['real'][2][n, 0]  != true_idx)

                        # calc image similarity metrics
                        table.loc[len(table)] = [
                            setup_name, attacker_cfg_str, stealth_loss, d_thr, classifier_name, t1_infer, t5_infer, t1_real,
                            t5_real, t1_untar_infer, t1_untar_real,
                            # model infer vs GT on the validation data
                            *valid_ret,

                            # targeted [0, n-1]
                            *calc_img_dists(prj_adv[:n], im_gray.expand_as(prj_adv[:n])),  # prj adv
                            *calc_img_dists(cc(cam_infer[:n], cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_infer[:n], cp_sz))),  # cam infer
                            *calc_img_dists(cc(cam_real[:n], cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_real[:n], cp_sz))),
                            # cam real (the last four columns of CAPAA paper Table)

                            # untargeted [n]
                            *calc_img_dists(prj_adv[n, None], im_gray.expand_as(prj_adv[n, None])),  # prj adv
                            *calc_img_dists(cc(cam_infer[n, None], cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_infer[n, None], cp_sz))),  # cam infer
                            *calc_img_dists(cc(cam_real[n, None], cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_real[n, None], cp_sz))),  # cam real

                            # both targeted and untargeted [0, n].
                            *calc_img_dists(prj_adv, im_gray.expand_as(prj_adv)),  # prj adv
                            *calc_img_dists(cc(cam_infer, cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_infer, cp_sz))),  # cam infer
                            *calc_img_dists(cc(cam_real, cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_real, cp_sz)))  # cam real
                        ]

            # print results
            print(f'\n-------------------- [{attacker_name}] results on [{setup_name}] --------------------')
            print(table.to_string(index=False, float_format='%.4f'))
            print('-------------------------------------- End of result table ---------------------------\n')

            # save stats to files
            ret_path = join(setup_path, 'ret', attacker_cfg_str)
            if not os.path.exists(ret_path): os.makedirs(ret_path)
            table.to_csv(join(ret_path, 'stats.txt'), index=False, float_format='%.4f', sep='\t')
            table.to_excel(join(ret_path, 'stats.xlsx'), float_format='%.4f', index=False)
    return table



def summarize_single_attacker_for_UAP_task(attacker_name, data_root, setup_list, device='cuda', device_ids=[0],pose='original', classifier_names = ['inception_v3', 'resnet18', 'vgg16']):
    # given the attacker_name and setup_list, summarize all attacks, create stats.txt/xls and montages
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
    columns = ['Setup', 'Attacker', 'Stealth_loss', 'd_thr', 'Classifier', 'Attacked_Classifier', 'T.top-1_infer', 'T.top-5_infer', 'T.top-1_real', 'T.top-5_real',
               'U.top-1_infer', 'U.top-1_real'] + [phase[0] + '_' + y for y in metrics] +\
              ['T.' + x + '_' + y for x in phase[1:] for y in metrics] + ['U.' + x + '_' + y for x in phase[1:] for y in metrics] + \
              ['All.' + x + '_' + y for x in phase[1:] for y in metrics]

    # stealth_losses = ['caml2', 'camdE', 'camdE_caml2', 'camdE_caml2_prjl2', '-']
    stealth_losses = ['camdE_caml2']
    d_threshes       = [2, 3, 4, 5]

    for setup_name in setup_list:
        for attacker_name in attacker_name_list:
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

            for stealth_loss in stealth_losses:
                for d_thr in d_threshes:
                    for classifier_name in classifier_names:

                        ret = {}  # classification result dict
                        attack_ret_folder = join(attacker_cfg_str, stealth_loss, str(d_thr), classifier_name)
                        prj_adv_path = join(setup_path, 'prj/adv', attack_ret_folder)
                        cam_infer_path = join(setup_path, 'cam/infer/adv', attack_ret_folder)
                        cam_real_path = join(setup_path, 'cam/raw/adv', attack_ret_folder)
                        warped_prj_adv_path = join(setup_path, 'prj/warped_adv', attack_ret_folder)

                        prj_adv = ut.torch_imread_mt(prj_adv_path).to(device)
                        cam_real = ut.torch_imread_mt(cam_real_path).to(device)
                        cam_infer = ut.torch_imread_mt(cam_infer_path).to(device)
                        warped_prj_adv = ut.torch_imread_mt(warped_prj_adv_path).to(device)

                        for attacked_name in classifier_names:


                            # check whether all images are captured for results summary
                            dirs_to_check = [prj_adv_path, cam_real_path]
                            skip = False
                            dirs_to_check.append(cam_infer_path)

                            for img_dir in dirs_to_check:
                                if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
                                    print(f'No such folder/images: {img_dir}\n'
                                          f'Maybe [{attacker_name}] has no [{join(stealth_loss, str(d_thr), classifier_name)}] attack cfg, or you forget to project and capture.\n')
                                    skip = True
                                    break
                            if skip:
                                break

                            with torch.no_grad():
                                classifier = Classifier(attacked_name, device, device_ids, fix_params=True)
                                ret['scene' + attacked_name] = classifier(cam_scene, cp_sz)
                                ret['infer' + attacked_name] = classifier(cam_infer, cp_sz)
                                ret['real' + attacked_name] = classifier(cam_real, cp_sz)

                            # Targeted: top-1 and top-5 success rate
                            # inferred attacks
                            t1_infer = np.count_nonzero(ret['infer' + attacked_name][2][:n, 0] == target_idx) / n
                            t5_infer = np.count_nonzero(
                                [target_idx[i] in ret['infer'+ attacked_name][2][i, :5] for i in range(n)]) / n

                            # real camera-captured attacks
                            t1_real = np.count_nonzero(ret['real' + attacked_name][2][:n, 0] == target_idx) / n
                            t5_real = np.count_nonzero(
                                [target_idx[i] in ret['real'+ attacked_name][2][i, :5] for i in range(n)]) / n

                            # Untargeted: top-1 success rate
                            true_idx = ret['scene' + attacked_name][2][0, 0]
                            t1_untar_infer = np.count_nonzero(ret['infer' + attacked_name][2][n, 0] != true_idx)
                            t1_untar_real = np.count_nonzero(ret['real' + attacked_name][2][n, 0] != true_idx)

                            # calc image similarity metrics
                            table.loc[len(table)] = [
                                setup_name, attacker_cfg_str, stealth_loss, d_thr, classifier_name, attacked_name,
                                t1_infer, t5_infer,
                                t1_real,
                                t5_real, t1_untar_infer, t1_untar_real,
                                # model infer vs GT on the validation data
                                *valid_ret,

                                # targeted [0, n-1]
                                *calc_img_dists(prj_adv[:n], im_gray.expand_as(prj_adv[:n])),  # prj adv
                                *calc_img_dists(cc(cam_infer[:n], cp_sz),
                                                cc(cam_scene, cp_sz).expand_as(cc(cam_infer[:n], cp_sz))),
                                # cam infer
                                *calc_img_dists(cc(cam_real[:n], cp_sz),
                                                cc(cam_scene, cp_sz).expand_as(cc(cam_real[:n], cp_sz))),
                                # cam real (the last four columns of CAPAA paper Table)

                                # untargeted [n]
                                *calc_img_dists(prj_adv[n, None], im_gray.expand_as(prj_adv[n, None])),  # prj adv
                                *calc_img_dists(cc(cam_infer[n, None], cp_sz),
                                                cc(cam_scene, cp_sz).expand_as(cc(cam_infer[n, None], cp_sz))),
                                # cam infer
                                *calc_img_dists(cc(cam_real[n, None], cp_sz),
                                                cc(cam_scene, cp_sz).expand_as(cc(cam_real[n, None], cp_sz))),
                                # cam real

                                # both targeted and untargeted [0, n].
                                *calc_img_dists(prj_adv, im_gray.expand_as(prj_adv)),  # prj adv
                                *calc_img_dists(cc(cam_infer, cp_sz),
                                                cc(cam_scene, cp_sz).expand_as(cc(cam_infer, cp_sz))),
                                # cam infer
                                *calc_img_dists(cc(cam_real, cp_sz),
                                                cc(cam_scene, cp_sz).expand_as(cc(cam_real, cp_sz)))
                                # cam real

                            ]

                        # no montage
                        # create the result montage as shown in CAPAA main paper
                        im_montage = []
                        if pose == 'original':
                            for t in range(n + 1):  # when t = n + 1, it is the untargeted attack result
                                im_montage.append(
                                    attack_results(ret, t, imagenet_labels, im_gray, prj_adv, warped_prj_adv,
                                                   cam_scene, cam_infer, cam_real, setup_info['prj_im_sz'], cp_sz,classifier_names))
                        else:
                            for t in range(n + 1):  # when t = n + 1, it is the untargeted attack result
                                im_montage.append(
                                    attack_results_change_pose(ret, t, imagenet_labels, im_gray, prj_adv,
                                                               warped_prj_adv, cam_scene, cam_infer,
                                                               cam_real, setup_info['prj_im_sz'], cp_sz,classifier_names))

                        # [debug] show montage in visdom
                        # vfs(torch.stack(im_montage, 0), ncol=1, title=attacker_cfg_str + '_' + stealth_loss + '_' + str(d_thr) + '_' + classifier_name)

                        # save montage
                        montage_path = join(setup_path, 'ret', attack_ret_folder)
                        ut.save_imgs(torch.stack(im_montage, 0), montage_path)



            # print results
            print(f'\n-------------------- [{attacker_name}] results on [{setup_name}] for classifier-agnostic attacks --------------------')
            print(table.to_string(index=False, float_format='%.4f'))
            print('-------------------------------------- End of classifier-agnostic attack result table ---------------------------\n')
            # print(table.filter(regex = 'top-[0-9]_', axis = 1).to_string(index = False, float_format = '%.2f'))  # columns that only contain success rates
            # print(table.filter(regex = '_L2'    , axis = 1).to_string(index = False, float_format = '%.4f'))  # columns that only contain L2
            # print(table.filter(regex = '_dE'    , axis = 1).to_string(index = False, float_format = '%.4f'))  # columns that only contain dE

            # save stats to files
            ret_path = join(setup_path, 'ret', attacker_cfg_str)
            if not os.path.exists(ret_path): os.makedirs(ret_path)
            table.to_csv(join(ret_path,'universal_stats.txt'), index=False, float_format='%.4f', sep='\t')
            table.to_excel(join(ret_path,'universal_stats.xlsx'), float_format='%.4f', index=False)
    return table
