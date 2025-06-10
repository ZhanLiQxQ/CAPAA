"""
Useful functions for projector-based adversarial attack
"""
import os
from os.path import join
import numpy as np
import pandas as pd
import torch
from torchvision import models
from torchvision import transforms
from omegaconf import DictConfig
from train_network import load_setup_info, train_eval_pcnet, get_model_train_cfg
from img_proc import resize, expand_4d, center_crop as cc, invertGrid
import utils as ut
from utils import calc_img_dists
from differential_color_functions import rgb2lab_diff, ciede2000_diff
import itertools
from classifier import Classifier, load_imagenet_labels
from tqdm import tqdm
import torch.nn.functional as F
from grad_cam import GradCAM,GradCAMpp,resize_to_original
from projector_based_attack_cam import spaa_and_classifier_specific_capaa,attack_results_change_pose,attack_results,summarize_single_attacker_for_UAP_task
from vit_model import vit_base_patch16_224

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


            cam_scene = cam_scene.to(device)

            # To automate the comparison tests of the 4 methods, the test sequence is set as 'training PCNet -> test SPAA -> test 3 CAPAA-related methods'.
            # If you want to re-train the PCNet for a specific method, please set its load_pretrained = False.
            if attacker_name == 'SPAA':
                # TODO: set load_pretrained to False
                load_pretrained = False
            else:
                load_pretrained = True
            # train or load PCNet model
            model_cfg = get_model_train_cfg(model_list=None, data_root=cfg.data_root, setup_list=[setup_name], device_ids=cfg.device_ids,
                                            load_pretrained=load_pretrained, plot_on=cfg.plot_on)
            model_cfg.model_list = ['PCNet']
            # model_cfg.max_iters = 100 # debug
            model, model_ret, model_cfg, fine_grid = train_eval_pcnet(model_cfg)


            # set to evaluation mode
            model.eval()

            # fix model weights
            for param in model.parameters():
                param.requires_grad = False


            attacker_cfg_str = to_attacker_cfg_str(attacker_name)[0]
            cfg.model_cfg = model_cfg

            # we perform n = 10 targeted attacks and 1 untargeted attack
            n = 10
            target_labels = dict(itertools.islice(target_labels.items(), n))
            target_idx    = list(target_labels.keys())

            for stealth_loss in cfg.stealth_losses:
                for d_thr in cfg.d_threshes:
                    # check whether the method is classifier-agnostic
                    if attacker_name in ['CAPAA (without attention)', 'CAPAA']:
                        # check whether the method is attenion-based
                        if attacker_name == 'CAPAA (without attention)':
                            attention_use = False
                        else:
                            attention_use = True

                        attack_ret_folder = join(attacker_cfg_str, stealth_loss, str(d_thr), "classifier_all")
                        cam_infer_adv_path = join(setup_path, 'cam/infer/adv', attack_ret_folder)
                        prj_adv_path = join(setup_path, 'prj/adv', attack_ret_folder)
                        warped_prj_adv_path = join(setup_path, 'prj/warped_adv', attack_ret_folder)

                        print("Classifier names:", cfg.classifier_names)
                        print("Number of classifier names:", len(cfg.classifier_names))

                        classifiers = {}
                        true_indices = []
                        true_labels_info = []  # For printing

                        for classifier_name_idx, classifier_name_val in enumerate(cfg.classifier_names):
                            classifier_obj = Classifier(classifier_name_val, device, cfg.device_ids, fix_params=True)
                            classifiers[classifier_name_val] = classifier_obj
                            with torch.no_grad():
                                _, p_val, pred_idx_val = classifier_obj(cam_scene, cp_sz)

                            current_true_idx = pred_idx_val[0, 0].item()  # Use .item() to get Python scalar
                            true_indices.append(current_true_idx)
                            current_true_label = imagenet_labels[current_true_idx]
                            true_labels_info.append(
                                f'Original prediction for {classifier_name_val} : {current_true_label}, p={p_val.max().item():.2f}'
                            )

                        print(
                            f'\n-------------------- [{attacker_name}] attacking ["all the classifiers"], Loss: [{stealth_loss}], d_thr: [{d_thr}] --------')
                        for info_line in true_labels_info:
                            print(info_line)

                        # Prepare true_indices for attack functions (list of lists, e.g., [[idx0], [idx1], ...])
                        true_indices_for_attack = [[ti] for ti in true_indices]

                        # untargeted attack
                        targeted_attack = False
                        print(f'[Untargeted] attacking [{"classifier_all"}]...')


                        # For 'CAPAA (without attention)', 'CAPAA' -> spaa_universal
                        # Uses all available classifiers from cfg.classifier_names
                        cam_infer_adv_untar, prj_adv_untar, warped_prj_untar = capaa(
                            model, classifiers, imagenet_labels,
                            target_idx, true_indices_for_attack,  # Pass lists
                            targeted_attack, cam_scene, d_thr, stealth_loss,
                            cfg.device, setup_info, fine_grid, attention_use
                        )

                        # targeted attack (batched)
                        targeted_attack = True
                        v = 7  # we only show one adversarial target in the console, v is the index
                        print(
                            f'\n[ Targeted ] attacking ["all the classifiers"], target: ({imagenet_labels[target_idx[v]]})...')


                         # For CAPAA (w/o attention), CAPAA -> spaa_universal
                        cam_infer_adv_tar, prj_adv_tar, warped_prj_tar = capaa(
                            model, classifiers, imagenet_labels, target_idx,
                            true_indices_for_attack,  # Pass lists
                            targeted_attack, cam_scene, d_thr, stealth_loss,
                            cfg.device, setup_info, fine_grid, attention_use
                        )

                        ut.save_imgs(expand_4d(torch.cat((cam_infer_adv_tar, cam_infer_adv_untar), 0)),
                                     cam_infer_adv_path)
                        ut.save_imgs(expand_4d(torch.cat((prj_adv_tar, prj_adv_untar), 0)), prj_adv_path)
                        ut.save_imgs(expand_4d(torch.cat((warped_prj_tar, warped_prj_untar), 0)), warped_prj_adv_path)


                    else:
                        # For 'SPAA', 'CAPAA (classifier-specific)' -> spaa
                        for classifier_name in cfg.classifier_names:
                            if attacker_name == 'CAPAA (classifier-specific)':
                                attention_use = True
                            else:
                                attention_use = False
                            attack_ret_folder = join(attacker_cfg_str, stealth_loss, str(d_thr), classifier_name)
                            cam_infer_adv_path = join(setup_path, 'cam/infer/adv', attack_ret_folder)
                            prj_adv_path = join(setup_path, 'prj/adv', attack_ret_folder)
                            warped_prj_adv_path = join(setup_path, 'prj/warped_adv', attack_ret_folder)

                            # get the true label of the current scene
                            classifier = Classifier(classifier_name, device, cfg.device_ids, fix_params=True)
                            with torch.no_grad():
                                _, p, pred_idx = classifier(cam_scene, cp_sz)
                            true_idx = pred_idx[0, 0].item()  # true label index of the scene given by the current classifier
                            true_label = imagenet_labels[true_idx]

                            print(
                                f'\n-------------------- [{attacker_name}] attacking [{classifier_name}], original prediction: ({true_label}, '
                                f'p={p.max():.2f}), Loss: [{stealth_loss}], d_thr: [{d_thr}] --------')

                            # untargeted attack
                            targeted_attack = False
                            print(f'[Untargeted] attacking [{classifier_name}]...')

                            cam_infer_adv_untar, prj_adv_untar, warped_prj_untar = spaa_and_classifier_specific_capaa(model, classifier,
                                                                                                                      imagenet_labels, [true_idx],
                                                                                                                      targeted_attack, cam_scene,
                                                                                                                      d_thr, stealth_loss,
                                                                                                                      cfg.device, setup_info,
                                                                                                                      fine_grid, attention_use)

                            # targeted attack (batched)
                            targeted_attack = True
                            v = 7  # we only show one adversarial target in the console, v is the index

                            print(
                                f'\n[ Targeted ] attacking [{classifier_name}], target: ({imagenet_labels[target_idx[v]]})...')
                            cam_infer_adv_tar, prj_adv_tar, warped_prj_tar = spaa_and_classifier_specific_capaa(model, classifier, imagenet_labels,
                                                                                                                target_idx, targeted_attack,
                                                                                                                cam_scene, d_thr, stealth_loss,
                                                                                                                cfg.device, setup_info, fine_grid,
                                                                                                                attention_use)
                            ut.save_imgs(expand_4d(torch.cat((cam_infer_adv_tar, cam_infer_adv_untar), 0)),
                                         cam_infer_adv_path)
                            ut.save_imgs(expand_4d(torch.cat((prj_adv_tar, prj_adv_untar), 0)), prj_adv_path)
                            ut.save_imgs(expand_4d(torch.cat((warped_prj_tar, warped_prj_untar), 0)),
                                         warped_prj_adv_path)

            print(f'\nThe next step is to project and capture [{attacker_name}] generated adversarial projections in {join(setup_path, "prj/adv", attacker_cfg_str)}')

    return cfg


def project_capture_real_attack(cfg, cam):
    attacker_name_list = cfg.attacker_name
    for attacker_name in attacker_name_list:

        setup_path = join(cfg.data_root, 'setups', cfg.setup_list[0])
        setup_info = load_setup_info(setup_path)

        for stealth_loss in cfg.stealth_losses:
            for d_thr in cfg.d_threshes:
                for classifier_name in cfg.classifier_names:
                    attacker_cfg_str = to_attacker_cfg_str(attacker_name)[0]
                    attack_ret_folder = join(attacker_cfg_str, stealth_loss, str(d_thr), classifier_name)
                    prj_input_path = join(setup_path, 'prj/adv', attack_ret_folder)
                    cam_cap_path = join(setup_path, 'cam/raw/adv', attack_ret_folder)
                    ut.project_capture_data(prj_input_path, cam_cap_path, setup_info, cam)


def get_attacker_cfg(attacker_name, data_root, setup_list, device_ids=[0], plot_on=True,d_threshes = [2, 3, 4, 5]):
    # default projector-based attacker configs
    cfg_default = DictConfig({})
    cfg_default.attacker_name = attacker_name
    # debug it
    cfg_default.classifier_names = ['inception_v3', 'resnet18', 'vgg16','vit_b_16']
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

    model_cfg = get_model_train_cfg(model_list=['PCNet'], single=True)
    model_cfg_str = f'{model_cfg.model_name}_{model_cfg.loss}_{model_cfg.num_train}_{model_cfg.batch_size}_{model_cfg.max_iters}'
    attacker_cfg_str = f'{attacker_name}_{model_cfg_str}'

    return attacker_cfg_str, model_cfg_str


def capaa(pcnet, classifier_list, imagenet_labels, target_idx_list_for_targeted, list_of_true_idx_lists, targeted, cam_scene, d_thr, stealth_loss, device, setup_info, fine_grid=None, attention_use=False):
    '''
    Classifier-Agnostic Projector-based Adversarial Attacks (CAPAA) and CAPAA (without attention) attack function.
    @param pcnet:
    @param classifier_list: dict of Classifier objects, e.g., {'inception_v3': Classifier(...), 'resnet18': Classifier(...), ...}
    @param imagenet_labels:
    @param target_idx_list_for_targeted: List of target indices for targeted attacks (e.g., [target_for_all_img1, target_for_all_img2,...])
    @param list_of_true_idx_lists: For untargeted, a list where each element is a list containing the true index for that classifier.
                                          # e.g., [[true_idx_for_classifier0], [true_idx_for_classifier1], ...]
    @param targeted:
    @param cam_scene:
    @param d_thr:
    @param stealth_loss:
    @param device:
    @param setup_info:
    @param fine_grid:
    @param attention_use: boolean, whether to use attention mechanism in the attack
    @return:
    '''

    print("Device configuration:", device)
    num_classifiers = len(classifier_list)

    if targeted:
        num_target = len(target_idx_list_for_targeted)
        target_idx_tensor = torch.tensor(target_idx_list_for_targeted, device=device, dtype=torch.long)
    else:
        # For untargeted attacks, each classifier has its own "original" class to move away from.
        num_target = len(list_of_true_idx_lists[0])

    cp_sz = setup_info['classifier_crop_sz']
    cam_scene_batch = cam_scene.expand(num_target, -1, -1, -1).to(device)
    im_gray = setup_info['prj_brightness'] * torch.ones(num_target, 3, *setup_info['prj_im_sz']).to(
        device)  # Projector input image (initially gray)
    prj_adv = im_gray.clone()
    prj_adv.requires_grad = True

    v = 7 if targeted else 0  # Index for debug printing

    adv_lr = 2  # Learning rate for adversarial loss
    col_lr = 1  # Learning rate for color loss (stealthiness loss)


    classifier_weights = [1.0 / num_classifiers] * num_classifiers

    adv_w = 1  # Weight for adversarial loss
    prjl2_w = 0.1 if 'prjl2' in stealth_loss else 0  # Weight for projector input image L2 loss
    caml2_w = 1 if 'caml2' in stealth_loss else 0  # Weight for camera captured image L2 loss
    camdE_w = 1 if 'camdE' in stealth_loss else 0  # Weight for camera captured image deltaE loss
    p_thresh = 0.9  # Threshold for adversarial confidence
    iters = 200  # you may improve it for better results

    prj_adv_best = prj_adv.clone()  # Stores the best adversarial projection
    cam_infer_best = cam_scene.repeat(prj_adv_best.shape[0], 1, 1,
                                      1)  # Stores the best inferred camera image (num_target, C, H, W)
    col_loss_best = 1e6 * torch.ones(prj_adv_best.shape[0]).to(device)  # Stores the best color loss for each batch item


    cam_models = {
        'vgg16': models.vgg16(weights=models.VGG16_Weights.DEFAULT).eval().to(device),
        'inception_v3': models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).eval().to(device),
        'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT).eval().to(device),
        'vit_b_16': vit_base_patch16_224().eval().to(device)
    }

    cam_models['vit_b_16'].load_state_dict(torch.load("./vit_base_patch16_224.pth", map_location=device))

    target_layers = {  # Target layers for different models
        'vgg16': cam_models['vgg16'].features,
        'inception_v3': cam_models['inception_v3'].Mixed_7c,
        'resnet18': cam_models['resnet18'].layer4,
        'vit_b_16': [cam_models['vit_b_16'].blocks[-1].norm1]
    }

    grad_cam_generators = {}  # Stores GradCAM generators
    for name, model_instance in cam_models.items():
        if name == 'vit_b_16':
            grad_cam_generators[name] = GradCAM(model=model_instance, target_layers=target_layers[name],
                                                use_cuda=torch.cuda.is_available(),
                                                reshape_transform=ReshapeTransform(model_instance))
        elif name in target_layers:  # CNN models
            grad_cam_generators[name] = GradCAMpp(model_instance, target_layers[name])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    pil_cam_scene = transforms.ToPILImage()(cam_scene.squeeze(0).cpu())  # (C, H, W) -> PIL 图像

    # ViT specific preprocessing for the scene
    vit_size = (224, 224)
    processed_cam_scene_vit = data_transform(pil_cam_scene.resize(vit_size)).unsqueeze(0).to(device) # tensor (1, 3, 224, 224)

    temp_cam_scene_for_transform = cam_scene.squeeze(0).permute(1, 2, 0).cpu().numpy()
    transformed_original_size_cam_scene = data_transform(temp_cam_scene_for_transform).unsqueeze(0).to(
        device)  # 1,C,H,W tensor (1,3,240,320)



    all_grayscale_cams = []  # grayscale CAM images for all classifiers

    for clf_name in classifier_list:
        if clf_name not in grad_cam_generators:
            print(f"Warning: No GradCAM generator configured for {clf_name}. Skipping CAM for this classifier.")
            dummy_cam = torch.ones_like(cam_scene).to(
                device)  # (3, H_scene, W_scene)
            all_grayscale_cams.append(dummy_cam)
            continue

        cam_gen = grad_cam_generators[clf_name]


        # Determine the input tensor for the current classifier's CAM
        if 'vit_b_16' in clf_name:
            input_tensor_for_cam = processed_cam_scene_vit
            # generate CAM
            grayscale_cam_single = cam_gen(input_tensor_for_cam)
            grayscale_cam_single = torch.from_numpy(grayscale_cam_single).unsqueeze(0).to(device)
        elif 'inception' in clf_name:
            inception_size = (299, 299)
            resized_pil_for_inception = pil_cam_scene.resize(inception_size)
            input_tensor_for_cam = data_transform(resized_pil_for_inception).unsqueeze(0).to(device) # (1,3,299,299) tensor
            # generate CAM
            grayscale_cam_single, _ = cam_gen(input_tensor_for_cam) # (1,1,8,8) tensor

        else:  # other CNN models like VGG16, ResNet18
            input_tensor_for_cam = transformed_original_size_cam_scene.to(device)
            # generate CAM
            grayscale_cam_single, _ = cam_gen(input_tensor_for_cam) # (num_target, 1, H_cam_output, W_cam_output)

        resized_cam = resize_to_original(cam_scene_batch, grayscale_cam_single) # (3, H_scene, W_scene) = (3, 240, 320)

        all_grayscale_cams.append(resized_cam)

    # Average all CAM images
    avg_grayscale_cam = torch.stack(all_grayscale_cams).mean(dim=0)
    CAM_attention = avg_grayscale_cam.expand(num_target, -1, -1, -1).to(device)

    # Inverse warp CAM (prj_CAM_attention)
    prj2cam_grid = fine_grid[0, :].unsqueeze(0)
    cam2prj_grid = torch.Tensor(invertGrid(prj2cam_grid.detach().cpu(), setup_info['prj_im_sz'])).unsqueeze(0).to(
        device)  # (1, H_prj, W_prj, 2)

    prj_cmp_init = F.grid_sample(CAM_attention,  # (num_target, C, H, W)
                                 cam2prj_grid.expand(CAM_attention.shape[0], -1, -1, -1),
                                 # (num_target, H_prj, W_prj, 2)
                                 align_corners=True)

    # desired cam-captured image fov warped to prj image space
    prj_mask_fov = F.grid_sample(pcnet.module.get_mask().float()[None, None], cam2prj_grid,
                                align_corners=True)

    prj_CAM_attention = (prj_cmp_init * prj_mask_fov).to(device)
    prj_CAM_attention = prj_CAM_attention[0, :] # (C, H_prj, W_prj)

    # --- Main attack loop ---
    for i in range(0, iters):
        cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)  # Infer camera-captured image

        total_adv_loss = torch.tensor(0.0, device=device)  # Initialize total adversarial loss

        # Store results for each classifier in this iteration
        current_preds_p = []
        current_preds_idx = []
        current_raw_scores = []

        for clf_idx,(clf_name, classifier_obj) in enumerate(classifier_list.items()):
            raw_score, p, pred_idx = classifier_obj(cam_infer, cp_sz)  # Get predictions
            current_raw_scores.append(raw_score.to(device))
            current_preds_p.append(p.to(device))  # Add p tensor to list
            current_preds_idx.append(pred_idx.to(device))  # Add pred_idx tensor to list

            # Use raw_score for untargeted attacks, log(softmax(raw_score/temp)) for targeted attacks
            temp_softmax_p = F.softmax(raw_score / (i // 40 + 1), dim=1)  # Softmax with temperature parameter

            if targeted:

                adv_loss_clf = classifier_weights[clf_idx] * adv_w * \
                               (-torch.log(temp_softmax_p[torch.arange(
                                   num_target), target_idx_tensor] + 1e-8)).mean()  # 加一个小的epsilon防止log(0)
            else:

                # For untargeted attacks, each classifier has its own list of "original" classes to move away from.
                # list_of_true_idx_lists[clf_idx] is the list of true indices for this classifier, for each batch item.
                true_indices_for_clf_tensor = torch.tensor(list_of_true_idx_lists[clf_idx], device=device,
                                                           dtype=torch.long)
                adv_loss_clf = classifier_weights[clf_idx] * adv_w * \
                               (raw_score[torch.arange(num_target), true_indices_for_clf_tensor]).mean()

            # Accumulate adversarial loss
            total_adv_loss += adv_loss_clf

        # Stealthiness loss (calculated once for the batch)
        prjl2 = torch.norm(im_gray - prj_adv, dim=1).mean(1).mean(1)  # Projector L2 loss
        col_loss_batch = prjl2_w * prjl2
        caml2 = torch.norm(cam_scene_batch - cam_infer, dim=1).mean(1).mean(1)  # Camera L2 loss
        col_loss_batch += caml2_w * caml2
        camdE = ciede2000_diff(rgb2lab_diff(cam_infer, device), rgb2lab_diff(cam_scene_batch, device), device).mean(
            1).mean(1)  # Camera deltaE loss
        col_loss_batch += camdE_w * camdE
        col_loss = col_loss_batch.mean()  # Average stealthiness loss

        # mask_high_conf: List of tensors, one for each classifier
        mask_high_conf_per_clf = [p_tensor[:, 0] > p_thresh for p_tensor in current_preds_p]

        # For overall success, all classifiers for that image must succeed.
        # mask_succ_adv_overall is initially all True for each batch item
        mask_succ_adv_overall = torch.ones(num_target, dtype=torch.bool, device=device)

        if targeted:
            for clf_idx in range(num_classifiers):
                # pred_idx[:,0] should be compared with target_idx_tensor
                mask_succ_clf = (current_preds_idx[clf_idx][:, 0].to(device) == target_idx_tensor)
        else:  # Untargeted
            for clf_idx in range(num_classifiers):
                true_indices_for_clf_tensor = torch.tensor(list_of_true_idx_lists[clf_idx], device=device,
                                                           dtype=torch.long)
                mask_succ_clf = (current_preds_idx[clf_idx][:, 0] != true_indices_for_clf_tensor)
                mask_succ_adv_overall &= mask_succ_clf


        mask_high_pert = (camdE > d_thr)

        # mask_best_adv: Combines overall success, high confidence for all classifiers (if targeted)
        mask_best_adv_overall = mask_succ_adv_overall.clone()
        if targeted:  # For targeted attacks, all classifiers also need high confidence
            for conf_mask_clf in mask_high_conf_per_clf:
                mask_best_adv_overall &= conf_mask_clf
        mask_best_adv_overall &= mask_high_pert

        # for adv_grad
        total_adv_loss.backward(retain_graph=True)
        adv_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        # prj_CAM_attention (C, H_prj, W_prj)
        cam_update_factor = prj_CAM_attention if attention_use else 1.0  # Multiply by CAM if attention is used


        # (num_target, C, H, W)
        norm_adv_grad = adv_grad / (
                    torch.norm(adv_grad.view(num_target, -1), dim=1, keepdim=True).view(num_target, 1, 1,
                                                                                        1) + 1e-8)  # 加epsilon防止除零

        # update unsuccessfully attacked samples using class_loss gradient by lr_class*g/||g||
        prj_adv.data[~mask_best_adv_overall] -= adv_lr * cam_update_factor * norm_adv_grad[~mask_best_adv_overall]



        # for clr_loss_grad
        col_loss.backward()
        col_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        # update successfully attacked samples using color_loss gradient by lr_color*g/||g||
        norm_col_grad = col_grad / (
                    torch.norm(col_grad.view(num_target, -1), dim=1, keepdim=True).view(num_target, 1, 1,
                                                                                        1) + 1e-8)

        prj_adv.data[mask_best_adv_overall] -= col_lr * cam_update_factor * norm_col_grad[mask_best_adv_overall]

        # Update best results
        # mask_best_color is True if current color loss is better than previous best for that batch item
        mask_best_color = col_loss_batch < col_loss_best
        # mask_best: successful overall, high confidence for all classifiers (if targeted) and better color loss
        mask_best_for_update = mask_best_color & mask_best_adv_overall
        col_loss_best[mask_best_for_update] = col_loss_batch.data[mask_best_for_update].clone()  # 更新最佳颜色损失


        # First, Ensure attack success rate is at east maintained
        prj_adv_best[mask_succ_adv_overall] = prj_adv.data[mask_succ_adv_overall].clone()
        cam_infer_best[mask_succ_adv_overall] = cam_infer.data[mask_succ_adv_overall].clone()

        # Then, among those successful samples, update if color loss is improved.
        prj_adv_best[mask_best_for_update] = prj_adv.data[mask_best_for_update].clone()
        cam_infer_best[mask_best_for_update] = cam_infer.data[mask_best_for_update].clone()

        if i % 30 == 0 or i == iters - 1:  # print logs
            probs_clf = []

            # For targeted attack (10 targets), only show results for v=7; for untargeted, v=0
            for clf_name in classifier_list:
                probs_clf.append(f"{clf_name}: y({imagenet_labels[pred_idx[v,0].item()]})={p[v, 0]:.4f}")
            probs_str = " | ".join(probs_clf)
            print(
                f'iter {i:>3}/{iters} | adv_loss = {total_adv_loss.item():<9.4f} | col_loss = {col_loss.item():<9.4f} | '
                f'prjl2 = {prjl2.mean().item() * 255:<9.4f} | caml2 = {caml2.mean().item() * 255:<9.4f} | '
                f'camdE = {camdE.mean().item():<9.4f} | {probs_str}')

    # --- Final stealthiness improvement loop ---
    # For projection whose camdE is greater than d_thr, its stealthiness will be forcibly enhanced.
    while torch.any(camdE > d_thr):
        col_lr = 0.2
        cam_infer = pcnet(torch.clamp(expand_4d(prj_adv), 0, 1), cam_scene_batch)



        prjl2 = torch.norm(im_gray - prj_adv, dim=1).mean(1).mean(1)  # 投影仪 L2 损失
        col_loss_batch = prjl2_w * prjl2

        caml2 = torch.norm(cam_scene_batch - cam_infer, dim=1).mean(1).mean(1)  # 相机 L2 损失
        col_loss_batch += caml2_w * caml2

        camdE = ciede2000_diff(rgb2lab_diff(cam_infer, device), rgb2lab_diff(cam_scene_batch, device), device).mean(
            1).mean(1)
        col_loss_batch += camdE_w * camdE

        col_loss = col_loss_batch.mean()

        mask_high_pert = (camdE > d_thr)

        col_loss.backward()
        col_grad = prj_adv.grad.clone()
        prj_adv.grad.zero_()

        norm_col_grad = col_grad / (
                torch.norm(col_grad.view(num_target, -1), dim=1, keepdim=True).view(num_target, 1, 1,
                                                                                    1) + 1e-8)
        # Update samples with excessively high perturbation
        prj_adv.data[mask_high_pert] -= col_lr * cam_update_factor * norm_col_grad[mask_high_pert]
        # then try to set the best
        prj_adv_best = prj_adv.clone()
        cam_infer_best = cam_infer.clone()

        if i % 10 == 0 or i == iters - 1:
            probs_clf = []
            for clf_name in classifier_list:
                probs_clf.append(f"{clf_name}: y({imagenet_labels[pred_idx[v,0].item()]})={p[v, 0]:.4f}")
            probs_str = " | ".join(probs_clf)
            print(
                f'iter {i:>3}/{iters} | adv_loss = {total_adv_loss.item():<9.4f} | col_loss = {col_loss.item():<9.4f} | '
                f'prjl2 = {prjl2.mean().item() * 255:<9.4f} | caml2 = {caml2.mean().item() * 255:<9.4f} | '
                f'camdE = {camdE.mean().item():<9.4f} | {probs_str}')

    prj_adv_best = torch.clamp(prj_adv_best, 0, 1)  # Clamp the final best projection to [0, 1] range


    warped_prj = F.grid_sample(prj_adv_best,
                               fine_grid[0, :].unsqueeze(0).expand(prj_adv_best.shape[0], -1, -1, -1),
                               align_corners=True)
    warped_prj_mask = pcnet.module.get_mask().float().unsqueeze(0).unsqueeze(0)
    warped_prj = warped_prj * warped_prj_mask

    return cam_infer_best, prj_adv_best, warped_prj





def summarize_single_universal_attacker(attacker_name, data_root, setup_list, device='cuda', device_ids=[0], pose = 'original',classifier_names = ['inception_v3', 'resnet18', 'vgg16']):
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
    columns = ['Setup', 'Attacker', 'Stealth_loss', 'd_thr', 'Classifier', 'T.top-1_infer', 'T.top-5_infer', 'T.top-1_real', 'T.top-5_real',
               'U.top-1_infer', 'U.top-1_real'] + [phase[0] + '_' + y for y in metrics] +\
              ['T.' + x + '_' + y for x in phase[1:] for y in metrics] + ['U.' + x + '_' + y for x in phase[1:] for y in metrics] + \
              ['All.' + x + '_' + y for x in phase[1:] for y in metrics]

    # stealth_losses = ['caml2', 'camdE', 'camdE_caml2', 'camdE_caml2_prjl2', '-']
    stealth_losses   = ['camdE_caml2']
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
                    cam_infer = ut.torch_imread_mt(cam_infer_path).to(device)
                    warped_prj_adv = ut.torch_imread_mt(warped_prj_adv_path).to(device)

                    ret = {}  # classification result dict

                    for classifier_name in classifier_names:

                        # check whether all images are captured for results summary
                        dirs_to_check = [prj_adv_path, cam_real_path]
                        skip = False
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
                            classifier = Classifier(classifier_name, device, device_ids, fix_params=True)
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

                            # untargeted [n]
                            *calc_img_dists(prj_adv[n, None], im_gray.expand_as(prj_adv[n, None])),  # prj adv
                            *calc_img_dists(cc(cam_infer[n, None], cp_sz),
                                            cc(cam_scene, cp_sz).expand_as(cc(cam_infer[n, None], cp_sz))),  # cam infer
                            *calc_img_dists(cc(cam_real[n, None], cp_sz),
                                            cc(cam_scene, cp_sz).expand_as(cc(cam_real[n, None], cp_sz))),  # cam real

                            # both targeted and untargeted [0, n].
                            *calc_img_dists(prj_adv, im_gray.expand_as(prj_adv)),  # prj adv
                            # cam infer
                            *calc_img_dists(cc(cam_infer, cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_infer, cp_sz))),
                            # cam real
                            *calc_img_dists(cc(cam_real, cp_sz), cc(cam_scene, cp_sz).expand_as(cc(cam_real, cp_sz)))
                        ]

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
            if attacker_name in ['SPAA', 'CAPAA (classifier-specific)']:
                if recreate_stats_and_imgs:
                    summarize_single_attacker_for_UAP_task(attacker_name=[attacker_name], data_root=data_root, setup_list=[setup_name], device='cuda', device_ids=[0],pose=pose)
                table.append(pd.read_csv(join(ret_path, 'universal_stats.txt'), index_col=None, header=0, sep='\t'))
            else:
                if recreate_stats_and_imgs:
                    summarize_single_universal_attacker(attacker_name=[attacker_name], data_root=data_root, setup_list=[setup_name], device='cuda', device_ids=[0],pose=pose)
                table.append(pd.read_csv(join(ret_path, 'stats.txt'), index_col=None, header=0, sep='\t'))

    table = pd.concat(table, axis=0, ignore_index=True)

    pivot_table = pd.pivot_table(table,
                                 # values=['T.top-1_real', 'T.top-5_real', 'U.top-1_real', 'T.real_L2', 'T.real_Linf', 'T.real_dE', 'T.real_SSIM'],
                                 values=['T.top-1_real', 'T.top-5_real', 'U.top-1_real', 'U.real_L2', 'U.real_Linf',
                                         'U.real_dE', 'U.real_SSIM', 'All.real_L2', 'All.real_Linf', 'All.real_dE',
                                         'All.real_SSIM'],
                                 index=['Attacker', 'd_thr', 'Stealth_loss', 'Classifier'], aggfunc=np.mean)
    pivot_table = pivot_table.sort_index(level=[0, 1], ascending=[False, True])  # to match CAPAA Table order

    # save tables
    table.to_csv(join(data_root, 'setups/stats_all.txt'), index=False, float_format='%.4f', sep='\t')
    table.to_excel(join(data_root, 'setups/stats_all.xlsx'), float_format='%.4f', index=False)
    pivot_table.to_excel(join(data_root, 'setups/pivot_table_all.xlsx'), float_format='%.4f', index=True)

    return table, pivot_table


def summarize_all_attackers_vit(attacker_names, data_root, setup_list, recreate_stats_and_imgs=False):
    """
    Given attacker_names and setup_list, summarize all attacks.
    Now reads from .xlsx files instead of .txt files.
    Simplified version with less try-except blocks, closer to original structure.

    :param attacker_names: List of attacker names.
    :param data_root: Root directory for data.
    :param setup_list: List of setup names.
    :param recreate_stats_and_imgs: When False, only gather all existing stats.xlsx/universal_stats.xlsx
                                     of all setups and create a pivot table [setup/pivot_table_all.xlsx].
                                     If True, it will first call functions to regenerate stats, which
                                     ideally should also be updated to produce .xlsx files.
    :return: Tuple (DataFrame of all stats, PivotTable DataFrame).
    """
    table = []

    for setup_name in tqdm(setup_list, desc="Processing setups"):
        setup_path = join(data_root, setup_name)
        for attacker_name in attacker_names:  # Removed tqdm here to be closer to original if it wasn't there
            # Assuming to_attacker_cfg_str is available in the global scope or imported
            # If not, this line will raise a NameError.
            attacker_cfg_str = to_attacker_cfg_str(attacker_name)[0]
            ret_path = join(setup_path, 'ret', attacker_cfg_str)
            print(f'Gathering stats from Excel files in: {ret_path}')  # Kept a basic print statement

            if 'original' in setup_name:
                pose = 'original'
            else:
                pose = 'changed'

            if attacker_name in ['SPAA', 'CAPAA (classifier-specific)']:
                stats_file_path = join(ret_path, 'universal_stats.xlsx')
                if recreate_stats_and_imgs:
                    # Assuming summarize_single_attacker_for_UAP_task is available
                    summarize_single_attacker_for_UAP_task(attacker_name=[attacker_name], data_root=data_root,
                                                           setup_list=[setup_name], device='cuda', device_ids=[0],
                                                           pose=pose)

                if os.path.exists(stats_file_path):
                    table.append(pd.read_excel(stats_file_path, index_col=None, header=0, engine='openpyxl'))
                else:
                    print(f"Warning: File not found {stats_file_path}. Skipping.")
            else:
                stats_file_path = join(ret_path, 'stats.xlsx')
                if recreate_stats_and_imgs:
                    # Assuming summarize_single_universal_attacker is available
                    summarize_single_universal_attacker(attacker_name=[attacker_name], data_root=data_root,
                                                        setup_list=[setup_name], device='cuda', device_ids=[0],
                                                        pose=pose)

                if os.path.exists(stats_file_path):
                    table.append(pd.read_excel(stats_file_path, index_col=None, header=0, engine='openpyxl'))
                else:
                    print(f"Warning: File not found {stats_file_path}. Skipping.")

    if not table:
        print("No dataframes were loaded. Returning empty DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    table_df = pd.concat(table, axis=0, ignore_index=True)

    pivot_values = ['T.top-1_real', 'T.top-5_real', 'U.top-1_real', 'U.real_L2', 'U.real_Linf',
                    'U.real_dE', 'U.real_SSIM', 'All.real_L2', 'All.real_Linf', 'All.real_dE',
                    'All.real_SSIM']

    # Check if all required index columns for pivot table exist
    required_index_cols = ['Attacker', 'd_thr', 'Stealth_loss', 'Classifier']
    if not all(idx_col in table_df.columns for idx_col in required_index_cols):
        missing_cols = [col for col in required_index_cols if col not in table_df.columns]
        print(
            f"Error: Missing one or more required index columns for pivot table: {missing_cols}. Cannot create pivot table.")
        pivot_table_df = pd.DataFrame()  # Return empty pivot table
    else:
        # Filter pivot_values to only include columns present in table_df
        available_pivot_values = [col for col in pivot_values if col in table_df.columns]
        if not available_pivot_values:
            print(
                "Error: None of the specified 'values' for the pivot table are present in the data. Cannot create pivot table.")
            pivot_table_df = pd.DataFrame()
        else:
            pivot_table_df = pd.pivot_table(table_df,
                                            values=available_pivot_values,
                                            index=required_index_cols,
                                            aggfunc=np.mean,
                                            dropna=False)
            pivot_table_df = pivot_table_df.sort_index(level=[0, 1], ascending=[False, True])

    output_dir = data_root
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    stats_all_txt_path = join(output_dir, 'stats_all.txt')
    stats_all_xlsx_path = join(output_dir, 'stats_all.xlsx')
    pivot_table_all_xlsx_path = join(output_dir, 'pivot_table_all.xlsx')

    table_df.to_csv(stats_all_txt_path, index=False, float_format='%.4f', sep='\t')
    table_df.to_excel(stats_all_xlsx_path, float_format='%.4f', index=False, engine='openpyxl')

    if not pivot_table_df.empty:
        pivot_table_df.to_excel(pivot_table_all_xlsx_path, float_format='%.4f', index=True, engine='openpyxl')
    else:
        print("Pivot table is empty, not saving pivot_table_all.xlsx.")

    return table_df, pivot_table_df
