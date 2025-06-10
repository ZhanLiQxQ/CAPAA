## Usage

1. Download the [CAPAA dataset][1].
2. Extract the downloaded archive. You should find two main folders: `setups` and `prj_share`.
3. Follow the subsequent steps in the main [README.md](../README.md), continuing from *STEP 4*.

After these steps, your `data/` directory structure should look approximately like this:


```
 data
    ├── prj_share                               # Shared projector patterns for all setups
    │   ├── numbers                             # Patterns for ProCam synchronization test
    │   ├── test                                # Test patterns for PCNet/model training
    │   └── train                               # Training patterns for PCNet/model training
    ├── setups                                  # Parent folder for all experimental setups
    │   └── teddy_3_60                          # An example setup name (e.g., teddy_3_60,basketball_original)
    │       ├── cam                             # Camera-captured images for this 'teddy' setup
    │       │   ├── infer
    │       │   │   ├── adv                     # Inferred images of adversarial projections
    │       │   │   │   ├── CAPAA_PCNet_l1+ssim_500_24_2000 # Attacker: CAPAA
    │       │   │   │   │   └── camdE_caml2     # Stealthiness loss
    │       │   │   │   │       ├── 2           # d_thr = 2
    │       │   │   │   │       │   └── classifier_all
    │       │   │   │   │       ├── 3
    │       │   │   │   │       │   └── classifier_all
    │       │   │   │   │       ├── 4
    │       │   │   │   │       │   └── classifier_all
    │       │   │   │   │       └── 5
    │       │   │   │   │           └── classifier_all
    │       │   │   │   ├── CAPAA (classifier-specific)_PCNet_l1+ssim_500_24_2000 # Attacker: CAPAA (classifier-specific)
    │       │   │   │   │   └── camdE_caml2
    │       │   │   │   │       ├── 2
    │       │   │   │   │       │   ├── inception_v3
    │       │   │   │   │       │   ├── resnet18
    │       │   │   │   │       │   └── vgg16
    │       │   │   │   │       ├── 3
    │       │   │   │   │       │   ├── inception_v3
    │       │   │   │   │       │   ├── resnet18
    │       │   │   │   │       │   └── vgg16
    │       │   │   │   │       ├── 4
    │       │   │   │   │       │   ├── inception_v3
    │       │   │   │   │       │   ├── resnet18
    │       │   │   │   │       │   └── vgg16
    │       │   │   │   │       └── 5
    │       │   │   │   │           ├── inception_v3
    │       │   │   │   │           ├── resnet18
    │       │   │   │   │           └── vgg16
    │       │   │   │   ├── SPAA_PCNet_l1+ssim_500_24_2000 # Attacker: SPAA
    │       │   │   │   │   └── camdE_caml2
    │       │   │   │   │       └── ... (structure as 'CAPAA_PCNet_l1+ssim_500_24_2000')
    │       │   │   │   └── CAPAA (without attention)_PCNet_l1+ssim_500_24_2000 # Attacker: CAPAA (w/o attention)
    │       │   │   │       └── camdE_caml2
    │       │   │   │           └── ... (structure as 'CAPAA_PCNet_l1+ssim_500_24_2000')
    │       │   │   └── test                    # Inferred clean test images
    │       │   │       └── PCNet_l1+ssim_500_24_2000
    │       │   └── raw                         # Real (physically captured) camera images
    │       │       ├── adv                     # Real captures of adversarial projections (mirrors cam\infer\adv structure)
    │       │       ├── cb                      # Captured checkerboard patterns
    │       │       ├── ref                     # Captured reference illuminations
    │       │       ├── test                    # Captured clean test patterns
    │       │       └── train                   # Captured training patterns
    │       ├── prj                             # Projector input images (patterns to be projected)
    │       │   ├── adv                         # Generated adversarial patterns (mirrors cam\infer\adv structure)
    │       │   ├── raw                         # Projector patterns for calibration/reference
    │       │   │   ├── cb
    │       │   │   └── ref                     
    │       │   └── warped_adv                  # Adversarial patterns warped to camera view (mirrors prj\adv structure)
    │       ├── ret                             # Attack results (output images, statistics for this 'teddy' setup)
    │       │   ├── CAPAA_PCNet_l1+ssim_500_24_2000 # Results for CAPAA
    │       │   │   └── camdE_caml2
    │       │   │       └── ... (structure mirrors cam\infer\adv for different d_thr, classifiers)
    │       │   ├── CAPAA (classifier-specific)_PCNet_l1+ssim_500_24_2000 # Results for CAPAA (classifier-specific)
    │       │   │   └── ...
    │       │   └── ... (similarly for SPAA, CAPAA (without attention))
    │       └── setup_info.yml                  # Setup-specific configuration for 'teddy_3_60'
    │   ├── ... (other setup folders like 'basketball_original', etc. follow the same structure as 'teddy_3_60')
    │   ├── supplementary_results_for_vit_b_16/ # Supplementary results including attacks against ViT-B/16 classifier
    │   ├──pivot_table_all.xlsx
    │   └──stats_all.xlsx                   
    ├── imagenet1000_clsidx_to_labels.txt   # ImageNet 1000 class index to labels mapping
    ├── imagenet10_clsidx_to_labels.txt     # ImageNet 10 (subset) for targeted attacks
    └── README.md                           # this file
```
## Citation

```bibtex
@inproceedings{li2025capaa,
  title={CAPAA: Classifier-Agnostic Projector-based Adversarial Attacks},
  author={Li, Zhan and Zhao, Mingyu and Dong, Xin and Ling, Haibin and Huang, Bingyao},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)},
  year={2025}
}
```

## Acknowledgments

- We thank the anonymous reviewers for their valuable and inspiring comments and suggestions.
- We thank the authors of any publicly available colorful textured sampling images used during experiments.
- Feel free to open an issue if you have any questions, suggestions, or concerns🥺🥹.

[1]: https://drive.google.com/file/d/1Kte3lONV2kRgg1hZtRr8ws503jq6JrPU/view?usp=sharing
