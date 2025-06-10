"""
A wrapper for image classification models
"""
import torch
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision import models, transforms as T
from img_proc import resize, center_crop as cc, expand_4d
from vit_model import vit_base_patch16_224


# A general classifier class, can be initialized by model name
class Classifier(object):
    def __init__(self, model_name, device, device_ids, fix_params=True):
        super(Classifier, self).__init__()
        self.name = model_name
        self.fix_params = fix_params
        self.device = device


        # select model by name
        if self.name == 'vgg16':
            self.input_sz = (224, 224)
            self.model = getattr(models, self.name)(weights=None)  # torchvision >= 0.13
            pretrained_model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        elif self.name == 'resnet18':
            self.input_sz = (224, 224)
            self.model = getattr(models, self.name)(weights=None)  # torchvision >= 0.13
            pretrained_model_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
        elif self.name == 'inception_v3':
            self.input_sz = (299, 299)
            self.model = getattr(models, self.name)(init_weights=False, transform_input=True)  # must set transform_input=True to reproduce old inception v3
            # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)  # torchvision >= 0.13
            pretrained_model_url = 'https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth'
        elif self.name == 'vit_b_16':
            self.input_sz = (224, 224)
            self.model = vit_base_patch16_224()
            pretrained_model_url = './vit_base_patch16_224.pth'
        # load weights from pytorch pretrained models, to fully reproduce CAPAA result, we need to ensure the weights are exactly the same

            # unseen classifiers

        elif self.name == 'mobilenet_v3_large':
            self.input_sz = (224, 224)
            self.model = getattr(models, 'mobilenet_v3_large')(weights=None)
            pretrained_model_url = 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth'


        elif self.name == 'efficientnet_b0':
            self.input_sz = (224, 224)
            self.model = getattr(models, 'efficientnet_b0')(weights=None)
            pretrained_model_url = 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth'


        elif self.name == 'convnext_base':
            self.input_sz = (224, 224)
            self.model = getattr(models, 'convnext_base')(weights=None)
            pretrained_model_url = 'https://download.pytorch.org/models/convnext_base-6075fbad.pth'


        elif self.name == 'swin_b':
            self.input_sz = (224, 224)
            self.model = getattr(models, 'swin_b')(weights=None)
            pretrained_model_url = 'https://download.pytorch.org/models/swin_b-68c6b09e.pth'

        if self.name == 'vit_b_16':
            self.model.load_state_dict(torch.load(pretrained_model_url, map_location=device))
        else:
            self.model.load_state_dict(load_state_dict_from_url(pretrained_model_url))

        if len(device_ids) >= 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids).to(self.device)

        if self.fix_params:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # ImageNet's normalization and unnormalization (only works for 3D tensor image)
        normalize   = T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        unnormalize = UnNormalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))

        # works for 4D tensor
        self.normalize   = T.Lambda(lambda x: torch.stack([normalize(x[i]) for i in range(x.shape[0])]  , 0))
        self.unnormalize = T.Lambda(lambda x: torch.stack([unnormalize(x[i]) for i in range(x.shape[0])], 0))

    # classify a 4D float tensor
    def classify(self, im, crop_sz=(240, 240)):
        if im.dtype == torch.uint8:
            im = im.type(torch.float32) / 255

        im_transformed = self.normalize(resize(cc(expand_4d(im), crop_sz), self.input_sz))
        raw_score      = self.model(im_transformed.to(self.device))

        # raw score to probability using softmax
        # p = F.softmax(raw_score, dim=1)  # for 4D tensor, raw_score.shape = [B, 1000], thus softmax in dim=1
        p = F.softmax(raw_score, dim=1).detach().cpu()  # for 4D tensor, raw_score.shape = [B, 1000], thus softmax in dim=1

        # get sorted probability and indices
        p_sorted, idx = p.sort(descending=True)


        # return raw_score, p_sorted.numpy(), idx.numpy()  # return raw_score, p_sorted, idx
        return raw_score, p_sorted, idx

    def __call__(self, im, crop_sz):
        return self.classify(im, crop_sz)


# inverse transformation of torchvision.transforms.Normalize
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_in):
        tensor = tensor_in.clone()
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)  # The normalize code -> t.sub_(m).div_(s)
        return tensor


def query_multi_classifiers(im, cp_sz, classifier_names, imagenet_labels, device, device_ids):
    labels, confidence = [], []
    for classifier_name in classifier_names:
        print(f'Testing {classifier_name}')
        classifier = Classifier(classifier_name, device, device_ids, fix_params=True)
        _, p, _ = classifier(im, cp_sz)
        p = p.numpy()
        labels.append(imagenet_labels[p.argmax()])
        confidence.append(p.max())
        print(f'{classifier_name:<15}: {imagenet_labels[p.argmax()]:<20} ({p.max():.2f})')
    return labels, confidence


def load_imagenet_labels(filename):
    with open(filename) as f:
        imagenet_labels = eval(f.read())

    # simplify labels
    for k in imagenet_labels:
        imagenet_labels[k] = imagenet_labels[k].split(',')[0]
    return imagenet_labels
