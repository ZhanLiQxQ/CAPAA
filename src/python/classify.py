import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image
from classifier import Classifier, load_imagenet_labels


def find_image_folders(root_folder):
    """
    遍历指定文件夹，返回所有包含 PNG 文件的子文件夹路径。
    """
    image_folders = []
    for root, _, files in os.walk(root_folder):
        if any(file.lower().endswith('.png') for file in files):
            image_folders.append(root)
    return image_folders


def get_last_image(folder_path):
    """
    获取文件夹中最后一个 PNG 文件的路径 (按文件名排序)。
    """
    images = sorted([file for file in os.listdir(folder_path) if file.lower().endswith('.png')])
    return os.path.join(folder_path, images[-1]) if images else None


def classify_images(base_folder, subfolders, additional_paths, model_names, labels_path, output_folder, device='cuda:0', device_ids=[0]):
    """
    遍历 base_folder 下的所有子目录 (来自 subfolders 列表)，拼接 additional_paths，再对每个子文件夹的最后一张图片进行分类，将结果保存为 Excel 文件。
    """

    # 加载 ImageNet 标签
    imagenet_labels = load_imagenet_labels(labels_path)

    # 定义 transforms: 仅将 PIL 转换为 tensor (无 resize)
    transform = T.Compose([
        T.ToTensor()  # 转换为 [0, 1] 范围的 Tensor
    ])

    # 遍历每个 additional_path，生成两个表格
    for additional_path, prefix in additional_paths.items():
        # 创建 DataFrame 用于保存结果
        results = []

        # 遍历每个上级目录
        for subfolder in subfolders:
            # 拼接路径：base_folder + subfolder + additional_path
            folder_path = os.path.join(base_folder, subfolder, additional_path)

            print(folder_path)

            # 查找所有包含 PNG 文件的子文件夹
            image_folders = find_image_folders(folder_path)

            # 遍历每个子文件夹，添加进度条
            for image_folder in tqdm(image_folders, total=len(image_folders), desc=f"Processing {subfolder} - {prefix}"):
                image_path = get_last_image(image_folder)
                if image_path:
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image_tensor = transform(image).unsqueeze(0).to(device)
                        row = {'SubFolder': subfolder}

                        # 遍历分类器
                        for model_name in model_names:
                            # 初始化分类器
                            classifier = Classifier(model_name=model_name, device=device, device_ids=device_ids, fix_params=True, sort_results=True)

                            # 分类
                            raw_score, p_sorted, idx = classifier(image_tensor, crop_sz=(240, 240))

                            # 保存分类结果 (0 或 1) 和 概率
                            prediction = imagenet_labels[idx[0][0]]
                            probability = p_sorted[0][0].item()
                            row[f'{model_name}_Result'] = 0 if prediction.lower() == subfolder.lower().split('_')[0] else 1
                            row[f'{model_name}_Probability'] = probability

                        results.append(row)

                    except Exception as e:
                        print(f"Error processing image: {image_path}, Error: {str(e)}")

        # 保存为 Excel
        output_path = os.path.join(output_folder, f"{prefix}_classification_results.xlsx")
        df = pd.DataFrame(results)
        df.to_excel(output_path, index=False)
        print(f'\nClassification results saved to {output_path}')


if __name__ == "__main__":
    # 基目录路径
    # base_folder = "D:\spaa-new\data\setups"
    base_folder = "D:\CAPAA\SPAA"
    # 需要遍历的子目录列表
    subfolders = [
        # "crock pot original pose", "crock pot 3-60", "crock pot 3-75", "crock pot 3-105", "crock pot 3-120",
        # "crock pot zoom in", "crock pot zoom out"
        "coffee mug original pose","coffee mug 3-60","coffee mug 3-75","coffee mug 3-105","coffee mug 3-120","coffee mug zoom in","coffee mug zoom out"
    ]

    # 需要拼接的路径 (key: additional_path, value: filename prefix)
    additional_paths = {
        r"cam\raw\adv\original SPAA_SPAA_PCNet_l1+ssim_500_24_2000\camdE_caml2": "spaa",
        # r"cam\raw\adv\all_SPAA_PCNet_l1+ssim_500_24_2000\camdE_caml2": "all"
    }

    # 标签路径
    labels_path = "D:\SPAA-main\SPAA-main\data\imagenet1000_clsidx_to_labels.txt"

    # 输出文件夹路径
    output_folder = "D:\CAPAA"

    # 需要使用的分类器
    model_names = ['mobilenet_v3_large', 'efficientnet_b0', 'convnext_base', 'swin_b','resnet18','inception_v3','vgg16']

    # 确定设备
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Start")
    # 执行分类并保存结果
    classify_images(base_folder, subfolders, additional_paths, model_names, labels_path, output_folder, device=device, device_ids=[0])
    print("Finish")