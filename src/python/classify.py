import os
import torch
import pandas as pd
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image
from classifier import Classifier, load_imagenet_labels
from os.path import join, abspath


def find_image_folders(root_folder):
    return [root for root, _, files in os.walk(root_folder)
            if any(f.lower().endswith('.png') for f in files)]


def get_last_image(folder_path):
    images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    return join(folder_path, images[-1]) if images else None


def process_classification(base_folder, subfolders, additional_paths, model_names, labels_path, device='cuda:0'):
    """Process classification and return results"""
    imagenet_labels = load_imagenet_labels(labels_path)
    transform = T.Compose([T.ToTensor()])

    # Initialize classifiers with progress bar
    classifiers = {name: Classifier(model_name=name, device=device, device_ids=[0], fix_params=True)
                   for name in tqdm(model_names, desc="Initializing classifiers")}

    # Store results for each method
    method_results = {prefix: [] for prefix in additional_paths.values()}

    # Process each method
    for additional_path, prefix in additional_paths.items():
        print(f"\nProcessing {prefix} data...")

        # Main progress bar for subfolders
        subfolder_pbar = tqdm(subfolders, desc=f"{prefix} subfolders", position=0, leave=True)

        for subfolder in subfolder_pbar:
            folder_path = join(base_folder, subfolder, additional_path)
            image_folders = find_image_folders(folder_path)

            # Image processing progress (nested, will overwrite)
            image_pbar = tqdm(image_folders, desc=f"Processing images", position=1, leave=False)

            for image_folder in image_pbar:
                image_path = get_last_image(image_folder)
                if image_path:
                    try:
                        image = Image.open(image_path).convert('RGB')
                        image_tensor = transform(image).unsqueeze(0).to(device)
                        row = {'SubFolder': subfolder}

                        # Model predictions
                        for model_name in model_names:
                            _, p_sorted, idx = classifiers[model_name](image_tensor, crop_sz=(240, 240))
                            predicted_label = imagenet_labels.get(int(idx[0][0]), "Unknown")
                            true_label = subfolder.lower().split('_')[0]
                            row[f'{model_name}_Result'] = 0 if predicted_label.lower() == true_label else 1

                        method_results[prefix].append(row)
                    except Exception as e:
                        print(f"\nError processing image: {image_path}, Error: {str(e)}")

            image_pbar.close()
        subfolder_pbar.close()

    return method_results


def generate_comparison_table(method_results, model_names):
    """Generate comparison table and save results to Excel with multiple sheets"""
    target_columns = [f'{m}_Result' for m in model_names]
    comparison_data = {}

    # Create Excel writer object
    output_path = abspath(join(os.getcwd(), '../../data/setups/classification_results.xlsx'))

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Save each method's detailed results to separate sheets
        for method, results in method_results.items():
            if results:
                df = pd.DataFrame(results)
                df.to_excel(writer, sheet_name=method, index=False)
                comparison_data[method] = df[target_columns].mean()

        # Save comparison table to a summary sheet
        comparison_df = pd.DataFrame(comparison_data).T
        comparison_df.columns = [col.replace('_Result', '') for col in comparison_df.columns]
        comparison_df.to_excel(writer, sheet_name='Comparison_Summary')

    return comparison_df


if __name__ == "__main__":
    # Configuration
    base_folder = abspath(join(os.getcwd(), '../../data/setups'))

    subfolders = [
        'basketball_original', 'basketball_3_60', 'basketball_3_75', 'basketball_3_105', 'basketball_3_120',
        'basketball_zoomin5mm', 'basketball_zoomout5mm',
        'backpack_original', 'backpack_3_60', 'backpack_3_75', 'backpack_3_105', 'backpack_3_120', 'backpack_zoomin5mm',
        'backpack_zoomout5mm',
        'envelope_original', 'envelope_3_60', 'envelope_3_75', 'envelope_3_105', 'envelope_3_120', 'envelope_zoomin5mm',
        'envelope_zoomout5mm',
        'lotion_original', 'lotion_3_60', 'lotion_3_75', 'lotion_3_105', 'lotion_3_120', 'lotion_zoomin5mm',
        'lotion_zoomout5mm',
        'packet_original', 'packet_3_60', 'packet_3_75', 'packet_3_105', 'packet_3_120', 'packet_zoomin5mm',
        'packet_zoomout5mm',
        'paper_towel_original', 'paper_towel_3_60', 'paper_towel_3_75', 'paper_towel_3_105', 'paper_towel_3_120',
        'paper_towel_zoomin5mm', 'paper_towel_zoomout5mm',
        'sunscreen_original', 'sunscreen_3_60', 'sunscreen_3_75', 'sunscreen_3_105', 'sunscreen_3_120',
        'sunscreen_zoomin5mm', 'sunscreen_zoomout5mm',
        'teddy_original', 'teddy_3_60', 'teddy_3_75', 'teddy_3_105', 'teddy_3_120', 'teddy_zoomin5mm',
        'teddy_zoomout5mm',
        'crock pot_original', 'crock pot_3_60', 'crock pot_3_75', 'crock pot_3_105', 'crock pot_3_120',
        'crock pot_zoomin5mm', 'crock pot_zoomout5mm',
        'coffee mug_original', 'coffee mug_3_60', 'coffee mug_3_75', 'coffee mug_3_105', 'coffee mug_3_120',
        'coffee mug_zoomin5mm', 'coffee mug_zoomout5mm',
    ]

    additional_paths = {
        r"cam\raw\adv\SPAA_PCNet_l1+ssim_500_24_2000\camdE_caml2": "SPAA",
        r"cam\raw\adv\CAPAA_PCNet_l1+ssim_500_24_2000\camdE_caml2": "CAPAA",
    }

    labels_path = abspath(join(os.getcwd(), '../../data/imagenet1000_clsidx_to_labels.txt'))
    model_names = ['mobilenet_v3_large', 'efficientnet_b0', 'convnext_base', 'swin_b']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print("Starting classification process...")
    results = process_classification(base_folder, subfolders, additional_paths, model_names, labels_path, device)

    print("\nGenerating comparison table and saving results...")
    comparison_table = generate_comparison_table(results, model_names)

    print("\nFinal comparison results:")
    print(comparison_table)
    print("\nResults saved to classification_results.xlsx with multiple sheets")