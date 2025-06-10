"""
Script to reproduce CAPAA (IEEE ICME'25) paper results on the benchmark dataset (10 * 7 setups).

1. We start by setting the training environment to GPU (if any).
2. Setups are listed in 'setup_list', more setups can be found in CAPAA/data/setups folder.
3. Run the script by `python reproduce_capaa_results.py`, the progress will be updated in console.
4. Upon finish, each setup's results will be saved to [data_root]/setups/[setup_name]/ret, [data_root]/setups/pivot_table_all.xlsx and [data_root]/setups/supplementary_results_for_vit_b_16/pivot_table_all.xlsx.
5. To perform real projector-based attacks, refer to `main.py`.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from os.path import join, abspath
from projector_based_attack_all import summarize_all_attackers_vit, summarize_all_attackers
# %% (1) [local or server] Summarize results of all projector-based attackers (CAPAA, CAPAA(w/o attention), CAPAA(classifier-specific),SPAA) on all setups

data_root = abspath(join(os.getcwd(), '../../data'))

setup_list = [
'basketball_original', 'basketball_3_60', 'basketball_3_75', 'basketball_3_105', 'basketball_3_120', 'basketball_zoomin5mm', 'basketball_zoomout5mm',
'backpack_original', 'backpack_3_60', 'backpack_3_75', 'backpack_3_105', 'backpack_3_120', 'backpack_zoomin5mm', 'backpack_zoomout5mm',
'envelope_original', 'envelope_3_60', 'envelope_3_75', 'envelope_3_105', 'envelope_3_120', 'envelope_zoomin5mm', 'envelope_zoomout5mm',
'lotion_original', 'lotion_3_60', 'lotion_3_75', 'lotion_3_105', 'lotion_3_120', 'lotion_zoomin5mm', 'lotion_zoomout5mm',
'packet_original', 'packet_3_60', 'packet_3_75', 'packet_3_105', 'packet_3_120', 'packet_zoomin5mm', 'packet_zoomout5mm',
'paper_towel_original','paper_towel_3_60','paper_towel_3_75','paper_towel_3_105','paper_towel_3_120', 'paper_towel_zoomin5mm', 'paper_towel_zoomout5mm',
'sunscreen_original','sunscreen_3_60','sunscreen_3_75','sunscreen_3_105','sunscreen_3_120', 'sunscreen_zoomin5mm', 'sunscreen_zoomout5mm',
'teddy_original','teddy_3_60','teddy_3_75','teddy_3_105','teddy_3_120', 'teddy_zoomin5mm', 'teddy_zoomout5mm',
'crock pot_original','crock pot_3_60','crock pot_3_75','crock pot_3_105','crock pot_3_120','crock pot_zoomin5mm','crock pot_zoomout5mm',
'coffee mug_original','coffee mug_3_60','coffee mug_3_75','coffee mug_3_105','coffee mug_3_120','coffee mug_zoomin5mm','coffee mug_zoomout5mm',
]

attacker_names = ['SPAA','CAPAA (classifier-specific)','CAPAA (without attention)','CAPAA']
# If you want to retest the results by the given captured images, set 'recreate_stats_and_imgs=True'.
# If you just want to contact the already generated result tables of all the setups, set 'recreate_stats_and_imgs = False'.
# all_ret, pivot_table = summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs = False)
all_ret, pivot_table = summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs = True)

print(f'\n------------------ Pivot table of {len(setup_list)} setups in {data_root} ------------------')
print(pivot_table.to_string(index=True, float_format='%.4f'))


# %% (2) [local or server] Reproduce supplementary material results: attacks against 4 classifiers (ViT-B/16, ResNet-18, Inception-V3, VGG-16)
data_root = abspath(join(os.getcwd(), '../../data/setups/supplementary_results_for_vit_b_16'))
setup_list = [
'teddy1_original','teddy1_15','teddy1_30','teddy1_105','teddy1_120'
]

attacker_names = ['SPAA','CAPAA (classifier-specific)','CAPAA (without attention)','CAPAA']

# If you want to retest the results by the given captured images, set 'recreate_stats_and_imgs=True'.
# If you just want to contact the already generated result tables of all the setups, set 'recreate_stats_and_imgs = False'.
# all_ret, pivot_table = summarize_all_attackers(attacker_names, data_root, setup_list, recreate_stats_and_imgs = True)
all_ret, pivot_table = summarize_all_attackers_vit(attacker_names, data_root, setup_list, recreate_stats_and_imgs = False)


print(f'\n------------------ Pivot table of {len(setup_list)} setups in {data_root} ------------------')
print(pivot_table.to_string(index=True, float_format='%.4f'))