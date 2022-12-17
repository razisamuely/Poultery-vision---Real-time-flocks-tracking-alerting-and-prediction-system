import os
import random
import shutil
from collections import Counter
import numpy as np
from augmentation import augmentation
import time

switchs = {
    "adjust_saturation": {"run_augmentation": True, "copy_xml": True, "return_transformation": False},
    "stateless_random_contrast": {"run_augmentation": True, "copy_xml": True, "return_transformation": False},
    "minus_two_hundred": {"run_augmentation": True, "copy_xml": True, "return_transformation": False},
    "rgb_to_grayscale": {"run_augmentation": True, "copy_xml": True, "return_transformation": False},
    "zebra_aug": {"run_augmentation": True, "copy_xml": True, "return_transformation": False},
    "BlendAlphaSimplexNoise_a": {"run_augmentation": True, "copy_xml": True, "return_transformation": False},
    "BlendAlphaSimplexNoise_b": {"run_augmentation": False, "copy_xml": True, "return_transformation": False},
    "CoarseDropout": {"run_augmentation": True, "copy_xml": True, "return_transformation": False},
    "AdditiveLaplaceNoise": {"run_augmentation": True, "copy_xml": True, "return_transformation": False},
    "ReplaceElementwise": {"run_augmentation": False, "copy_xml": True, "return_transformation": False},
    "Fliplr": {"run_augmentation": True, "copy_xml": "horizontally", "return_transformation": False},
    "Flipud": {"run_augmentation": True, "copy_xml": "vertically", "return_transformation": False},
    "reduce_ractangle": {"run_augmentation": False, "copy_xml": "reduce_ractangle", "return_transformation": False}
}
#
# Copy source files_images before augmentation

folder_list = [
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-09_growing_day=0/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-09_growing_day=0/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-10_growing_day=1/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-10_growing_day=1/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-11_growing_day=2/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-11_growing_day=2/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-12_growing_day=3/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-12_growing_day=3/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-13_growing_day=4/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-13_growing_day=4/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-14_growing_day=5/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-14_growing_day=5/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-15_growing_day=6/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-15_growing_day=6/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-16_growing_day=7/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-16_growing_day=7/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-17_growing_day=8/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-17_growing_day=8/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-18_growing_day=9/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-18_growing_day=9/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-19_growing_day=10/camera=201",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-19_growing_day=10/camera=301",
    "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-20_growing_day=11/camera=201",
    "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-21_growing_day=12/camera=201",
    "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-22_growing_day=13/camera=201",
    "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-23_growing_day=14/camera=201",
    "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-24_growing_day=15/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-25_growing_day=16/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-26_growing_day=17/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-27_growing_day=18/camera=301",
    # "/Users/raz.shmuely/Documents/privet/chickens/images_for_labeling/NirIsrael/2022-05/date=2022-05-28_growing_day=19/camera=301"
]

# 1 ---  Copy labeled images only to current project ---
# list of files with xml detection file
# files = {d: [k for k, v in Counter([f.split('.')[0] for f in os.listdir(d)]).items() if v > 1] for d in folder_list}
# files_to_copy = [[f"{d}/{f}" for f in os.listdir(d) if f.split('.')[0] in fl] for d, fl in files.items()]
# files_to_copy = [item for sublist in files_to_copy for item in sublist]
# #
# output_raw_images_folder = 'labeld_images_and_xml'
# for f in files_to_copy[:]:
#     input_path = f
#     output_path = f"{output_raw_images_folder}/{f.split('/')[-1]}"
#     shutil.copyfile(input_path, output_path)
#     time.sleep(0.1)

# %%
# # 2 ---  Copy labeled images to augmented file ---
input_folder = 'labeld_images_and_xml'
output_folder = 'augmented_images_and_xml'
files_images = [f for f in os.listdir(input_folder) if ("png" in f) or ("jpeg" in f) or ("jpg" in f) or ("xml" in f)]
for f in files_images[:]:
    inputh_path = f"{input_folder}/{f}"
    output_path = f"{output_folder}/{f}"
    shutil.copyfile(inputh_path, output_path)

# # %%
# # 3 ---  Sample random subset for augmentation ---
input_folder = 'augmented_images_and_xml'
output_folder = 'augmented_images_and_xml'
files_images = [f for f in os.listdir(input_folder) if ("png" in f) or ("jpeg" in f) or ("jpg" in f) or ("xml" in f)]

items_to_sample = int(len(files_images) / 2)
print(f"\nlen beofre sampeling = {len(files_images)}")
files_images = random.sample(files_images, items_to_sample)
print(f"len after sampeling = {len(files_images)}\n")

# # 4 ---  Iterate and execute augmentation ---
aug = augmentation()
failures = {"files_images": [], "counter": 0}

for i, f in enumerate(files_images):
    print(len(files_images) - i)
    point = f.index('.')
    file_name = f[:point]
    typ = f[point + 1:]
    try:
        aug.copy_xml_paths_names(input_folder=input_folder,
                                 output_folder=output_folder,
                                 file_name=file_name,
                                 typ="png")

        transforms = aug.augment_image(folder_path=input_folder,
                                       output_folder_path=output_folder,
                                       file_name=file_name,
                                       typ=typ,
                                       switchs=switchs)
        pass
    except Exception as e:
        failures["files_images"].append(f)
        failures["counter"] += 1
        print(f"\n====Failure {f}")
        print(e)

print(failures)

# # 5 ---  Copy Augmented images to training file--
input_folder = 'augmented_images_and_xml'
output_folder = "/Users/raz.shmuely/Documents/privet/chickens/install_object_detection_api/TensorFlow/workspace/drinking-images_3/train/"

files_images = [f for f in os.listdir(input_folder) if ("png" in f) or ("jpeg" in f) or ("jpg" in f) or ("xml" in f)]
for f in files_images:
    inputh_path = f"{input_folder}/{f}"
    output_path = f"{output_folder}/{f}"
    shutil.copyfile(inputh_path, output_path)




