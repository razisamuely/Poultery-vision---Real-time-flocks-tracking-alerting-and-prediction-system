import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import xml.etree.ElementTree as ET
from imgaug import augmenters as iaa


class augmentation(object):
    def __init__(self):
        pass

    def minus_two_hundred(self, image, scale: int = 255):
        return scale - image

    def zebra_aug(self, image, lines_number):
        # Find lower number for deviation without reminder
        for i in range(lines_number, lines_number * 4):
            if image.shape[1] % i == 0:
                lines_number = i
                break

        arr = [[2, 2, 2] for i in range(int(image.shape[1] / lines_number))]
        for j in range(lines_number - 1):
            if j % 2 == 0:
                u = 1
            else:
                u = 2
            arr += [[u, u, u] for i in range(int(image.shape[1] / lines_number))]
        transdformed_image = image * np.array(arr)

        return transdformed_image

    def visualize(self, original, augmented, f=None):
        plt.subplot(1, 2, 1)
        plt.title('Original image')
        plt.imshow(original)
        plt.subplot(1, 2, 2)
        plt.title(f'Augmented image - {f}')
        plt.imshow(augmented)

    def save_augmented_image_and_copy_labeling_xml(self,
                                                   augmented_image: np.array,
                                                   augmentation_type: str,
                                                   output_folder: str,
                                                   file_name: str,
                                                   typ: str):

        file = f"{file_name}_{augmentation_type}.{typ}"
        full_path = os.path.join(output_folder, file)
        tf.keras.utils.save_img(
            path=full_path,
            x=augmented_image,
            data_format=None,
            file_format=None,
            scale=True
        )

    def copy_xml_paths_names(self,
                             input_folder: str,
                             output_folder: str,
                             file_name: str,
                             typ: str
                             ):

        # reading xml
        xml_input_path = f"{input_folder}/{file_name}.xml"
        xml_output_path = f"{output_folder}/{file_name}.xml"
        mytree = ET.parse(xml_input_path)
        myroot = mytree.getroot()

        for folder in myroot.iter('folder'):
            folder.text = input_folder

        for filename in myroot.iter('filename'):
            file_name = f"{file_name}.{typ}"
            filename.text = file_name

        for path in myroot.iter('path'):
            wd = os.getcwd()
            path_name = os.path.join(wd, input_folder, file_name)
            path.text = path_name

        if not os.path.exists(output_folder):
            os.makedirs(name=output_folder)

        mytree.write(file_or_filename=xml_output_path)

    def copy_xml_horizontal_augmentation(self,
                                         xml_input_path: str,
                                         xml_output_path: str):
        mytree = ET.parse(xml_input_path)
        myroot = mytree.getroot()

        for width in myroot.iter('width'):
            width = int(width.text)

        x_shape = width
        # iterating through the price values.
        for xmin, xmax in zip(myroot.iter('xmin'), myroot.iter('xmax')):
            xmin_t = x_shape - int(xmax.text)
            xmax_t = x_shape - int(xmin.text)

            xmax.text = str(int(xmax_t))
            xmin.text = str(int(xmin_t))

        mytree.write(xml_output_path)

    def copy_xml_vertical_augmentation(self,
                                       xml_input_path: str,
                                       xml_output_path: str
                                       ):
        mytree = ET.parse(xml_input_path)
        myroot = mytree.getroot()

        for height in myroot.iter('height'):
            height = int(height.text)

        y_shape = height
        # iterating through the price values.
        for ymin, ymax in zip(myroot.iter('ymin'), myroot.iter('ymax')):
            ymin_t = y_shape - int(ymax.text)
            ymax_t = y_shape - int(ymin.text)

            ymax.text = str(int(ymax_t))
            ymin.text = str(int(ymin_t))

        mytree.write(xml_output_path)

    def reduce_ractangle(self, xml_input_path: str, xml_output_path: str, y: int, x: int):
        mytree = ET.parse(xml_input_path)
        myroot = mytree.getroot()

        # iterating through the price values.
        for ymin, ymax in zip(myroot.iter('ymin'), myroot.iter('ymax')):
            ymax.text = str(int(ymax.text) - y)
            ymin.text = str(int(ymin.text) + y)

        for xmin, xmax in zip(myroot.iter('xmin'), myroot.iter('xmax')):
            xmax.text = str(int(int(xmax.text) - x))
            xmin.text = str(int(int(xmin.text) + x))

        mytree.write(xml_output_path)

    def augment_image(self, folder_path: str, file_name: str, typ: str, output_folder_path: str, switchs: dict):
        transforms = {}
        # Read file
        file = f"{file_name}.{typ}"
        full_path = os.path.join(folder_path, file)

        if not os.path.exists(output_folder_path):
            os.makedirs(name=output_folder_path)

        # Convert to tensor and cast values to int
        Load_image = tf.keras.preprocessing.image.load_img(full_path)
        image = tf.keras.preprocessing.image.img_to_array(Load_image)
        image = tf.cast(image, tf.int32)

        augmentation_type = "adjust_saturation"
        if switchs.get(augmentation_type).get("run_augmentation"):
            transforms[augmentation_type] = tf.image.adjust_saturation(image, 3)

        augmentation_type = "stateless_random_contrast"
        if switchs.get(augmentation_type).get("run_augmentation"):
            seed = (4, 0)
            transforms[augmentation_type] = tf.image.stateless_random_contrast(
                image, lower=0.1, upper=0.9, seed=seed)

        augmentation_type = "minus_two_hundred"
        if switchs.get(augmentation_type).get("run_augmentation"):
            seed = (4, 0)
            transforms[augmentation_type] = self.minus_two_hundred(image)

        augmentation_type = "rgb_to_grayscale"
        if switchs.get(augmentation_type).get("run_augmentation"):
            transforms[augmentation_type] = tf.image.rgb_to_grayscale(image)

        augmentation_type = "zebra_aug"
        if switchs.get(augmentation_type).get("run_augmentation"):
            transforms[augmentation_type] = self.zebra_aug(image=image, lines_number=80)

        augmentation_type = "BlendAlphaSimplexNoise_a"
        if switchs.get(augmentation_type).get("run_augmentation"):
            f = iaa.BlendAlphaSimplexNoise(
                foreground=iaa.BlendAlphaSimplexNoise(
                    foreground=iaa.EdgeDetect(1.0),
                    background=iaa.LinearContrast((0.5, 2.0)),
                    per_channel=True
                ),
                background=iaa.BlendAlphaFrequencyNoise(
                    exponent=(-2.5, -1.0),
                    foreground=iaa.Affine(
                        translate_px={"x": (-4, 4), "y": (-4, 4)}
                    ),
                    background=iaa.AddToHueAndSaturation((-40, 40)),
                    per_channel=True
                ),
                per_channel=True,
                aggregation_method="max",
                sigmoid=False
            )
            seq = iaa.Sequential([f], random_order=True)
            img = image.numpy().astype(np.uint8)
            images_aug = seq.augment_images([img])
            transforms[augmentation_type] = images_aug[0]

        augmentation_type = "BlendAlphaSimplexNoise_b"
        if switchs.get(augmentation_type).get("run_augmentation"):
            f = iaa.BlendAlphaSimplexNoise(
                foreground=iaa.BlendAlphaSimplexNoise(
                    foreground=iaa.EdgeDetect(1.0),
                    background=iaa.LinearContrast((0.5, 2.0)),
                    per_channel=True
                ),

                background=iaa.BlendAlphaFrequencyNoise(
                    exponent=(-2.5, -1.0),
                    foreground=iaa.Affine(
                        translate_px={"x": (-4, 4), "y": (-4, 4)}
                    ),
                    background=iaa.AddToHueAndSaturation((-40, 40)),
                    per_channel=True
                ),
                per_channel=True,
                aggregation_method="max",
                sigmoid=False
            )
            seq = iaa.Sequential([f], random_order=True)
            img = image.numpy().astype(np.uint8)
            images_aug = seq.augment_images([img])
            transforms[augmentation_type] = images_aug[0]

        augmentation_type = "CoarseDropout"
        if switchs.get(augmentation_type).get("run_augmentation"):
            f = iaa.CoarseDropout(
                (0.03, 0.15), size_percent=(0.02, 0.05),
                per_channel=0.2
            )
            seq = iaa.Sequential([f], random_order=True)
            img = image.numpy().astype(np.uint8)
            images_aug = seq.augment_images([img])
            transforms[augmentation_type] = images_aug[0]

        augmentation_type = "AdditiveLaplaceNoise"
        if switchs.get(augmentation_type).get("run_augmentation"):
            f = iaa.AdditiveLaplaceNoise(scale=0.2 * 255, per_channel=True)
            seq = iaa.Sequential([f], random_order=True)
            img = image.numpy().astype(np.uint8)
            images_aug = seq.augment_images([img])
            transforms[augmentation_type] = images_aug[0]

        augmentation_type = "ReplaceElementwise"
        if switchs.get(augmentation_type).get("run_augmentation"):
            import imgaug.parameters as iap
            f = iaa.ReplaceElementwise(
                iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
                iap.Normal(128, 0.4 * 128),
                per_channel=0.5)
            seq = iaa.Sequential([f], random_order=True)
            img = image.numpy().astype(np.uint8)
            images_aug = seq.augment_images([img])
            transforms[augmentation_type] = images_aug[0]

        augmentation_type = "Fliplr"
        if switchs.get(augmentation_type).get("run_augmentation"):
            f = iaa.Fliplr()
            seq = iaa.Sequential([f], random_order=True)
            img = image.numpy().astype(np.uint8)
            images_aug = seq.augment_images([img])
            transforms[augmentation_type] = images_aug[0]

        augmentation_type = "Flipud"
        if switchs.get(augmentation_type).get("run_augmentation"):
            f = iaa.Flipud()
            seq = iaa.Sequential([f], random_order=True)
            img = image.numpy().astype(np.uint8)
            images_aug = seq.augment_images([img])
            transforms[augmentation_type] = images_aug[0]

        augmentation_type = "reduce_ractangle"
        if switchs.get(augmentation_type).get("run_augmentation"):
            transforms[augmentation_type] = image

        keys = list(transforms.keys())
        for augmentation_type in keys:
            v = switchs[augmentation_type]

            if v.get("copy_xml") == True:
                os.system(
                    f"cp  {folder_path}/{file_name}.xml  {output_folder_path}/{file_name}_{augmentation_type}.xml")

            if v.get("copy_xml") == 'horizontally':
                self.copy_xml_horizontal_augmentation(xml_input_path=f'{folder_path}/{file_name}.xml',
                                                      xml_output_path=f'{output_folder_path}/{file_name}_{augmentation_type}.xml')

            elif v.get("copy_xml") == 'vertically':
                self.copy_xml_vertical_augmentation(xml_input_path=f'{folder_path}/{file_name}.xml',
                                                    xml_output_path=f'{output_folder_path}/{file_name}_{augmentation_type}.xml')

            elif v.get("copy_xml") == 'reduce_ractangle':
                self.reduce_ractangle(xml_input_path=f'{folder_path}/{file_name}.xml',
                                      xml_output_path=f'{output_folder_path}/{file_name}_{augmentation_type}.xml',
                                      x=20,
                                      y=20)

            transforms[augmentation_type] = tf.convert_to_tensor(transforms[augmentation_type])
            self.save_augmented_image_and_copy_labeling_xml(augmented_image=transforms[augmentation_type],
                                                            output_folder=output_folder_path,
                                                            augmentation_type=augmentation_type,
                                                            file_name=file_name,
                                                            typ=typ
                                                            )
            print(f"{folder_path}/{file_name}_{augmentation_type}")

            self.copy_xml_paths_names(input_folder=output_folder_path,
                                      output_folder=output_folder_path,
                                      file_name=f"{file_name}_{augmentation_type}",
                                      typ=typ)

            if not v["return_transformation"]:
                transforms.pop(augmentation_type)

        return transforms
