from typing import List
import os
from IPython.display import clear_output
import cv2
import uuid
import time
import json


class imageLabel(object):
    def __init__(self, configs_path: str = 'configs/labelImageConfigs.json'):
        self._configs_path = configs_path

    def get_label_image_confs(self):
        with  open(self._configs_path) as f:
            data = json.load(f)
        label_image_configs = imageLabel()
        label_image_configs.__dict__ = data

        return label_image_configs

    def installations(self, installations_commands_path: str, LABELING_PATH):
        with open(installations_commands_path) as f:
            commandes = [l.replace('\n', '') for l in f.readlines() if '#' not in l]

        for c in commandes:
            os.system(f"{c}")
            clear_output()

        if os.name == 'posix':
            os.system(f"cd {LABELING_PATH} make qt5py3")

        if os.name == 'nt':
            os.system(f"cd {LABELING_PATH} pyrcc5 -o libs/resources.py resources.qrc")

    def set_folders(self, image_path: str, labels: List[str], lablimg_path: str):
        if not os.path.exists(image_path):
            if os.name == 'posix':
                os.system(f"mkdir -p {image_path}")
            if os.name == 'nt':
                os.system(f"mkdir  {image_path}")

        labels = labels + ['label_image_configs.json', 'train']
        for label in labels:
            path = os.path.join(image_path, label)
            if not os.path.exists(path):
                os.system(f"mkdir -p {path}")

        os.system(f"mkdir -p {lablimg_path}")

    def capture_images_using_webcame(self, labels, image_path):
        number_imgs = 5
        for label in labels:
            cap = cv2.VideoCapture(0)
            print('Collecting images for {}'.format(label))
            time.sleep(5)
            for imgnum in range(number_imgs):
                print('Collecting image {}'.format(imgnum))
                ret, frame = cap.read()
                imgname = os.path.join(image_path, label, label + '.' + '{}.jpg'.format(str(uuid.uuid1())))
                cv2.imwrite(imgname, frame)
                cv2.imshow('frame', frame)
                time.sleep(2)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

    def open_labelimg(self, path):
        os.system(f'cd {path} && python labelImg.py')

    def compress_images(self):
        TRAIN_PATH = os.path.join('TensorFlow', 'workspace', 'images', 'collectedimages', 'train')
        TEST_PATH = os.path.join('TensorFlow', 'workspace', 'images', 'collectedimages', 'label_image_configs.json')
        ARCHIVE_PATH = os.path.join('TensorFlow', 'workspace', 'images', 'archive.tar.gz')
        os.system(f'tar -czf {ARCHIVE_PATH} {TRAIN_PATH} {TEST_PATH}')
