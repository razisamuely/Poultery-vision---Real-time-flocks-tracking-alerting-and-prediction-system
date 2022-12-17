import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2
import numpy as np
from matplotlib import pyplot as plt

class DetectfromImage(object):

    def __init__(self, pipline_config_path: str, check_point_folder: str, check_point_number: int, label_map_path: str):
        self._pipline_config_path = pipline_config_path
        self._check_point_number = check_point_number
        self._check_point_folder = check_point_folder
        self._label_map_path = label_map_path
        self.configs = config_util.get_configs_from_pipeline_file(self._pipline_config_path)

        self.detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)
        self.ckpt = self.restore_checkpoint(check_point_folder=self._check_point_folder,
                                            check_point_number=self._check_point_number)

        self._category_index = label_map_util.create_category_index_from_labelmap(label_map_path)

    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def restore_checkpoint(self, check_point_folder: str, check_point_number: int):
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(check_point_folder, f'ckpt-{check_point_number}')).expect_partial()
        return ckpt

    def detect_image(self, image_path: str):
        img = cv2.imread(image_path)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = self.detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'] + label_id_offset,
            detections['detection_scores'],
            self._category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

        plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
        plt.show()
