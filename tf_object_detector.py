# Specify model imports
import datetime

from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import numpy as np
import os
import tensorflow as tf
from collections import Counter
# tf.config.run_functions_eagerly(True)
from tensorflow.python.ops.numpy_ops import np_config
import json
import os
import glob

np_config.enable_numpy_behavior()
import time

# Disable GPU if necessary
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Create object detector
class TFObjectDetector():

    # Constructor
    def __init__(self,
                 path_to_object_detection='./models/research/object_detection/configs/tf2',
                 path_to_model_checkpoint='./checkpoint',
                 checkpoint_num=0,
                 path_to_labels='./labels.pbtxt',
                 model_name='ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8',
                 ):

        self.model_name = model_name
        self.pipeline_config_path = path_to_object_detection
        self.pipeline_config = self.pipeline_config_path
        self.full_config = config_util.get_configs_from_pipeline_file(self.pipeline_config)
        self.checkpoint_num = checkpoint_num
        self.path_to_model_checkpoint = path_to_model_checkpoint
        self.path_to_labels = path_to_labels
        self.setup_model()

    # Set up model for usage
    def setup_model(self):
        self.build_model()
        self.restore_checkpoint()
        self.detection_function = self.get_model_detection_function()
        self.prepare_labels()

        # Build detection model

    def build_model(self):
        model_config = self.full_config['model']
        assert model_config is not None
        self.model = model_builder.build(model_config=model_config, is_training=False)
        return self.model

        # Restore checkpoint into model

    def restore_checkpoint(self):
        assert self.model is not None
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint.restore(
            os.path.join(self.path_to_model_checkpoint, f'ckpt-{self.checkpoint_num}')).expect_partial()

        # Get a tf.function for detection

    def get_model_detection_function(self):
        assert self.model is not None

        @tf.function
        def detection_function(image):
            image, shapes = self.model.preprocess(image)
            prediction_dict = self.model.predict(image, shapes)
            detections = self.model.postprocess(prediction_dict, shapes)
            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detection_function

        # Prepare labels
        # Source: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb

    def prepare_labels(self):
        label_map = label_map_util.load_labelmap(self.path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.label_map_dict = {i['id']: i['name'] for i in categories}

        # Get keypoint tuples
        # Source: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb

    def get_keypoint_tuples(self, eval_config):
        tuple_list = []
        kp_list = eval_config.keypoint_edge
        for edge in kp_list:
            tuple_list.append((edge.start, edge.end))
        return tuple_list

        # Prepare image

    def prepare_image(self, image):
        return tf.convert_to_tensor(
            np.expand_dims(image, 0), dtype=tf.float32
        )

        # Perform detection

    def detect(self, image, label_offset=1):
        # Ensure that we have a detection function
        assert self.detection_function is not None

        # Prepare image and perform prediction
        image = image.copy()
        image_tensor = self.prepare_image(image)
        detections, predictions_dict, shapes = self.detection_function(image_tensor)

        # Use keypoints if provided
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in detections:
            keypoints = detections['detection_keypoints'][0].numpy()
            keypoint_scores = detections['detection_keypoint_scores'][0].numpy()
        threshold = 0.3

        # Perform visualization on output image/frame
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=threshold,
            agnostic_mode=False,
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
            keypoint_edges=self.get_keypoint_tuples(self.full_config['eval_config']))

        # raz counter to text

        #         label_mapping = {1: 'drinking', 2: 'eating', 3: 'standing', 4: 'laying'}
        #         # print(detections['detection_classes'].tolist())
        #         c_dict = Counter([cl for cl, sc in
        #                           zip(detections['detection_classes'].tolist()[0], detections['detection_scores'].tolist()[0])
        #                           if sc > threshold])
        #         d = {label_mapping[k + 1]: v for k, v in c_dict.items()}
        #         text = str(d).replace("{", "").replace("}", "").replace(", ", "\n").replace("'", "")
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         cv2.putText(image, text, (0, int(image.shape[1] / 2)),
        #                     font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # raz counter to text

        # Return the image
        return image

    def detect_counts(self, image, label_offset=1):
        # Ensure that we have a detection function
        assert self.detection_function is not None

        # Prepare image and perform prediction
        image = image.copy()
        image_tensor = self.prepare_image(image)
        detections, predictions_dict, shapes = self.detection_function(image_tensor)

        # Use keypoints if provided
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in detections:
            keypoints = detections['detection_keypoints'][0].numpy()
            keypoint_scores = detections['detection_keypoint_scores'][0].numpy()

        threshold = 0.4

        # raz counter to text

        r = detections['detection_classes'][0][detections['detection_scores'][0] > 0.45]
        c_dict = Counter(r.tolist())

        d = {self.label_map_dict[k + 1]: v for k, v in c_dict.items()}

        return d

    # Predict image from folder

    def detect_image(self, path, output_path):
        # Load image
        image = cv2.imread(path)

        # Perform object detection and add to output file
        output_file = self.detect(image)

        # Write output file to system
        cv2.imwrite(output_path, output_file)

        # Predict video from folder

    def detect_video(self, path, output_path):

        # Read the video
        vidcap = cv2.VideoCapture(path)
        frame_read, image = vidcap.read()
        count = 0
        print(vidcap.isOpened())
        if vidcap.isOpened():
            print('isOpened')
            width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
            height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height
            print(width, height)
        # Set output video writer with codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        out = cv2.VideoWriter(output_path, fourcc, 25.0, (int(width), int(height)))

        # Iterate over frames and pass each for prediction
        while frame_read:
            # Perform object detection and add to output file
            output_file = self.detect(image)

            # Write frame with predictions to video
            out.write(output_file)

            # Read next frame
            frame_read, image = vidcap.read()
            count += 1

        # Release video file when we're ready
        out.release()

    def video_detection_to_counter(self, input_path, output_path, video_id,
                                   params={"camera_id": "101", "cycle_id": "1"}):
        # Read the video
        vidcap = cv2.VideoCapture(input_path)
        frame_read, image = vidcap.read()
        video_start_time, camera_id = video_id.split("-")
        camera_id = camera_id[:camera_id.index(".")]
        video_start_time = datetime.datetime.strptime(video_start_time, "%Y%m%d%H%M%S")

        if vidcap.isOpened():
            width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
            height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height

        # Iterate over frames and pass each for prediction
        c = 0
        image_0 = image + 2  # Adding 2 as some noise for first diff
        while frame_read:
            c += 1
            if c % 8 == 0:
                c = 0
                # Perform object detection and add to output file
                d = self.detect_counts(image)

                # Measure images diff
                images_diff = round(((image - image_0) ** 2).mean(), 2)

                frame_time = video_start_time + datetime.timedelta(milliseconds=vidcap.get(cv2.CAP_PROP_POS_MSEC))
                with open(f'{output_path}/{frame_time}.txt', 'w') as file:
                    d.update({"event_time": f"{frame_time}",
                              "movement": images_diff})

                    d.update(params)
                    file.write(json.dumps(d))

            # Read next frame
            image_0 = image
            frame_read, image = vidcap.read()

    def get_part(self, frame, p):
        if len(frame.shape) == 3:
            return frame[p[0][0]:p[0][1], p[1][0]:p[1][1], :]
        else:
            return frame[p[0][0]:p[0][1], p[1][0]:p[1][1]]

    def color_upper_lower_distribution_diff(self, image, color):
        """
        Get image, devid into 4 parts, subtract the two upper and the two lower
        """
        r1 = int(image.shape[0] / 2)
        c1 = int(image.shape[1] / 2)

        p11 = ((0, r1), (0, c1))
        p12 = ((0, r1), (c1, c1 * 2))
        p21 = ((r1, r1 * 2), (0, c1))
        p22 = ((r1, r1 * 2), (c1, c1 * 2))

        a_p = self.get_part(frame=image, p=p11)
        b_p = self.get_part(frame=image, p=p12)
        c_p = self.get_part(frame=image, p=p21)
        d_p = self.get_part(frame=image, p=p22)

        hists = []
        for n, f in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [a_p, b_p, c_p, d_p]):
            histr = []
            for i, col in enumerate(color):
                hist = cv2.calcHist([f], [i], None, [256], [0, 256])
                histr.append(hist)
            hists.append(np.array(histr))

        diff_upper = np.linalg.norm(hists[0] - hists[1])
        diff_lower = np.linalg.norm(hists[2] - hists[3])
        diff_left = np.linalg.norm(hists[0] - hists[2])
        diff_right = np.linalg.norm(hists[1] - hists[3])
        return diff_upper, diff_lower, diff_left, diff_right

    def pictures_detection_to_counter(self,
                                      input_path,
                                      pictures_list,
                                      output_path,
                                      last_image,
                                      params={"cycle_id": "1"}):

        picture_time, camera_id = pictures_list[0].split("-")
        camera_id = camera_id[:camera_id.index(".")]
        image_path = input_path + '/' + pictures_list[0]
        image = cv2.imread(image_path)

        c = 0
        while image is None:
            c += 1
            print("image = ", image, image_path)
            time.sleep(2)
            image = cv2.imread(input_path + '/' + pictures_list[0])
            print("image = ", image)
            print("c = ", c)
            if c > 10:
                if len(pictures_list) > 1:
                    pictures_list = pictures_list[1:]
                    c = 0
                else:
                    return None

        image_0 = image + 2 if last_image is None else last_image
        for pictur_name in pictures_list[1:]:
            picture_time, _ = pictur_name.split("-")

            image = cv2.imread(input_path + '/' + pictur_name)
            try:
                # Events detection
                d = self.detect_counts(image)

                # Gray image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2GRAY)

                # Movement detection
                images_diff = round(((image - image_0) ** 2).mean(), 2)
                images_gray_diff = round(((gray - gray_0) ** 2).mean(), 2)

                # Color distribution
                diff_upper, diff_lower, diff_left, diff_right = self.color_upper_lower_distribution_diff(image=image,
                                                                                                         color=(
                                                                                                         'b', 'g', 'r'))
                # Color distribution gray scale
                diff_upper_gray, diff_lower_gray, diff_left_gray, diff_right_gray =  self.color_upper_lower_distribution_diff(image=gray, color=['gray'])

                path = f'{output_path}/{picture_time}.txt'
                with open(path, 'w') as file:
                    d.update({"event_time": f"{picture_time}",
                              "movement": images_diff,
                              "movement_gray": images_gray_diff,
                              "diff_upper": int(diff_upper),
                              "diff_lower": int(diff_lower),
                              "diff_left": int(diff_left),
                              "diff_right": int(diff_right),
                              "diff_upper_gray": int(diff_upper),
                              "diff_lower_gray": int(diff_lower_gray),
                              "diff_left_gray": int(diff_left_gray),
                              "diff_right_gray": int(diff_right_gray),
                              "camera_id": int(camera_id)
                              })
                    d.update(params)
                    file.write(json.dumps(d))
                print(d)

                image_0 = image
            except Exception as e:
                if image is None:
                    print(f"{pictur_name} is None")
                else:
                    raise e

        return image_0
