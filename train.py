import os
import json



class dirToClass(object):
    paths: None = None
    pass


class trainingPreperations(object):
    def __init__(self, configs_path: str = 'configs/trainconfigs.json'):
        self._configs_path = configs_path
        self.configs = self._get_configs()
        # self._PRETRAINED_MODEL_URL = PRETRAINED_MODEL_URL
        # self._TF_RECORD_SCRIPT_NAME = TF_RECORD_SCRIPT_NAME
        # self._CUSTOM_MODEL_NAME = CUSTOM_MODEL_NAME
        # self._LABEL_MAP_NAME = LABEL_MAP_NAME

    def _get_configs(self):
        with  open(self._configs_path) as f:
            data = json.load(f)
        configs = dirToClass()
        configs.__dict__ = data
        configs.paths = self.set_paths(configs.CUSTOM_MODEL_NAME,configs.IMAGE_PATH)

        configs.files = self.set_files(paths=configs.paths,
                                       CUSTOM_MODEL_NAME=configs.CUSTOM_MODEL_NAME,
                                       TF_RECORD_SCRIPT_NAME=configs.TF_RECORD_SCRIPT_NAME,
                                       LABEL_MAP_NAME=configs.LABEL_MAP_NAME)

        return configs

    def set_folders(self, image_path: str):
        if not os.path.exists(image_path):
            if os.name == 'posix':
                os.system(f"mkdir -p {image_path}")
            if os.name == 'nt':
                os.system(f"mkdir  {image_path}")

    def set_paths(self, CUSTOM_MODEL_NAME,IMAGE_PATH):
        paths_dict = {
            'WORKSPACE_PATH': os.path.join('TensorFlow', 'workspace'),
            'SCRIPTS_PATH': os.path.join('TensorFlow', 'scripts'),
            'APIMODEL_PATH': os.path.join('TensorFlow', 'models'),
            'ANNOTATION_PATH': os.path.join('TensorFlow', 'workspace', 'annotations'),
            'IMAGE_PATH': os.path.join('TensorFlow', 'workspace', IMAGE_PATH),
            'MODEL_PATH': os.path.join('TensorFlow', 'workspace', 'models'),
            'PRETRAINED_MODEL_PATH': os.path.join('TensorFlow', 'workspace', 'pre-trained-models'),
            'PRETRAINED_MODEL_PATH': os.path.join('TensorFlow', 'workspace', 'pre-trained-models'),
            'CHECKPOINT_PATH': os.path.join('TensorFlow', 'workspace', 'models', CUSTOM_MODEL_NAME),
            'OUTPUT_PATH': os.path.join('TensorFlow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'export'),
            'TFJS_PATH': os.path.join('TensorFlow', 'workspace', 'models', CUSTOM_MODEL_NAME, 'tfjsexport'),
            'TFLITE_PATH': os.path.join('TensorFlow', 'workspace', 'models', CUSTOM_MODEL_NAME,
                                        'tfliteexport'),
            'PROTOC_PATH': os.path.join('TensorFlow', 'protoc'),
            "VERIFICATION_SCRIPT_PATH": os.path.join('TensorFlow', 'models', 'research', 'object_detection', 'builders',
                                                     )

        }
        paths_class = dirToClass()

        paths_class.__dict__ = paths_dict

        return paths_class

    def set_files(self, paths, CUSTOM_MODEL_NAME, TF_RECORD_SCRIPT_NAME, LABEL_MAP_NAME):
        files = {
            'PIPELINE_CONFIG': os.path.join('TensorFlow', 'workspace', 'models', CUSTOM_MODEL_NAME,
                                            'pipeline.config'),
            'TF_RECORD_SCRIPT': os.path.join(paths.SCRIPTS_PATH, TF_RECORD_SCRIPT_NAME),
            'LABELMAP': os.path.join(paths.ANNOTATION_PATH, LABEL_MAP_NAME)
        }

        files_class = dirToClass()

        files_class.__dict__ = files

        return files_class

    def create_paths(self, paths):
        for path in paths.values():
            if os.name == 'posix':
                os.system(f"mkdir -p {path}")
            if os.name == 'nt':
                os.system(f"mkdir  {path}")

    def installations(self):
        if os.name == 'nt':
            os.system(f"pip install wget")

        if not os.path.exists(os.path.join(self.configs.paths.APIMODEL_PATH, 'research', 'object_detection')):
            os.system(f"git clone https://github.com/tensorflow/models {self.configs.paths.APIMODEL_PATH}")

        # Install Tensorflow Object Detection
        if os.name == 'posix':
            os.system(f"""sudo apt-get install protobuf-compiler""")
            os.system(
                f"""cd TensorFlow/models/research/ && protoc objlsect_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . """)
  #      clear_output()

        if os.name == 'nt':
            os.system(
                f"""url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip""")
            os.system(f"""wget.download(url)""")
            os.system(f"""move protoc-3.15.6-win64.zip {self.configs.paths.PROTOC_PATH}""")
            os.system(f"""cd {self.configs.paths.PROTOC_PATH} && tar -xf protoc-3.15.6-win64.zip""")
            os.system(
                f"""os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))""")
            os.system(
                f"""cd TensorFlow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python os.system(f"setup.py install")""")
            os.system(f"""cd TensorFlow/models/research/slim && pip install -e . """)

        os.system(f"""pip install tensorflow --upgrade""")
        os.system(f"""pip uninstall protobuf matplotlib -y""")
        os.system(f"""pip install protobuf matplotlib==3.2""")

        # tf records generator
        os.system(f"""git clone https://github.com/nicknochnack/GenerateTFRecord {self.configs.paths.SCRIPTS_PATH}""")
    #    clear_output()

    def run_varification_script(self, VERIFICATION_SCRIPT_PATH):
        VERIFICATION_SCRIPT_PATH = os.path.join(VERIFICATION_SCRIPT_PATH, 'model_builder_tf2_test.py')
        os.system(f"""python {VERIFICATION_SCRIPT_PATH}""")
   #     clear_output()

    def download_pre_trained_models(self, PRETRAINED_MODEL_URL, PRETRAINED_MODEL_NAME, PRETRAINED_MODEL_PATH):
        if os.name == 'posix':
            os.system(f"wget {PRETRAINED_MODEL_URL}")
            os.system(f"mv {PRETRAINED_MODEL_NAME + '.tar.gz'} {PRETRAINED_MODEL_PATH}")
            os.system(f"cd {PRETRAINED_MODEL_PATH} && tar -zxvf {PRETRAINED_MODEL_NAME + '.tar.gz'}")
        if os.name == 'nt':
            os.system(f"wget.download(PRETRAINED_MODEL_URL)")
            os.system(f"move {PRETRAINED_MODEL_NAME + '.tar.gz'} {PRETRAINED_MODEL_PATH}")
            os.system(f"cd {PRETRAINED_MODEL_PATH} && tar -zxvf {PRETRAINED_MODEL_NAME + '.tar.gz'}")

      #  clear_output()

    def create_label_map(self, labels, label_map_path):
        with open(label_map_path, 'w') as f:
            for label in labels:
                f.write('item { \n')
                f.write('\tname:\'{}\'\n'.format(label['name']))
                f.write('\tid:{}\n'.format(label['id']))
                f.write('}\n')

    def generate_tf_records(self):
        # generate into train/test/validation etc...
        for sub in self.configs.subsets:
            os.system(f"python {self.configs.files.TF_RECORD_SCRIPT} "
                      f"-x {os.path.join(self.configs.paths.IMAGE_PATH, sub)} "
                      f"-l {self.configs.files.LABELMAP} "
                      f"-o {os.path.join(self.configs.paths.ANNOTATION_PATH, f'{sub}.record')} ")

    def copy_model_configs(self, path_from: str, path_to: str):
        if os.name == 'posix':
            os.system(f"cp {path_from} {path_to}")
        if os.name == 'nt':
            os.system(f"copy {path_from} {path_to}")


import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format


class training(object):
    def __init__(self, configs):
        self.configs = configs

    def get_configs_from_pipeline(self):
        return config_util.get_configs_from_pipeline_file(self.configs.files.PIPELINE_CONFIG)

    def custom_train_configs(self, batch_size=4):
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
        with tf.io.gfile.GFile(self.configs.files.PIPELINE_CONFIG, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)

        pipeline_config.model.ssd.num_classes = len(self.configs.labels)
        pipeline_config.train_config.batch_size = batch_size
        pipeline_config.train_config.fine_tune_checkpoint = os.path.join(self.configs.paths.PRETRAINED_MODEL_PATH,
                                                                         self.configs.PRETRAINED_MODEL_NAME,
                                                                         'checkpoint',
                                                                         'ckpt-0')
        pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
        pipeline_config.train_input_reader.label_map_path = self.configs.files.LABELMAP
        pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
            os.path.join(self.configs.paths.ANNOTATION_PATH, 'train.record')]
        pipeline_config.eval_input_reader[0].label_map_path = self.configs.files.LABELMAP
        pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
            os.path.join(self.configs.paths.ANNOTATION_PATH, 'test.record')]

        config_text = text_format.MessageToString(pipeline_config)
        with tf.io.gfile.GFile(self.configs.files.PIPELINE_CONFIG, "wb") as f:
            f.write(config_text)

    def print_train_command(self,num_train_steps):
        TRAINING_SCRIPT = os.path.join(self.configs.paths.APIMODEL_PATH, 'research', 'object_detection', 'model_main_tf2.py')
        command = f"python {TRAINING_SCRIPT} " \
                  f"--model_dir={ self.configs.paths.CHECKPOINT_PATH} " \
                  f"--pipeline_config_path={self.configs.files.PIPELINE_CONFIG} " \
                  f"--num_train_steps={num_train_steps}"
        return command

    def evaluate_model(self):
        TRAINING_SCRIPT = os.path.join(self.configs.paths.APIMODEL_PATH, 'research', 'object_detection',
                                       'model_main_tf2.py')
        command = f"python {TRAINING_SCRIPT} " \
                  f"--model_dir={self.configs.paths.CHECKPOINT_PATH} " \
                  f"--pipeline_config_path={self.configs.files.PIPELINE_CONFIG} " \
                  f"--checkpoint_dir={self.configs.paths.CHECKPOINT_PATH}"


        return command

