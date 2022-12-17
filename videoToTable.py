import json
from datetime import datetime
from tf_object_detector import TFObjectDetector
from train import trainingPreperations


class dirToClass(object):
    paths: None = None
    pass


class videoToTable(object):
    """
    The "videoToTable" class is is taking care all  the technical issue of detecting videos.
    This class using the training configs in order to run a detection, so make sure you go over the training configs first.
    The class is uses the configs which can be found over the configs_path
    The class is maneging "new_videos_table", table which holds video ids and detection processes statuses.
    The class is load detection awaiting videos, run detection process, write counters (detection output) and
    update the video detection statuses
    """

    def __init__(self, configs_path: str):
        """
        --- defining class instances ---
        """
        self._configs_path = configs_path
        self.configs = self._get_configs()
        self._trn = trainingPreperations()
        self.tfObjectDetector = self.getTFObjectDetectorObject()

    def getTFObjectDetectorObject(self):
        """
        -- init TFObjectDetector object, which lying on the train configs --
        """
        print("self._trn.configs.paths.CHECKPOINT_PATH", self._trn.configs.paths.CHECKPOINT_PATH)
        tfo = TFObjectDetector(path_to_object_detection=self._trn.configs.files.PIPELINE_CONFIG,
                               path_to_model_checkpoint=self._trn.configs.paths.CHECKPOINT_PATH,
                               checkpoint_num=self.configs.checkpoint_num,
                               path_to_labels=self._trn.configs.files.LABELMAP,
                               model_name=self._trn.configs.PRETRAINED_MODEL_NAME)
        return tfo

    def _get_configs(self):
        """
         -- Read json configs and convert it to class for object oriented use. --
        """
        with  open(self._configs_path) as f:
            data = json.load(f)
        configs = dirToClass()
        configs.__dict__ = data
        return configs




    def get_video_path_from_new_video_table(self, new_videos_table_path: str, awaiting_video=None) -> json:
        """
        -- Pull video path, of videos which not went through detection process. --
        0.Read "new_videos_table"
        1.Iterate through all videos, break if at list one isn't detected yet
        2.1.Check for awaiting videos
        2.2.If awaiting video is exist
        2.3.updated properties as status ,times etc'...
        2.4.Update "new_videos_table"
        2.5.Break loop
        3.Return awaiting video
        """
        # 0 Read table
        with open(new_videos_table_path) as f:
            new_videos_table: json = json.load(f)

        # 1.Iterate through all videos, break if at list one isn't detected yet
        for k, v in new_videos_table.items():
            # 2.1 Check for awaiting videos
            # 2.2 If awaiting video is exist
            if v.get("status").get("awaiting").get("is_current"):
                # save video properties in "awaiting_video"
                awaiting_video = {k: v}

                # 2.3 updated properties as status ,times etc'...
                new_videos_table[k]["status"]["awaiting"]["is_current"] = False
                now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                new_videos_table[k]["status"]["in_process"] = {"is_current": True, "time": now, "is_succeeded": -1}

                # 2.4 Update "new_videos_table"
                with open(new_videos_table_path, 'w') as outfile:
                    json.dump(new_videos_table, outfile)

                break
        # Return awaiting video
        return awaiting_video

    def updated_video_detection_status(self, awaiting_video: json, is_detection_passed: bool, processing_time: float):
        """
        --- updated video detection status for awaiting detection videos ---
        if the detection process is failed, "new_videos_table" get updated any way in order to be able to track the fails.
        1.Updating statuses and times
        2.Updating "new_videos_table"
        2.1.read table
        2.2.update table dict
        2.3.save table as json

        """
        # 1.Updating statuses and times
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        video_id, value = next(iter(awaiting_video.items()))
        awaiting_video[video_id]["status"]["in_process"]["is_current"] = False
        awaiting_video[video_id]["status"]["in_process"]["is_succeeded"] = is_detection_passed
        awaiting_video[video_id]["status"]["done"] = {"is_succeeded": is_detection_passed,
                                                      "time": now,
                                                      "is_current": is_detection_passed,
                                                      "processing_time": processing_time}
        # 2. Updating "new_videos_table"
        # 2.1.read table
        with open(self.configs.new_videos_table_path) as f:
            new_videos_table: json = json.load(f)
        # 2.2.update table dict
        new_videos_table.update(awaiting_video)

        # 2.3.save table as json
        with open(self.configs.new_videos_table_path, 'w') as outfile:
            json.dump(new_videos_table, outfile)
