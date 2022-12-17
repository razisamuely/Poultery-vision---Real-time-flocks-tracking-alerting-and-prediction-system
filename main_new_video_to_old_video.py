import os
import json
import time
from typing import List
from datetime import datetime

class dirToClass(object):
    paths: None = None
    pass

configs = dirToClass()

videos_management_configs: str = "realtime_configuration/videos_management.json"



while True:
    with  open(videos_management_configs) as f:
        configs.__dict__ = json.load(f)

    current_time = datetime.now().strftime("%H:%M:%S")
    new_videos_name: List[str] = os.listdir(configs.new_videos_path)
    new_videos_name = [v for v in new_videos_name if configs.video_type in v]
    old_videos_name: List[str] = os.listdir(configs.old_video_table_path)

    videos_for_processing: List[str] = []
    videos_for_deleting: List[str] = []

    with open(configs.new_video_table_path) as f:
        new_videos_table: dict = json.loads(f.read())

    for v in new_videos_name:
        if v not in new_videos_table:
            videos_for_processing.append(v)

        elif new_videos_table.get(v).get("status").get("done", {}).get("is_succeeded", False):
            videos_for_deleting.append(v)

    len_videos_for_processing = len(videos_for_processing)
    len_videos_for_deleting = len(videos_for_deleting)

    if len_videos_for_processing > 0:
        print(f"""{current_time} - New videos for processing : {"--".join(videos_for_processing)}""")
        d = {new_video: {"status": {"awaiting": {"is_current": True}}} for new_video in videos_for_processing
             if new_video not in old_videos_name
             }
        new_videos_table.update(d)

    if len_videos_for_deleting > 0:
        print(f"""{current_time} - Deleting the following processed videos : {"--".join(videos_for_deleting)}""")
        # Old videos

        current_old_videos = {}
        for k in videos_for_deleting:
            del new_videos_table[k]
            delete_video_path = os.path.join(configs.new_videos_path, k)
            os.remove(delete_video_path) if configs.delete_videos_after_processing else True

    if len_videos_for_processing > 0 or len_videos_for_deleting > 0:
        # Save new_videos_table
        with open(configs.new_video_table_path, 'w') as outfile:
            json.dump(new_videos_table, outfile)

    if len_videos_for_deleting == 0:
        sleeping_time = 10
        print(
            f"{current_time} - len_videos_for_processing = {len_videos_for_processing}, len_videos_for_deleting = {len_videos_for_deleting}, taking sleep of {sleeping_time} sec")
        time.sleep(sleeping_time)
