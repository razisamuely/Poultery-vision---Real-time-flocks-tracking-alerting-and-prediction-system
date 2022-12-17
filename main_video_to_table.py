import time
from videoToTable import videoToTable
from datetime import datetime
import os
import threading
from get_videos_from_nvr import getVideosFromNVR
import multiprocessing



def picture_to_table(pictures_path, vtt,last_image = None):
    while True:
        files = os.listdir(pictures_path)
        files = [i for i in files if ".jpg" in i]
        files = [f for f in files if os.path.exists(f"{pictures_path}/{f}")]
        start = time.time()

        if len(files) > 0:
            try:
                files = sorted(files)
                print(f"""{datetime.now().strftime("%H:%M:%S")} - Start detecting - {pictures_path}""")
                last_image = vtt.tfObjectDetector.pictures_detection_to_counter(input_path=pictures_path,
                                                                                pictures_list=files,
                                                                                output_path=vtt.configs.output_path,
                                                                                last_image=last_image)
                processing_time = time.time() - start

                for f in files:
                    os.remove(pictures_path + '/' + f)

                print(
                    f"""{pictures_path} , {datetime.now().strftime("%H:%M:%S")} - Detection finished, event saved to {vtt.configs.output_path}, processing_time = {processing_time}, len = {len(files)} """)

            except BaseException as err:
                raise

        else:

            print(
                f"""{pictures_path} , {datetime.now().strftime("%H:%M:%S")} - No video for detection found under {vtt.configs.videos_path},"""
                f"\ntaking a sleep of {vtt.configs.No_videos_for_detection_sleep_time_while_loop} sec")
            time.sleep(vtt.configs.No_videos_for_detection_sleep_time_while_loop)
            # break


if __name__ == '__main__':
    # Params
    configs_path = "realtime_configuration/detection_properties.json"
    vtt = videoToTable(configs_path=configs_path)

    videos_configs_path: str = "realtime_configuration/videos_configs.json"
    getVidNVR = getVideosFromNVR(videos_configs_path=videos_configs_path,
                                 zone="Israel")
    args = getVidNVR.args
    cameras = args.cameras.split(',')

    for camera in cameras:
        pictures_path = vtt.configs.pictures_path + '/' + camera
        print(pictures_path)
        t = threading.Thread(target=picture_to_table, args=(pictures_path, vtt,))
        t.start()
        # p = multiprocessing.Process(target=picture_to_table, args=(pictures_path, vtt,))
        # p.start()
