import os
from get_videos_from_nvr import getVideosFromNVR
import cv2
import pytz
from datetime import datetime, timedelta, timezone
import threading



def get_pictures(user, password, host, camera, start, end, output_file, country_time_zone):
    isExist = os.path.exists(output_file)
    if not isExist:
        # Create folder if noe exist
        os.makedirs(output_file)
    else:
        # If file exist, empty it
        os.system(f"rm {output_file}/*")

    # path = f'rtsp://{user}:{password}@{host}/Streaming/tracks/{camera}/?starttime={start}&endtime={end}'
    # print("path", path)
    path = f'rtsp://{user}:{password}@{host}/Streaming/Channels/{camera}/picture'

    while True:
        tries = 0
        try:
            print('start')
            video = cv2.VideoCapture(path)
            start_time = datetime.now(country_time_zone)
            ret, frame = video.read()

            if (video.isOpened() == False):
                print("Error reading video file")


            c = 0
            while True:
                c += 1
                if c % 20 == 0:
                    c = 1
                    while tries < 10:
                        tries += 1
                        try:
                            frame_time = start_time + timedelta(milliseconds=video.get(cv2.CAP_PROP_POS_MSEC))
                            fram_time_edited = frame_time.strftime("%Y%m%d%H%M%S")
                            cv2.imwrite(f"{output_file}/{fram_time_edited}-{camera}.jpg", frame)
                            duration = datetime.now(country_time_zone) - start_time
                            print("try = ", tries, duration, " ", fram_time_edited)
                            tries = 0
                            break

                        except Exception as e:
                                print("tries = ", tries, "c = ", c, e)

                ret, frame = video.read()

                # Each hour create new connection"
                if tries >= 10:
                    print(f"exceeded maximum tries - {tries}")
                    break

                if start_time + timedelta(seconds=60 * 60) < datetime.now(country_time_zone):
                    print('raz')
                    break


        except Exception as e:
            raise e


if __name__ == '__main__':
    videos_configs_path: str = "realtime_configuration/videos_configs.json"
    getVidNVR = getVideosFromNVR(videos_configs_path=videos_configs_path,
                                 zone="Israel")
    args = getVidNVR.args
    videos_configs = getVidNVR.videos_configs
    args = getVidNVR.define_start_end_time(args=args,
                                           run_from_last_video_time=videos_configs.run_from_last_video_time)
    print(args.__dict__)


    country_time_zone = pytz.timezone("Israel")
    video_end_time = datetime.now(country_time_zone) - timedelta(minutes=2)
    video_start_time = video_end_time - timedelta(minutes=1)

    start = video_start_time.strftime("%Y%m%dT%H%M%SZ")
    end = video_end_time.strftime("%Y%m%dT%H%M%SZ")

    start = '20220323T201923Z'
    end   = '20220323T224023Z'

    password = args.password
    host = args.host
    user = args.user

    cameras = args.cameras.split(',')
    print(args.cameras.split(','))
    for camera in cameras:
        output_file = f"{args.downloads}/{camera}"
        print(output_file)

        t = threading.Thread(target=get_pictures,
                             args=(user, password, host, camera, start, end, output_file, country_time_zone))
        t.start()
