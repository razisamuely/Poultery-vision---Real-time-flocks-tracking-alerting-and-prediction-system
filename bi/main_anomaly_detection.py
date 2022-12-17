from anomaly_detection_utils import anomalyDetection
import pandas as pd
import pytz
from datetime import datetime
from datetime import timedelta
from get_configs_as_class import get_configs_as_class
import time
from pathlib import Path
import json
import os
configs_path = "../realtime_configuration/anomaly_detection_configs.json"
ad = anomalyDetection(configs_path=configs_path)


## --# bs_path = Path(__file__).absolute()
# tables_configs_path: str = "../realtime_configuration/bigQueryConfigs.json"
#
# with open(tables_configs_path) as f:
#     configs: dict = json.loads(f.read())
#
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = configs["GOOGLE_APPLICATION_CREDENTIALS_PATH"]


## --
while True:
    start_time = datetime.now() if ad.configs.start_time == 'now' else ad.configs.start_time
    back_time = start_time - timedelta(hours=ad.configs.hours_back)

    back_time_query, start_time_query = ad.get_start_back_time_for_querying(
        df=ad.df_bq,
        start_time=start_time,
        back_time=back_time)

    query = ad.create_query(start_time=start_time,
                            back_time=back_time_query)

    df_origin = ad.run_query_get_df(query=query)
    # df_origin = pd.read_csv("df_origin.csv")

    df = ad.processing_big_query_raw_data(df=df_origin)
    assert df.index.max() > pytz.UTC.localize(start_time - timedelta(minutes=10))

    print("df.index.min()", df.index.min(), "df.shape", df.shape)
    print("ad.df_bq.index.max()", ad.df_bq.index.max(), "ad.df_bq.shape", ad.df_bq.shape)

    df_bq = ad.join_old_and_big_query_data(df, ad.df_bq)

    df_anomaly = ad.run_anomaly_detection(df=df_bq,
                                          indx=df_bq.index.max(),
                                          anomaly_detection_interval_mnts=ad.configs.alerts_minutes_back,
                                          anomaly_train_interval_hrs=ad.configs.hours_back,
                                          night_filter_seconds=ad.configs.night_filter_seconds
                                          )

    if df_anomaly.shape[0] > 0:
        df_anomaly_filtered = ad.filter_anomaly_data(df=df_anomaly)

        if df_anomaly_filtered.shape[0] > 0:
            # Prepare RTSP path
            d = df_anomaly_filtered[:]
            rtsp_configs = get_configs_as_class("../realtime_configuration/videos_configs.json")
            rtsp_paths = ad.create_rtsp_path(times=d.index,
                                             camera='301',
                                             seconds_interval='120',
                                             ip=rtsp_configs.hikload_args.host,
                                             password=rtsp_configs.hikload_args.password,
                                             user_name=rtsp_configs.hikload_args.user
                                             )
            mail_text = rtsp_paths + '\n\n\n' + d.to_string()
            ad.take_anomaly_pictures(camera=ad.configs.camera, time=d.index[:1])
            email_configs = get_configs_as_class("../realtime_configuration/usersNamesAndPasswords.json")
            mail = ', '.join(email_configs.receivers)
            pictures_files = ad.get_pictures_paths_list(ad.configs.pictures_path)
            ad.send_mail(send_from=email_configs.sender.user_name,
                         subject=f"Anomaly | Alert - {str(datetime.now())[:19]}",
                         text=mail_text,
                         send_to=', '.join(email_configs.receivers),
                         password=email_configs.sender.password,
                         files=pictures_files)
            print(f"!!! Alert !!! Anomaly detected {rtsp_paths} '\n\n\n' {d.to_string()}")
        else:
            print(f" -- {datetime.now():%H:%M:%S} No anomaly detected")

    t = ad.configs.querying_minutes_back * 60
    print(f" -- {datetime.now():%H:%M:%S} Anomaly test finished, taking sleep of {t} seconds")
    time.sleep(t)
