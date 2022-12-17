from sklearn.ensemble import IsolationForest
import pytz
from datetime import datetime
import pandas as pd
from datetime import timedelta
import time
import json
from recordclass import recordclass
import os
from google.cloud import bigquery
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
import smtplib
from os.path import basename
from get_configs_as_class import get_configs_as_class
import cv2


class anomalyDetection():
    def __init__(self, configs_path):
        self.df_bq = pd.DataFrame(
            index=[pytz.UTC.localize(datetime(year=2032, month=6, day=4, hour=6, minute=4)),
                   pytz.UTC.localize(datetime(year=2032, month=6, day=4, hour=6, minute=5))])

        self._configs_path = configs_path
        self.configs = get_configs_as_class(configs_path)
        self._seconds_interval = self.get_seconds_interval_between_rows()

    def get_configs_as_class(self):
        with open(self._configs_path, encoding='utf-8', errors='ignore') as json_file:
            initConfigs = json.load(json_file, object_hook=lambda d: recordclass('X', d.keys())(*d.values()))
        return initConfigs

    def get_seconds_interval_between_rows(self):
        nl = self.df_bq.index.to_series().nlargest(2)
        seconds_interval = (nl[0] - nl[1]).seconds
        return seconds_interval

    def get_start_back_time_for_querying(self, df, start_time: datetime, back_time: datetime):
        if df.index.min() < pytz.UTC.localize(back_time):

            # Convert for proper format
            start_time_f = start_time.strftime("%Y-%m-%d-%H%M%SZ")
            back_time_f = back_time.strftime("%Y-%m-%d-%H%M%SZ")

            # Filtering relevant times only
            df = df[df.index.to_series().between(back_time_f, start_time_f)]

            # Gap size
            start_exist_diff = pytz.UTC.localize(start_time) - df.index.max()

            if start_exist_diff.seconds / 60 < self.configs.querying_minutes_back:
                sleep_time = (self.configs.querying_minutes_back * 60 - start_exist_diff.seconds)
                print(
                    f"{datetime.now():%m-%d} - Taking sleep of {sleep_time / 60 :.2} minutes, till accumulation of enough data")
                time.sleep(sleep_time)

            back_time_query = df.index.max() + timedelta(seconds=self._seconds_interval)
            start_time_query = start_time + timedelta(seconds=self._seconds_interval)



        else:
            print(f"== {datetime.now():%m-%d} - New full dataframe ==\n")
            back_time_query = back_time
            start_time_query = start_time + timedelta(seconds=self._seconds_interval)
            self.df_bq = pd.DataFrame()

        return back_time_query, start_time_query

    def create_query(self, start_time, back_time):
        cols = ',\n'.join(self.configs.big_query.columns)
        read_event_query: str = f"""select {cols}\nfrom 
                                        `{self.configs.big_query.project}.{self.configs.big_query.table}.{self.configs.big_query.sub_table}` 
                                        where event_time between '{back_time}' and '{start_time}';"""

        return read_event_query

    def run_query_get_df(self, query):
        # os.environ[
            # 'GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/raz.shmuely/Documents/privet/chickens/sshkeys/rational-hydra-337609-b3c222590c8e.json"

        bqclient = bigquery.Client()
        df_origin = (
            bqclient.query(query)
                .result()
                .to_dataframe(
                create_bqstorage_client=True,
            )
        )

        return df_origin

    def processing_big_query_raw_data(self, df):
        df.movement = df.movement.astype(float)
        df.index = pd.to_datetime(df['event_time'])
        df.drop(columns=["event_time"], inplace=True)
        df = df.fillna(0)
        return df

    def join_old_and_big_query_data(self, df, df_bq):
        df_bq = pd.concat([df, df_bq])
        print(f"{datetime.now():%m-%d %H:%M} - Shape before index deduping {df_bq.shape[0]}")
        self.df_bq = df_bq[~df_bq.index.duplicated(keep='last')]
        print(f"{datetime.now():%m-%d %H:%M} - Shape after index deduping {df_bq.shape[0]}")
        return self.df_bq

    def run_anomaly_detection(self,
                              df,
                              indx,
                              anomaly_detection_interval_mnts,
                              anomaly_train_interval_hrs,
                              night_filter_seconds
                              ):
        d = df.sort_index()
        night_filter = d.freeFSN_counter.rolling(f'{night_filter_seconds}s').sum() < 2
        d_n_f = d[night_filter]
        d_s = d_n_f.div(d_n_f.cnt, axis=0)
        d_r = d_s[self.configs.anomaly_columns][:]

        df = pd.DataFrame()

        if indx <= d.index.max():
            indx = min([indx + timedelta(minutes=30), d_r.index.max()])
            start_indx = indx - timedelta(hours=anomaly_train_interval_hrs)
            d_i = d_r.loc[start_indx: indx]
            X = (d_i - d_i.mean()) / d_i.std()
            clf = IsolationForest(n_estimators=300, contamination=0.001, max_features=1).fit(X)
            test = X[indx - timedelta(minutes=anomaly_detection_interval_mnts):indx]
            if test.shape[0] > 0:
                y_pred = clf.predict(test)
                extreem_movment = test[y_pred == -1]
                if extreem_movment.shape[0] > 0:
                    df = pd.concat([df, extreem_movment])
            return df


        else:
            print(f"The given index {indx} is above exist max index which is {d.index.max()}")

    def filter_anomaly_data(self, df):
        fltr = (df.movement > self.configs.filters.movement_thresh) & \
               (df.diff_lower > self.configs.filters.difflower_thresh) & \
               (df.diff_upper > self.configs.filters.diffupper_thresh)
        return df[fltr]

    def send_mail(self, send_from: str, subject: str, text: str,
                  send_to: list, password, files=None):

        msg = MIMEMultipart()
        msg['From'] = send_from
        msg['To'] = ', '.join(send_to)
        msg['Subject'] = subject

        msg.attach(MIMEText(text, 'html'))

        for f in files or []:
            with open(f, "rb") as fil:
                ext = f.split('.')[-1:]
                attachedfile = MIMEApplication(fil.read(), _subtype=ext)
                attachedfile.add_header(
                    'content-disposition', 'attachment', filename=basename(f))
            msg.attach(attachedfile)

        smtp = smtplib.SMTP(host="smtp.gmail.com", port=587)
        smtp.starttls()
        smtp.login(send_from, password)
        smtp.sendmail(send_from, send_to, msg.as_string())
        smtp.close()

    def create_rtsp_path(self, times, camera, seconds_interval, ip, password, user_name):

        rtsp_paths = ''
        for t in times:
            start = t - timedelta(seconds=30)
            end = t + timedelta(seconds=60)

            start = start.strftime("%Y%m%dT%H%M%SZ")
            end = end.strftime("%Y%m%dT%H%M%SZ")

            path = f"rtsp://{user_name}:{password}@{ip}/Streaming/tracks/{camera}/?starttime={start}&endtime={end}"
            rtsp_paths += f"""<a {path}">{path}</a>""" + '\n\n\n'

        rtsp_paths = f"""<pre>\n{rtsp_paths}</pre>"""
        return rtsp_paths

    def take_anomaly_pictures(self, camera, time):
        # Delete all files
        os.system("rm pictures/*")

        for i in time:
            times = [i - timedelta(seconds=15),
                     i + timedelta(seconds=15),
       ]

            for t in times:
                t = t.strftime("%Y%m%dT%H%M%SZ")
                path = f"rtsp://admin:Rev45tal@141.226.94.56/Streaming/tracks/{camera}/?starttime={t}/picture"

                cap = cv2.VideoCapture(path)

                counter = 0
                while counter <= 2:

                    ret, frame = cap.read()
                    if counter == 2:
                        saving_path = f"pictures/time={t}_camera={camera}.png"
                        cv2.imwrite(saving_path, frame)

                    counter += 1

    def get_pictures_paths_list(self, path):
        return [f"{path}/{i}" for i in os.listdir(path) if ('png' in i) | ('jpg' in i) | ('jpeg' in i)]
