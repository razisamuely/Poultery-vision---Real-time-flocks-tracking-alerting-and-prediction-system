import json
import time
from google.cloud import bigquery
from datetime import timedelta
from pathlib import Path
import os

abs_path = Path(__file__).absolute()
tables_configs_path: str = "../realtime_configuration/bigQueryConfigs.json"

# tables_configs_path: str = "mySqlCommands/databBasesConfigs/bigQueryDataBaseConfigs/bigQueryConfigs.json"
with open(tables_configs_path) as f:
    configs: dict = json.loads(f.read())

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = configs["GOOGLE_APPLICATION_CREDENTIALS_PATH"]

while True:
    # Define queries
    read_event_query: str = """select freeFS_counter,
                                      freeFSN_counter,
                                      freeD_counter,
                                      freeDC_counter,
                                      eating_24_counter,
                                      eating_s_counter ,
                                      eating_b_counter,
                                      eating_g_counter,
                                      eating_l_counter,
                                      drinking_counter,
                                      drinking_b_counter,
                                      drinking_g_counter,
                                      mushroom_counter,
                                      movement,
                                      event_time
                               from `rational-hydra-337609.frame_counters.frame_counter_NirIsrael_brn1_cy1_05_2022`
                               where event_time is not null;"""

    # Connect tot mysql
    try:
        bqclient = bigquery.Client()
        df = (
            bqclient.query(read_event_query)
                .result()
                .to_dataframe(
                # Optionally, explicitly request to use the BigQuery Storage API. As of
                # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
                # API is used by default.
                create_bqstorage_client=True,
            )
        )

    # 4.1 If loading failed raise exception
    except BaseException as err:
        raise err

    # 5. Close connection
    finally:
        bqclient.close()

    # Defind plot functions
    # Todo: move these functions to a separate class
    import pandas as pd
    import datetime
    import matplotlib.pyplot as plt


    def read_counter_date(location_file):
        df = pd.read_csv(location_file)
        return df


    def polt_drinking(df, location_export='foo'):
        df.movement = df.movement.astype(float)
        df.index = pd.to_datetime(df['event_time'])
        df.drop(columns=["event_time"], inplace=True)
        df = df.fillna(0)
        plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
               axisbelow=True, grid=True)
        ax = df.plot(lw=1, kind='line', figsize=(20, 13))
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

        fig = ax.get_figure()
        fig.savefig('foo.png')
        plt.close(fig)

        ## Without movment
        plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
               axisbelow=True, grid=True)
        dft = df.drop(columns =['movement'])
        ax = dft.plot(lw=1, kind='line', figsize=(20, 13))
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

        fig = ax.get_figure()
        fig.savefig('foo_movement_out.png')
        plt.close(fig)

        # ------  12 #
        df_rolled = df.sort_index(ascending=True)

        eating_cols = ["eating_24_counter", "eating_b_counter", "eating_s_counter", "eating_l_counter"]
        drinking_cols = ["drinking_counter", "drinking_b_counter","drinking_g_counter"]
        free_drinkers_cols = ["freeD_counter", "freeDC_counter"]
        eating_group_cols = ["eating_g_counter", "mushroom_counter"]

        df_rolled["eating"] = df_rolled[eating_cols].sum(axis=1)
        df_rolled["drinking"] = df_rolled[drinking_cols].sum(axis=1)
        df_rolled["freeD"] = df_rolled[free_drinkers_cols].sum(axis=1)
        df_rolled["eating_g"] = df_rolled[eating_group_cols].sum(axis=1)

        df_rolled.drop(columns=eating_cols + drinking_cols + free_drinkers_cols + eating_group_cols, inplace=True)
        df_rolled.head()

        minutes = 10
        minutes_norm = 60 * 12
        df_rolled = df_rolled[df_rolled.index > df_rolled.index.max() - timedelta(minutes=minutes_norm)]

        df_rolled = df_rolled / df_rolled.max()

        df_rolled = df_rolled.rolling(f'{60 * minutes}s').mean()


        plt.rc('axes',
               facecolor='#E6E6E6',
               edgecolor='none',
               axisbelow=True,
               grid=True
               )

        ax = df_rolled.plot(lw=1,
                            kind='line',
                            title=f"Rolling mean = {minutes}, mnts standardized = {int(minutes_norm / 60)} hrs",
                            figsize=(20, 13))
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        plt.xlim(left=df_rolled.index.min(), right=df_rolled.index.max())
        fig = ax.get_figure()
        fig.savefig('foo_norm_12.png')
        plt.close(fig)

        # ----- 24
        # ------  12 #
        df_rolled = df.sort_index(ascending=True)

        eating_cols = ["eating_24_counter", "eating_b_counter", "eating_s_counter", "eating_l_counter"]
        drinking_cols = ["drinking_counter", "drinking_b_counter", "drinking_g_counter"]
        free_drinkers_cols = ["freeD_counter", "freeDC_counter"]
        eating_group_cols = ["eating_g_counter", "mushroom_counter"]

        df_rolled["eating"] = df_rolled[eating_cols].sum(axis=1)
        df_rolled["drinking"] = df_rolled[drinking_cols].sum(axis=1)
        df_rolled["freeD"] = df_rolled[free_drinkers_cols].sum(axis=1)
        df_rolled["eating_g"] = df_rolled[eating_group_cols].sum(axis=1)

        df_rolled.drop(columns=eating_cols + drinking_cols + free_drinkers_cols + eating_group_cols, inplace=True)
        df_rolled.head()

        minutes = 10
        minutes_norm = 60 * 24
        df_rolled = df_rolled[df_rolled.index > df_rolled.index.max() - timedelta(minutes=minutes_norm)]

        df_rolled = df_rolled / df_rolled.max()

        df_rolled = df_rolled.rolling(f'{60 * minutes}s').mean()


        plt.rc('axes',
               facecolor='#E6E6E6',
               edgecolor='none',
               axisbelow=True,
               grid=True
               )

        ax = df_rolled.plot(lw=1,
                            kind='line',
                            title=f"Rolling mean = {minutes}, mnts standardized = {int(minutes_norm / 60)} hrs",
                            figsize=(20, 13))
        plt.xlim(left=df_rolled.index.min(), right=df_rolled.index.max())
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        fig = ax.get_figure()
        fig.savefig('foo_norm_24.png')
        plt.close(fig)

        #


    polt_drinking(df=df)

    ### TODO : TILL NOE THE CODE IS ONE SCRIPT THAT RUN EITHOUT CLASS
    ### TODO : OREDERING THE CODE TO SEPERATE CLASS
    ### TODO : THINKING ABOUT MOVE THIS CODE ALL TOE "APP" DIRECTORY

    # Send mail with attachment

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.application import MIMEApplication
    from os.path import basename


    def send_mail(send_from: str, subject: str, text: str,
                  send_to: list, files=None):

        send_to = default_address if not send_to else send_to

        msg = MIMEMultipart()
        msg['From'] = send_from
        msg['To'] = ', '.join(send_to)
        msg['Subject'] = subject

        msg.attach(MIMEText(text))

        for f in files or []:
            with open(f, "rb") as fil:
                ext = f.split('.')[-1:]
                attachedfile = MIMEApplication(fil.read(), _subtype=ext)
                attachedfile.add_header(
                    'content-disposition', 'attachment', filename=basename(f))
            msg.attach(attachedfile)

        smtp = smtplib.SMTP(host="smtp.gmail.com", port=587)
        smtp.starttls()
        smtp.login(username, password)
        smtp.sendmail(send_from, send_to, msg.as_string())
        smtp.close()


    with open("../realtime_configuration/usersNamesAndPasswords.json") as f:
        emailConfigs = json.loads(f.read())

    username = emailConfigs["sender"]["user_name"]
    password = emailConfigs["sender"]["password"]
    default_address = emailConfigs["receivers"]

    send_mail(send_from=username,
              subject=f"local_chicks report- test - {str(datetime.datetime.now())[:19]}",
              text=f"{str(datetime.datetime.now())[:19]}",
              send_to=["raz.shmuely@pipl.com"],
              files=[
                  "foo.png", "foo_norm_24.png", "foo_norm_12.png","foo_movement_out.png"]
              )
    print(f'{datetime.datetime.now().strftime("%H:%M:%S")} -- email sent to , taking a sleep of 5 mint')
    time.sleep(30 * 60)
