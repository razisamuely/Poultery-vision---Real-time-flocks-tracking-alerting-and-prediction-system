import json
import time
import os
from get_configs_as_class import getConfigs
from os.path import isfile
from datetime import datetime
from google.cloud import bigquery
from google.api_core.exceptions import BadRequest


class loadEventsToBigQuery():
    def __init__(self, load_events_to_bigquery_configs_path):
        self._load_events_to_mysql_configs_path: str = load_events_to_bigquery_configs_path
        self._load_events_configs = getConfigs().get_configs_as_class(configs_path = self._load_events_to_mysql_configs_path)
        # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self._load_events_configs.GOOGLE_APPLICATION_CREDENTIALS_PATH

    def insert_event_to_bigquery(self, event: str, client, schema: dict = None):
        '''
        ---- Lading events to bigqueryl ------
        1. Load events as text file
        3.1 Connect to database
        3.3 load event to data base
        3.4 Delete the event text format from the event file
        4.1 If loading failed raise exception
        5. Close connection
        '''

        # 1. Load events as text file
        event_path = os.path.join(self._load_events_configs.events_path, event)

        # Check file is realy exist, sometimes, files get deleted but the os not get updated.
        if isfile(event_path):
            dataset_ref = client.dataset(self._load_events_configs.tables_creation["dataset"])
            table_id = "frame_counter_NirIsrael_brn1_cy2_20_07_2022"
            table_ref = dataset_ref.table(table_id)
            job_config = bigquery.LoadJobConfig(autodetect=False)
            job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
            with open(event_path, "rb") as source_file:
                try:
                    job = client.load_table_from_file(
                        source_file,
                        table_ref,  # Must match the destination dataset location.
                        job_config=job_config,
                    )
                    # 3.4 Delete the event text format from the event file
                    job.result()  # Waits for table load to complete.
                except BadRequest as e:
                    print(job.errors)
                    raise e

            os.remove(event_path)

            current_time = datetime.now().strftime("%H:%M:%S")
            dataset_id = self._load_events_configs.tables_creation["dataset"]
            print(
                f"{current_time} - Loaded {job.output_rows} rows into {dataset_id}:{table_id} - deleted : {event_path}")

    def insert_all_events_to_bigquery(self, endswith: str = "json", events_path: str = None,
                                      sleeping_time_load_events: int = None):
        if events_path == None:
            load_events_configs = self._load_events_configs.events_path
        if sleeping_time_load_events == None:
            sleeping_time_load_events = self._load_events_configs.sleeping_time_load_events

        tries = 0
        while True:
            current_time = datetime.now().strftime("%H:%M:%S")
            events_names_list = [f for f in os.listdir(load_events_configs) if f.endswith(f'.{endswith}')]
            tries += 1
            len_events_names_list = len(events_names_list)
            if len_events_names_list > 0:
                self._client = bigquery.Client()
                try:
                    # Read all events into list
                    events = ''
                    for event in events_names_list:
                        event_path = f"{self._load_events_configs.events_path}{event}"
                        with open(event_path) as f:
                            event = f.read()

                        # Add event previous
                        events += ',' + event[1:-1].replace(']','')
                        print("reading from local ", event)

                        # remove i'th event
                        os.remove(event_path)
                    # Dump list of json into json file
                    events_file = f"{datetime.now()}_event.json"
                    events_path = f"{self._load_events_configs.events_path}/{events_file}"
                    events = events.replace("},{", "}\n{")[1:]
                    with open(events_path, "w") as output:
                        output.write(events)

                    # Feed function with new merged jsons
                    self.insert_event_to_bigquery(events_file, self._client)
                    tries = 0
                except BaseException as err:
                    raise err
                finally:
                    self._client.close()

            else:
                print(f"{current_time} - No new events detected, taking {sleeping_time_load_events} of ZZZZ")
                time.sleep(sleeping_time_load_events)
                tries = 0
