# todo convert to class


from datetime import datetime
import pandas as pd
import json
import glob
from datetime import datetime
from typing import List
import time
import os
from IPython.display import display, HTML
import numpy as np


def events_records_files_to_records_list(files: List[str], index_columns: List[str]):
    # Loop through files
    data = []
    c = 0
    for single_file in files:
        with open(single_file, 'r') as f:
            try:
                c += 1
                json_file = json.load(f)
                if not all([json_file.get(i) for i in index_columns]):
                    raise Exception(
                        f'file = {single_file} is not containing one of the following keys = {index_columns} json = {json_file}')
                data.append(json_file)
                c = 0

            except Exception as e:
                if os.stat(single_file).st_size == 0:
                    pass
                else:
                    print(single_file)
                    if c == 5:
                        raise e
                    else:
                        print(f"try number {c}")
                        pass

            os.remove(single_file)

    return data


def split_aggregation_and_write_completed_intervals(event_df_aggreagted: pd,
                                                    event_df_aggreagted_count,
                                                    seconds_to_write: float,
                                                    aggregated_events_write_path: str):
    # 1 aggregated_events_write_path: Completed time intrvals filter out and write
    # 2 not completed filter in and continue to aggreagte

    # Filter on time intervals
    # filter 1 do not pull times withini nterval:
    a = datetime.utcnow() - event_df_aggreagted.index.get_level_values("event_time")
    ready_events_filter = a.seconds > 60 * seconds_to_write

    # filter2 leav last time interval in
    date = event_df_aggreagted.index.get_level_values("event_time")

    # filter final
    ready_events_filter = (ready_events_filter) & (date != date.max())

    # Send all finished interval
    event_df_aggreagted_write = event_df_aggreagted[ready_events_filter]
    event_df_aggreagted_count_write = event_df_aggreagted_count[ready_events_filter]
    # event_df_aggreagted_write_mean = event_df_aggreagted_write.div(event_df_aggreagted_count_write.iloc[:, 0], axis=0)
    event_df_aggreagted_write_mean = event_df_aggreagted_write
    event_df_aggreagted_write_mean['cnt'] = event_df_aggreagted_count_write.iloc[:, 0]

    # Keep all intervals in process
    event_df_aggreagted = event_df_aggreagted[~ready_events_filter]
    df_count = event_df_aggreagted_count[~ready_events_filter]

    if len(event_df_aggreagted_write_mean) > 0:
        path = f"{aggregated_events_write_path}{datetime.utcnow()}.json"
        print(f"{datetime.utcnow().strftime('%H:%M:%S')} -- Writning aggreagted json to {path}")

        # Reset index and by that convert indexes to columns
        event_df_aggreagted_write_mean.reset_index(inplace=True)

        # Remove coluns with nulls
        event_df_aggreagted_write_mean.dropna(how='all', axis='columns', inplace=True)

        # Replace 0 i=with nan, in order to save place

        # Mapping floats to integers
        float_int_map = {c: "int" for d, c in zip(event_df_aggreagted_write.dtypes, event_df_aggreagted_write.columns)
                         if "float" in d.name}
        event_df_aggreagted_write_mean = event_df_aggreagted_write_mean.fillna(0)
        display(event_df_aggreagted_write_mean)
        event_df_aggreagted_write_mean = event_df_aggreagted_write_mean.astype(float_int_map)
        display(event_df_aggreagted_write_mean)
        event_df_aggreagted_write_mean = event_df_aggreagted_write_mean.agg(lambda x: x[x != 0].to_dict(), axis=1)
        event_df_aggreagted_write_mean.to_json(path, orient="records", date_format="iso")

        print("=" * 30)
        print(f"$%$$%$% utc now = {datetime.utcnow()} %$%$%$%$%")
        print("\n -------- event_df_aggreagted_write ---------- \n")
        display(event_df_aggreagted_write)
        print("\n -------- event_df_aggreagted_count_write ---------- \n")
        display(event_df_aggreagted_count_write)
        print("\n -------- event_df_aggreagted_write_mean ---------- \n")
        display(event_df_aggreagted_write_mean)
        print("\n -------- event_df_aggreagted ---------- \n")
        display(event_df_aggreagted)
        print("=" * 30)

    # Return df for contuing aggregation
    return event_df_aggreagted, df_count


def aggregate_events(events_read_path: str,
                     aggregated_events_write_path: str,
                     seconds_to_write: float = 1,
                     index_columns: List[str] = ["cycle_id", "camera_id"],
                     event_df_aggreagted=pd.DataFrame(),
                     df_count=pd.DataFrame(),
                     start_time=datetime.now(),
                     sleep_time: int = 10):
    """
    Aggregating events with respect to cycle_id, camera_id and time interval.
    sections :
    1. Get events
    2. If events are exits, group by cycle_id, camera_id and time interval and sum counters
    3. Filter out and write completed time interval
    4. Keep not completed interval and continue to aggregate till interval time is completed
    5. If no new events are exist, go to sleep and wait.
    """

    while True:
        # 1. Get events
        files = glob.glob(events_read_path, recursive=True)
        number_of_files = len(files)

        # 2. Aggregate events
        if number_of_files > 0:
            # Log
            print(f"{datetime.utcnow().strftime('%H:%M:%S')} -- {number_of_files} new events detected")

            # Get list of records
            records_list = events_records_files_to_records_list(files=files, index_columns=index_columns)

            # Convert to pandas DataFrame
            event_df = pd.DataFrame.from_records(records_list, index=index_columns)
            event_df.event_time = event_df.event_time.apply(pd.to_datetime)
            event_df.set_index('event_time', append=True, inplace=True)

            # Concate with previous aggs
            event_df = pd.concat([event_df_aggreagted, event_df], axis=0)

            # Aggregate with summation
            event_df_aggreagted = event_df.groupby(
                ["cycle_id", "camera_id", pd.Grouper(level="event_time", freq=f"{seconds_to_write}Min")]).sum()
            #             return event_df, event_df_aggreagted,records_list
            #             Insome case we in a middle of video writing, so wait to directory to get empty

            # Count
            event_df_aggreagted_count = event_df.groupby(
                ["cycle_id", "camera_id", pd.Grouper(level="event_time", freq=f"{seconds_to_write}Min")]).size()

            df_count = pd.concat([df_count, event_df_aggreagted_count], axis=0)
            df_count = df_count.groupby(level=0).sum()

            time.sleep(0.5)
        # If no more the 1 events found, go to sleep and wait for future events
        if len(event_df_aggreagted) > 3:  # why bigger then 1 and not 0? cause the single is the last time nterval which isnt complete yet
            print("event_df_aggreagted\n", event_df_aggreagted)
            print("len(event_df_aggreagted)", len(event_df_aggreagted))

            # 3. split aggregation and_write completed intervals:
            event_df_aggreagted, df_count = split_aggregation_and_write_completed_intervals(
                event_df_aggreagted=event_df_aggreagted,
                event_df_aggreagted_count=df_count,
                seconds_to_write=seconds_to_write,
                aggregated_events_write_path=aggregated_events_write_path)
        else:
            print(f"{datetime.utcnow().strftime('%H:%M:%S')} -- No new events detected, taking a sleep of {sleep_time}")
            time.sleep(sleep_time)
