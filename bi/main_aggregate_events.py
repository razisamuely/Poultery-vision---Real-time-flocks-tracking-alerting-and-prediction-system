from aggregate_events_functions import aggregate_events
from datetime import datetime
import pandas as pd

events_read_path = "data/counters_data/events/*.txt"
aggregated_events_write_path = "data/counters_data/events_aggregated/"
index_columns = ["cycle_id", "camera_id"]
seconds_to_write = 1
sleep_time = 10
event_df, event_df_aggreagted, records_list = aggregate_events(events_read_path=events_read_path,
                                                               aggregated_events_write_path=aggregated_events_write_path,
                                                               seconds_to_write=0.5,
                                                               index_columns=index_columns,
                                                               event_df_aggreagted=pd.DataFrame(),
                                                               start_time=datetime.now(),
                                                               sleep_time=sleep_time)
