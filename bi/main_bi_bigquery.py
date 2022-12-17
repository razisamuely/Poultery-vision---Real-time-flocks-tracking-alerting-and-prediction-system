from loadsEventsToBigQuery import loadEventsToBigQuery
from pathlib import Path

if __name__ == '__main__':
    '''
    --- Run load_events_to_bigquery proccess  --- 
    make sure:
    1. Events tables are exists, if not, run createSqlTables.py, and make sure all tabeles schames are update over "createMySqlTables" folder
    2. counter_columns_map, the dictionary which map events name to columns names, is updated 
    3. Mysql server is up
    4. Run configs under "loadEventsToSqlConfigs.json" is updated 
    '''

    abs_path = Path(__file__).absolute()
    load_events_to_bigqueryl_configs_path = "../realtime_configuration/bigQueryConfigs.json"
    letm = loadEventsToBigQuery(load_events_to_bigqueryl_configs_path)
    letm.insert_all_events_to_bigquery()
