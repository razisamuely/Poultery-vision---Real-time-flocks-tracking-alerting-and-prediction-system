from google.cloud import bigquery
import os
import json

# Get tables configs
tables_configs_path: str = "mySqlCommands/databBasesConfigs/bigQueryDataBaseConfigs/bigQueryConfigs.json"
with open(tables_configs_path) as f:
    configs: dict = json.loads(f.read())

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = configs["GOOGLE_APPLICATION_CREDENTIALS_PATH"]
table_creation_queries_path: str = configs["table_creation_queries_path"]
create_tables_queries_list: list = os.listdir(table_creation_queries_path)
d:dict = configs["tables_creation"]


# Create connection
client = bigquery.Client()
try:
    # Run creation table query one by one and create tables
    for table in create_tables_queries_list:
        # Make sure "sql" file type
        if ".sql" in table:
            # Generate query path
            path = f'{table_creation_queries_path}/{table}'
            print("\n====", path, "====\n")

            # Read query and format "data_set" and "partition_expiration_days"
            with open(path, 'r') as f:
                create_table_query = f.read().format(**d)
                print(create_table_query)

            client.query(create_table_query)

except Exception as e:
    raise e

finally:
    client.close()
