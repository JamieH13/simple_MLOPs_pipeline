from datetime import datetime
import os
import pandas as pd
import sys
import boto3

def read_file(filepath):

    if 's3' in filepath:
        AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

        data = pd.read_csv(filepath,
            storage_options={
                "key": AWS_ACCESS_KEY_ID,
                "secret": AWS_SECRET_ACCESS_KEY,
                "token": AWS_SESSION_TOKEN,
            },
        )

    elif filepath.split('.')[-1] == 'csv':
        data = pd.read_csv(filepath)

    elif filepath.split('.')[-1] == 'parquet':
        data = pd.read_parquet(filepath)

    else:
        sys.exit('Unrecognised file type. Please check config file')

    return data


def upload_dataset(final_dataset, config):
    name = config['dataset_name']
    name += datetime.now().strftime("%Y%m%d-%H%M%S")
    name += '.csv'

    output_path = os.path.join(config['output_path'], name)

    print('Saving labeled dataset to {}'.format(output_path))

    final_dataset.to_csv(output_path, index=False)

