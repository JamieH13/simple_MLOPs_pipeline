from datetime import datetime
import os
import pandas as pd
import sys

def read_file(filepath):

    if filepath.split('.')[-1] == 'csv':
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

