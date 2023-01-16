from data_input import *
import argparse
import json
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--pipeline",
    type=str,
    choices=["data_pipeline", "train", "predict"],
    required=True,
    help="Pipeline (stages) to run from the application.",
)
parser.add_argument(
    "-c", "--config", type=str, help="Config file",
)

args = parser.parse_args()
print(args.config)
with open(args.config, 'r') as config_file:
    config = json.load(config_file)

print('Reading file {}'.format(config['dataset_creation']['credit_records_fp']))
credit_records = read_file(config['dataset_creation']['credit_records_fp'])

print('Processing credit records')
credit_pipe = Pipeline([
    ('Generate Labels',DataLabeler()),
    ('Drop Debt Columns', NameDropper(cols=['Bad_Debt', 'Good_Debt', 'Neutral_Debt']))
])

labels = credit_pipe.fit_transform(credit_records)


print('Reading file {}'.format(config['dataset_creation']['application_records_fp']))
application_records = read_file(config['dataset_creation']['application_records_fp'])

print('Processing application records')
applications_pipe=Pipeline([
    ('Join Labels', LabelJoiner(labels=labels)),
    ('Test Train Split', TrainTestSplit()),
    ('Scale Income', MinMaxScaler('AMT_INCOME_TOTAL')),
    ('One Hot Encode', OneHotEncode(cols=['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE'])),
    ('Gender Map', ColumnMapper(col='CODE_GENDER', mapper={'M':1, 'F':0})),
    ('Car Map', ColumnMapper(col='FLAG_OWN_CAR', mapper={'Y':1, 'N':0})),
    ('Realty Map', ColumnMapper(col='FLAG_OWN_REALTY', mapper={'Y':1, 'N':0})),
    ('Redundant Columns', NameDropper(cols=['FLAG_MOBIL'])),
    ('Drop Duplicates', DropDuplicates(col='ID'))])

labelled_dataset = applications_pipe.fit_transform(application_records)

upload(labelled_dataset, config)






