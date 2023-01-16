from data_processing.methods import *
from data_processing.helpers import *

from modelling.helpers import *
import argparse
import json
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score

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
    ('One Hot Encode', OneHotEncode(cols=['NAME_INCOME_TYPE',
                                          'NAME_EDUCATION_TYPE',
                                          'NAME_FAMILY_STATUS',
                                          'NAME_HOUSING_TYPE',
                                          'OCCUPATION_TYPE'])),
    ('Gender Map', ColumnMapper(col='CODE_GENDER', mapper={'M':1, 'F':0})),
    ('Car Map', ColumnMapper(col='FLAG_OWN_CAR', mapper={'Y':1, 'N':0})),
    ('Realty Map', ColumnMapper(col='FLAG_OWN_REALTY', mapper={'Y':1, 'N':0})),
    ('Drop Duplicates', DropDuplicates(col='ID'))
])

labelled_dataset = applications_pipe.fit_transform(application_records)

upload(labelled_dataset, config)

#    if not data_pipeline:
#        labelled_dataset = pd.read_csv(config['training']['labelled_dataset'])

print('Training Model')
train_pipeline=Pipeline([
    ('Drop Columns', NameDropper(cols=['FLAG_MOBIL', 'ID', 'train', 'label']))
])

X_train, Y_train, X_test, Y_test = test_train_split(labelled_dataset)

X_train = train_pipeline.fit_transform(X_train)
X_test = train_pipeline.fit_transform(X_test)

model = load_model(config)

model.fit(X_train, Y_train)

print('Scoring model')
print('Training accuracy: ' + str(model.score(X_train,Y_train)))
print('Training AUC ' + str(roc_auc_score(Y_train, model.predict_proba(X_train)[:,1])))
print('Test accuracy: ' + str(model.score(X_test,Y_test)))
print('Test AUC: ' + str(roc_auc_score(Y_test, model.predict_proba(X_test)[:,1])))







