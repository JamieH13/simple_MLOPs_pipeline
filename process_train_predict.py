from data_processing.methods import *
from data_processing.helpers import *

from modelling.helpers import *

from sklearn.pipeline import Pipeline

import argparse
import json
import joblib

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

################################ DATASET PROCESSING ################################

if args.pipeline == 'data_pipeline':
    print('Reading file {}'.format(config['credit_records_fp']))
    credit_records = read_file(config['credit_records_fp'])

    print('Processing credit records')
    credit_pipe = Pipeline([
        ('Generate Labels', DataLabeler()),
        ('Drop Debt Columns', NameDropper(cols=['Bad_Debt', 'Good_Debt', 'Neutral_Debt']))
    ])

    labels = credit_pipe.fit_transform(credit_records)

    print('Reading file {}'.format(config['application_records_fp']))
    application_records = read_file(config['application_records_fp'])

    print('Processing application records')
    applications_pipe = Pipeline([
        ('Join Labels', LabelJoiner(labels=labels)),
        ('Test Train Split', TrainTestSplit()),
        ('Scale Income', MinMaxScaler('AMT_INCOME_TOTAL')),
        ('One Hot Encode', OneHotEncode(cols=['NAME_INCOME_TYPE',
                                              'NAME_EDUCATION_TYPE',
                                              'NAME_FAMILY_STATUS',
                                              'NAME_HOUSING_TYPE',
                                              'OCCUPATION_TYPE'])),
        ('Gender Map', ColumnMapper(col='CODE_GENDER', mapper={'M': 1, 'F': 0})),
        ('Car Map', ColumnMapper(col='FLAG_OWN_CAR', mapper={'Y': 1, 'N': 0})),
        ('Realty Map', ColumnMapper(col='FLAG_OWN_REALTY', mapper={'Y': 1, 'N': 0})),
        ('Drop Duplicates', DropDuplicates(col='ID'))
    ])

    # Transform and fit columns
    labelled_dataset = applications_pipe.fit_transform(application_records)

    upload_dataset(labelled_dataset, config)

################################ MODEL TRAINING ################################

if args.pipeline == 'train':
    print('Reading Dataset')
    labelled_dataset = read_file(config['labelled_dataset_fp'])

    print('Training Model')
    train_pipeline = Pipeline([
        ('Drop Columns', NameDropper(cols=['FLAG_MOBIL', 'ID', 'train', 'label']))
    ])

    # Split dataset into training and test sets
    X_train, Y_train, X_test, Y_test = test_train_split(labelled_dataset)

    X_train = train_pipeline.fit_transform(X_train)
    X_test = train_pipeline.fit_transform(X_test)

    model = load_model(config)

    model.fit(X_train, Y_train)

    train_acc, train_auc, test_acc, test_auc = model_summary(model, X_train, Y_train, X_test, Y_test)

    # Save model and log metrics
    upload_model(model, config, train_acc, train_auc, test_acc, test_auc)

################################ MODEL PREDICTIONS ################################

if args.pipeline == 'predict':
    print('Reading Model')
    model = joblib.load(config['model_fp'])

    print('Reading Dataset')
    labelled_dataset = read_file(config['labelled_dataset_fp'])

    prediction_pipeline = Pipeline([
        ('Drop Columns', NameDropper(cols=['FLAG_MOBIL', 'ID', 'train', 'label']))
    ])

    # Remove columns from dataset before predicting
    prediction_dataset = prediction_pipeline.fit_transform(labelled_dataset)

    print('Generating Predictions')
    labelled_dataset['prediction'] = model.predict_proba(prediction_dataset)[:, 1]

    upload_predictions(labelled_dataset, config)
