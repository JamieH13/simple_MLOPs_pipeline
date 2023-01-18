from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from datetime import datetime
import os
import joblib


def test_train_split(dataset):
    X_train = dataset[dataset.train == 1]
    Y_train = dataset[dataset.train == 1]['label']

    X_test = dataset[dataset.train == 0]
    Y_test = dataset[dataset.train == 0]['label']

    return X_train, Y_train, X_test, Y_test

def load_model(config):

    model = RandomForestClassifier(n_estimators = config['n_estimators'],
                                   min_samples_split = config['min_samples_split'])

    return model

def upload_model(model, config):

    name = config['model_name']
    name += datetime.now().strftime("%Y%m%d-%H%M%S")

    output_path = os.path.join(config['output_path'], name)

    print('Saving model to {}'.format(output_path))

    joblib.dump(model, output_path)

def model_summary(model, X_train, Y_train, X_test, Y_test):
    print('Model Summary')
    print('Training accuracy: ' + str(model.score(X_train, Y_train)))
    print('Training AUC ' + str(roc_auc_score(Y_train, model.predict_proba(X_train)[:, 1])))
    print('Test accuracy: ' + str(model.score(X_test, Y_test)))
    print('Test AUC: ' + str(roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])))

def upload_predictions(predictions, config):
    name = config['predictions_name']
    name += datetime.now().strftime("%Y%m%d-%H%M%S")
    name += '.csv'

    output_path = os.path.join(config['output_path'], name)

    print('Saving predictions to {}'.format(output_path))

    predictions.to_csv(output_path, index=False)



