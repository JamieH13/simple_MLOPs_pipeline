from sklearn.ensemble import RandomForestClassifier


def test_train_split(dataset):
    X_train = dataset[dataset.train == 1]
    Y_train = dataset[dataset.train == 1]['label']

    X_test = dataset[dataset.train == 0]
    Y_test = dataset[dataset.train == 0]['label']

    return X_train, Y_train, X_test, Y_test

def load_model(config):

    model = RandomForestClassifier(n_estimators = config['training']['n_estimators'],
                                   min_samples_split = config['training']['min_samples_split'])

    return model

