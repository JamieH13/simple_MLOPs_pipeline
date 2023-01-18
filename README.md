# WovenLight Take Home Task - Jamie Howie

Take home task from WovenLight. This a lightweight processing pipeline for a new analytics project.

The dataset is credit card approval data, available from Kaggle [here](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

## Setup

Clone the repository to your local machine:
```
https://github.com/JamieH13/WovenLight_task_Jamie.git
```

Create a new python environment using:
```
conda create -n myenv python=3.7 pip=22.0
```

Install requirements:
```
pip install -r requirements.txt
```

Check the installation by running:
```
python -m pytest
```

## Example run
The data_pipeline config is used to specify the datasets to be fed in and processed, and where the output files are stored. This pipeline ingests the files, processes the columns ready for training, labels the data, and saves the output ready for the model training.
```
{
  "application_records_fp": "credit_card_data/application_record.csv",
  "credit_records_fp": "credit_card_data/credit_record.csv",
  "output_path": "processed_data",
  "dataset_name": "labelled_dataset_"
}
```
This stage of the pipeline is accessed using
```
python process_train_predict.py -p "data_pipeline" -c data_pipeline_config.json
```

Once the labelled dataset has been created, this can be piped into the model training stage using the train config. This stage reads in the dataset, and then trains a classifer (currently and random forest) on the data. The final model is then saved and uploaded.
```
{
    "labelled_dataset_fp": "processed_data/labelled_dataset_20230118-012423.csv",
    "output_path": "models",
    "model_name": "sample_model_",
    "n_estimators": 100,
    "min_samples_split": 2
}
```

This stage of the pipeline is accessed using
```
python process_train_predict.py -p "train" -c train_config.json
```

The trained model can then be used to predict on any provided dataset.
```
{
    "model_fp": "models/sample_model_20230118-012729",
    "labelled_dataset_fp": "processed_data/labelled_dataset_20230118-012423.csv",
    "output_path": "predictions",
    "predictions_name": "scored_dataset_"
}
```

This stage of the pipeline is accessed using
```
python process_train_predict.py -p "predict" -c predict_config.json
```
