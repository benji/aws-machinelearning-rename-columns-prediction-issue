import os
import base64
import boto3

# Use this AWS profile and that region
boto3.setup_default_session(profile_name='personal', region_name='us-east-1')

# Pick a temp bucket
bucket = 'kaggles'

s3 = boto3.client('s3')
ml = boto3.client('machinelearning')


def train_and_predict(datafolder):
    """Creates train datasource and test datasource
    Train model with train datasource
    Run Batch Predictions with test datasource

    Expected in datafolder:
        train.csv: the training dataset to use to train a regression model
        test.csv: the test dataset to use for batch predictions
        aml_schema.json: the AML schema of the datasets

    Args:
        datafolder: The path to the data folder.
    """
    ds_schema_str = open(datafolder + '/aml_schema.json').read()

    train_ds_id = create_datasource('train', datafolder + '/train.csv',
                                    ds_schema_str)
    test_ds_id = create_datasource('test', datafolder + '/test.csv',
                                   ds_schema_str)

    model_id = 'ml-' + base64.b32encode(os.urandom(10))
    ml_recipe_str = open('aml_recipe.json').read()

    print 'Creating AML model ' + model_id + '...'
    ml.create_ml_model(
        MLModelId=model_id,
        MLModelName=model_id,
        MLModelType="REGRESSION",
        Parameters={
            "sgd.maxPasses": "100",
            "sgd.maxMLModelSizeInBytes": "104857600",  # 100 MiB
            "sgd.l2RegularizationAmount": "1e-6",
            "sgd.shuffleType": "auto"
        },
        Recipe=ml_recipe_str,
        TrainingDataSourceId=train_ds_id)

    bp_id = 'bp-' + base64.b32encode(os.urandom(10))

    print 'Creating AML batch prediction ' + bp_id + '...'
    data_s3_url = 's3://' + bucket

    ml.create_batch_prediction(
        BatchPredictionId=bp_id,
        BatchPredictionName=bp_id,
        MLModelId=model_id,
        BatchPredictionDataSourceId=test_ds_id,
        OutputUri=data_s3_url)

    print 'Batch prediction ' + bp_id + ' has been scheduled for data folder: ' + datafolder + ''


def create_datasource(name, dataset_filepath, ds_schema_str):
    """Uploads dataset to S3 and creates datasource.
    """
    s3path = 's3://' + bucket + '/' + name + '.csv'
    print 'Uploading dataset ' + dataset_filepath + ' to ' + s3path
    s3.upload_file(dataset_filepath, bucket, name + '.csv')

    ds_id = 'ds-' + name + '-' + base64.b32encode(os.urandom(10))
    print 'Creating AML datasource ' + ds_id
    ml.create_data_source_from_s3(
        DataSourceId=ds_id,
        DataSpec={"DataLocationS3": s3path,
                  "DataSchema": ds_schema_str},
        DataSourceName="ds-" + name,
        ComputeStatistics=True)

    return ds_id


# Do it!
#train_and_predict('data/1stFlrSF')
train_and_predict('data/FirstFlrSF')