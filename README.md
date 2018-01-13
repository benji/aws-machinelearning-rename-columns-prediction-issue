# aws-machinelearning-rename-columns-prediction-issue
Demonstration of Amazon Machine Learning issue that produces different model predictions when the name of the columns change in the data.

Train dataset in data/1stFlrSF sometimes produces different predictions than data/FirstFlrSF. The only difference is a column "1stFlrSF" in the first, has been renamed to "FirstFlrSF" in the other.

Originally reported at: https://forums.aws.amazon.com/thread.jspa?threadID=270993

```
$ head -n 2 bp*
==> bp-BPC7FXLCXIZ6LS56-test.csv <==
tag,trueLabel,score
1461,0,1.323967E5

==> bp-GODIBIZ3T7OBQKSA-test.csv <==
tag,trueLabel,score
1461,0,1.339596E5
```

## Parameters

```
# Use this AWS profile and that region
boto3.setup_default_session(profile_name='personal', region_name='us-east-1')

# Pick a temp bucket
bucket = 'kaggles'
```
## Run

```
python main.py
```

## Sample Output

```
$ python main.py
Uploading dataset data/1stFlrSF/train.csv to s3://kaggles/train.csv
Creating AML datasource ds-train-Z5U25BH3FRCVWIKQ
Uploading dataset data/1stFlrSF/test.csv to s3://kaggles/test.csv
Creating AML datasource ds-test-HOXLIXV6QS7T24Q3
Creating AML model ml-GUX25TXULJRU5DPU...
Creating AML batch prediction bp-M7N6QGUXNHQV5MKP...
Batch prediction bp-M7N6QGUXNHQV5MKP has been scheduled for data folder: data/1stFlrSF.
```