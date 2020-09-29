# Relation Model
Here binary Classifiers are trained and evaluated. The Classifier has to decide whether there is a relation
from one sentence to the other (label: 1) or not (label: 0).

## datasets
The dataset has the columns Parent, Child and Class. Class is either no relation (n), relation support (s) or relation attack (a).
The relation holds form Child to Parent, so the data reads: "Child sentence has a n/s/a relation to Parent sentence."  
In my thesis the MyDataset_balanced.csv was a selected subset of the [corpus](https://www.doc.ic.ac.uk/~ft/softwareArg.html) provided by Lucas Carstens and Fransesca Toni in their work "Towards relation based argumentation mining".
Due to licensing, the data has to be prepared and inserted in the file by oneself.

## saved models and cached features
Models are saved under model/datasetName/classifierType-date-time/classifierType/classifierName.joblib  
Their evaluation results are saved under model/datasetName/classifierType-date-time/classifierType/classifierName-result.txt

Features are cached under features/datasetName/featureType/featureName_train_eval.joblib  
If parameters change (test/eval split percentage and seed, bert RepresentationModel config values, ...) cached joblib files have
to be deleted or relocated. Otherwise outdated cache is used!

## Python files - not executable
### util.py
Contains all util functions that are used by several other python files. These functions do not have to do something directly
with the Relation Model generation task.

### dataset.py
Contains functions to split data into train and eval set. Dataset labels are matched to 0 or 1 and sentences might be
turned to lowercase.

Ordering: split into train and eval before changing labels to have balanced train and eval set not only between relation
and no relation but also between attack and support 

### features.py
Contains functions to generate features out of the sentence pairs.  
For each feature there is a method using sentence pairs as input and a method using a dataset as input.
The feature extraction process should use dataset methods which call the sentence pair methods for each pair in the set.
This is for convenience and in case of BERT embeddings for fastness.  
BERT embeddings are cached and reused if cached files are not deleted (see saved models and cached features).

### classifier_sklearn.py
Classifiers from sklearn to be trained and evaluated with feature vectors.

### classifier_bert.py
BERT fine-tuning and evaluation with sentence pairs.


## Python files - executable
### run.py
Start training model with values defined in this file
* prepares data with dataset.py
    * this data can directly be input into classifier_bert.py
* generate feature vectors out of prepared data
    * this data can directly be input into classifier_sklearn.py
    
### dataset_statistics
Used to create statistics on sentence lengths in dataset.

### result_maker
Helper to create .csv files with the results of the training and evaluation of the models. 

