# Argumentative Relation Classification Pipeline
The pipeline queries the argument search engine, performs argumentative relation classification and outputs the resulting argumentation structure.

## Project Structure
### common
In this directory everything having to do with more than one of the following tasks is placed.

### search_engine
In this directory everything having to do with retrieving arguments from the search engine - in our case ArgumenText - is done.

### relation_processing
In this directory everything having to do with processing the argumentative relations between the argumentative sentences is done.  

### output_processing
In this directory everything having to do with outputting the argumentation structure is done.

### output
Here the ouput files are dumped.

### run.py
Python file to run pipeline with command line interface.

### util.py
Python file offering optional services. Used to initialize logging.

## Running the Argumentative Relation Classification Pipeline
### Prerequisites
Install all requirements from the requirements.txt file.  
In line 36 of run.py enter your ``userID`` and ``apiKey`` to access ArgumenText.  
In the directory relation_processing/predict_relation replace the content of the model folder with the [trained models](https://drive.google.com/drive/folders/1wqd2sR-i8MInHCVeFPk-PLWh9Cg3HAtv?usp=sharing).
Leave the directory structure as is.
### Command Line Interface
To show the possible configurations and details about the single parameters execute ``run.py -h``.  