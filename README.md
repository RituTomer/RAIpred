# **RAIred**
A computational approach to predict the Rheumatoid arthritis inducing peptides using the sequence information.
## Introduction
RAIpred is a bioinformatic-ware with the ability to discriminate RA-causing peptides from RA Non-causing peptides. It is developed to predict, scan, and, design the rheumatoid arthritis disease causing peptides using sequence information only. In the standalone version, we have provided XGB classifier based model to make prediction which is developed using experimentally-validated peptides that can cause rheumatoid arthritis disease and experimentally-validated peptides that do not cause disease. This method is available as five major modules such as:
- (i)   Predict     : It used machine learning model in the backend developed on Composition enhanced transition and distribution (CeTD) based features as input features.
- (ii)  Design      : This module generates all the possible mutants of a peptide by altering a single amino acid at a time and used the model to predict its potential to cause disease.
- (iii) Scan        : This module generates overlapping patterns from the submitted sequence of specified length and use them to make predictions.
- (iv)  Ensemble    : This module joins motif search and machine learning model to predict the potential of the submitted peptides to cause disease.

RAIpred is also available as web-server at https://webs.iiitd.edu.in/raghava/raipred. Please read/cite the content about the RAIpred for complete information including algorithm behind the approach.

## Standalone
The Standalone version of raipred is written in python3 and following libraries are necessary for the successful run:
- scikit-learn
- Pandas
- Numpy
-xgboost

https://webs.iiitd.edu.in/raghava/raipred/download.html

## Minimum USAGE
To know about the available option for the stanadlone, type the following command:
```
python raipred.py -h
```
To run the example, type the following command:
```
python3 raipred.py -i test_sequences.csv
```
This will predict if the submitted sequences are Disease-Causing or Non-disease-causing. It will use other parameters by default. It will save the output in "outfile.csv" in CSV (comma seperated variables).

## Full Usage
```
usage: 	raipred.py [-h] 
                       [-i INPUT 
                       [-o OUTPUT]
		       [-j {1,2,3, 4}]
		       [-t THRESHOLD]
                       [-w {9,10,11,12,13,14,15,16,17,18,19,20}]
                       [-p {3,4,5,6,7,8,9}]
		       [-d {1,2}]
```
```
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input: protein or peptide sequence(s) in FASTA format
                        or single sequence per line in single letter code
  -o OUTPUT, --output OUTPUT
                        Output: File for saving results by default outfile.csv
  -m {1,2}, --method {1,2}
                        Method Type: 1:Main (Disease Vs Random), 2:Alternate
                        (Disease Vs Non-Disease), by default 1
  -j {1,2,3,4,5}, --job {1,2,3,4,5}
                        Job Type: 1:Predict, 2:Design, 3:Scan, 4: Ensemble Method,
                        5: Ensemble Method, by default 1
  -t THRESHOLD, --threshold THRESHOLD
                        Threshold: Value between 0 to 1 by default 0.32
  -w {9,10,11,12,13,14,15,16,17,18,19,20}, --winleng {9,10,11,12,13,14,15,16,17,18,19,20}
                        Window Length: 9 to 20 (scan mode only), by default 9
  -d {1,2}, --display {1,2}
                        Display: 1:Only Disease-Causing Peptides, 2: All
                        peptides, by default 1
```

**Input File:** It allow users to provide input in the FASTA format.

**Output File:** Program will save the results in the CSV format, in case user do not provide output file name, it will be stored in "outfile.csv".

**Job:** User is allowed to choose between four different modules, such as, 1 for prediction, 2 for Designing, 3 for scanning, and 4 for ensemble method, by default its 1.

**Threshold:** User should provide threshold between 0 and 1, by default its 0.32 for other jobs except PQ Density for which the default threshold is 0.72.

**Window length**: User can choose any pattern length between 9 and 20 in long sequences. This option is available for only scanning module.

**Display type:** This option allow users to fetch either only disease causing peptides by choosing option 1 or prediction against all peptides by choosing option 2.

RAIpred Package Files
=======================
It contains following files, brief descript of these files given below

INSTALLATION                        : Installations instructions

LICENSE                             : License information

README.md                           : This file provide information about this package

model.zip                           : This zipped file contains the compressed version of model

cdpred.py                           : Main python program


# Reference
