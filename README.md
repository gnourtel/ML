# Machine Learning
The Machine Learning wrote in Python

This is a plug-and-play application which purpose is to quickly testing
and adjusting method / agorithm without recoding or revise the workflow.
All the activies is provoked by function calls and APIs.

The code structure:
/.data_feed
 |__ machine_feed.csv
 |__ machine_validation.csv
/.data_output
/classifier.py
/app.py

The flows for app will go:
app.py [args] > checking the ML methods > load the data_feed > fork the
process in to multiprocess > improve the base knowledge meantime > get
result and validate data again.

The usage is simply app call with parsing agruments:
usage: app.py [-h] (-s | -k | -b) -c CORE

Run the ML training

optional arguments:
  -h, --help            show this help message and exit
  -s, --svm             using the K Classifier
  -k, --kclass          using the K Classifier
  -b, --baye            using the Naive Bayesian
  -c CORE, --core CORE  number of cores using
