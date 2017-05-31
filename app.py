#!/usr/bin/python3
# pylint: disable=C0103

""" Execution module
!!! Careful, dont mess with first letter of each method_dict since we will use it for
argument passing :)
"""

from configparser import ConfigParser
from collections import Counter
import argparse
import os
import csv

method_dict = {
    '--baye' : 'using the Naive Bayesian',
    '--kclass': 'using the K Classifier',
    '--svm': 'using the SVM'
}

abspath = os.path.abspath(__file__)
wkpath = os.path.dirname(abspath)
os.chdir(wkpath)

config = ConfigParser()
config.read(wkpath + '\\classifier\\setting.ini')
data_loc = config['data_location']

def run(methods, core):
    """ main function to run the training and """
    if methods == 'baye':
        from classifier import NBaye

        nbaye = config['N-baye']
        classify = NBaye.NBaye(nbaye, core)

    elif methods == '':
        pass
        #kclass = config['K-classifier']

    elif methods == '':
        pass
        #svm = config['SVM']

    with open(wkpath + data_loc['feeding'], 'r', encoding='utf-8') as rd:
        csvreader = csv.reader(rd)
        data = list(csvreader)[1:]

    max_looping = len(data) // int(nbaye['sample'])

    for x in range(max_looping):
        classify.training([[y[1], y[4]] for y in data[x * 1000 : (x + 1) * 1000]])
        rs = []
        test_gen = [[y[1], y[4]] for y in data[(x + 1) * 1000 : (x + 2) * 1000]]
        for z in test_gen:
            rs.append(classify.validate(z))

        true_counter = Counter([x[0] for x in rs])
        print(true_counter[True] / len(test_gen), '% predicted correct')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the ML training')

    method_group = parser.add_mutually_exclusive_group(required=True)
    for k, v in method_dict.items():
        method_group.add_argument(k[1:3], k, help=v, action='store_true')
    parser.add_argument('-c', '--core', type=int, help='number of cores using', required=True)

    args = parser.parse_args()

    #Return the method used
    for k, v in args.__dict__.items():
        if v is True and k in ('baye', 'kclass', 'svm'):
            method = k
            break

    try:
        run(method, args.core)
    except KeyboardInterrupt:
        print("Cancelling process...")
