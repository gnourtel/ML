#!/usr/bin/python3
# pylint: disable=C0103

""" Execution module
!!! Careful, dont mess with first letter of each method_dict since we will use it for
argument passing :)
"""
import argparse
import classifier

method_dict = {
    '--baye' : 'using the Naive Bayesian',
    '--kclass': 'using the K Classifier',
    '--svm': 'using the SVM'
}

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

    classifier.run(method, args.core)
