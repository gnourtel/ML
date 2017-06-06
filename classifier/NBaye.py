# pylint: disable=C0103
""" Naive Bayessian Classifer """

from multiprocessing import Pool
from collections import Counter
import json
import time
import sys
import csv
import numpy as np

def parallel_call(params):
    """ a helper for calling 'remote' instances """
    cls = getattr(sys.modules[__name__], params[0])  # get our class type
    instance = cls.__new__(cls)  # create a new instance without invoking __init__
    instance.__dict__ = params[1]  # apply the passed state to the new instance
    method = getattr(instance, params[2])  # get the requested method
    args = params[3] if isinstance(params[3], (list, tuple)) else [params[3]]
    return method(args)  # expand arguments, call our method and return the result


class NBaye(object):
    """ Naive Bayesian object for trainning
        from the initial loading, with n category rule, the training sample will be splitted
        into n + 1 list, the dict will be flexible adapt as following:
        dict1 will hold all the dictionary records < rule 1
        dict2 will hold all the dictionary records < rule 2
        ....
        dict(n) will hold all the dictionary records < rule n
        dict(n+1) will hold all the dictionary records >= rule n
    """

    def __init__(self, setting, core):
        self.core = core
        self.lamda = int(setting['lambda'])
        self.rules = json.loads(setting['rule'])
        self.rules_len = len(self.rules) + 1
        self.master_dict = {}
        self.pcx_set = {
            'dataset': 0
        }

        #this will load initial value the pc1_set .. pc(n+1)_set to pcx_set
        for x in range(len(self.rules) + 1):
            label = 'pc' + str(x + 1)
            self.pcx_set[label] = 0

    def training(self, dataset):
        """ data receive in a list of list
        In: [[w1, val1], [w2, val2], ...]
        Out: {'a': [1, 2, ...], 'b': [0, 1, ...], .....} | to update to master_dict

        The first value in each key value is number of freqs of words appear in Cat 1
        the second value in each key value is number of freqs of words appear in Cat 2
        """

        start = time.time()

        """ list_gen_pcx is the list generator which hold all the SKU that match the rule,
        ex: [[w1, w2, w3], [w3, w5, w7], ...]
        """
        for data in dataset:
            for x, rule in enumerate(self.rules):
                if float(data[1]) <= rule:
                    idx = x
                    break
            else:
                idx = len(self.rules)

            for word in data[0].split():
                if word not in self.master_dict:
                    self.master_dict[word] = [0] * self.rules_len
                self.master_dict[word][idx] += 1
            self.pcx_set['pc' + str(idx + 1)] += 1
            self.pcx_set['dataset'] += 1

        print('Training of', len(dataset), 'words takes', round(time.time() - start, 4), 's')
        print("Dict size:", len(self.master_dict))

    def validate(self, data):
        """ Calculate the data input according to master dict and calculate the division.
        Create two numpy array of 1 x m which hold all records:
        + Pbx_Cy = P(x | C)
        + Pb_C = P(C)
        then create 2 array of m x m for P(xi | Cj) and P(Cj) / P(Ci)

        the category satisfied that which row has the min value > 1 is the chosen Cat

        In: [word, value]
        Out: [True / False, [score]]
        """
        words_list = data[0].split()

        Fr_C = np.array([self.pcx_set['pc' + str(x + 1)] for x in range(self.rules_len)])
        Pb_C = Fr_C / self.pcx_set['dataset']

        Pbx_Cy = np.ones((1, self.rules_len))

        for k, v in self.master_dict.items():
            if k in words_list:
                Pbx_Cy *= (np.array(v) + 1) / (Fr_C + 1)
            else:
                Pbx_Cy *= 1 - (np.array(v) + 1) / (Fr_C + 1)

        lambda_X = np.array([])
        with np.errstate(divide='ignore', invalid='ignore'):
            for rule in range(self.rules_len):
                tmp_rs = Pbx_Cy[0][rule] / Pbx_Cy - self.lamda * Pb_C / Pb_C[rule]
                lambda_X = np.append(lambda_X, tmp_rs)

        lambda_X = lambda_X.reshape(self.rules_len, self.rules_len)

        #select in lambda_X which row (cat) has the min >= 0
        min_lambda = np.amin(lambda_X, axis=1) #return array of min value of each row
        cat_pos = np.nanargmax(min_lambda) #since we only have 1 solution which min is >=0

        for x, rule in enumerate(self.rules):
            if float(data[1]) <= rule:
                cat_rule = x
                break
        else:
            cat_rule = len(self.rules)

        return [cat_pos == cat_rule, lambda_X, cat_pos, cat_rule]

    def prepare_call(self, name, args):
        """ creates a 'remote call' package for each argument """
        for arg in args:
            yield [self.__class__.__name__, self.__dict__, name, arg]

    def run(self, dataset, output_path=None):
        """ Process the dataset by dividing the data stream into multiple process to speed
        up the validiton
        In: [[w1, val1], [w2, val2], ...]
        Sub-Out: [[True/False, score], ......]
        Out: % of sub-out is True and list of item is False and it's score
        """
        process_pool = Pool(processes=self.core)

        st = time.time()
        result = process_pool.map(parallel_call, self.prepare_call("validate", dataset))

        process_pool.close()

        rs_counter = Counter([x[0] for x in result])

        print('Validated', len(dataset), 'records - finish in', round(time.time() - st, 2))
        print(round(rs_counter[True] / len(dataset) * 100, 1), '% predicted correct')

        #output into log
        if output_path is not None:
            file_name = output_path + 'NBaye-{:1.0f}'.format(time.time())
            with open(file_name + '-dict.csv', 'w', encoding='utf-8') as wr:
                for k, v in self.master_dict.items():
                    wr.write('"{}": {}\n'.format(k, v))
            with open(file_name + '-test.csv', 'w', encoding='utf-8') as wr:
                for data, ele in zip(dataset, result):
                    wr.write('{}\n'.format(data))
                    wr.write('{}\n\n'.format(ele))

# For internal testing
if __name__ == '__main__':
    settng = {
        'rule': '[20, 30, 40]',
        'lambda': 1
    }

    import os

    dr = os.path.dirname(__file__)
    feeder = os.path.join(dr, '\\data_feed\\machine_feed.csv')

    with open(feeder, 'r', encoding='utf-8') as rd:
        csvreader = csv.reader(rd, delimiter=',', quotechar='"')
        dt = [[x[1], x[4]] for x in list(csvreader)[1:10000]]

    a = NBaye(settng, 4)

    try:
        for z, y in enumerate(range(10)):
            print(z)
            a.training(dt[y * 1000 : (y + 1) * 1000])

            a.run(dt[(y + 1) * 1000 : (y + 2) * 1000])
            print('----------------------\r\n')
    except KeyboardInterrupt:
        print("cancel...")
