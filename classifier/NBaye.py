# pylint: disable=C0103
""" Naive Bayessian Classifer """

# from multiprocessing import Process #, Pool
#from collections import Counter
import json
import time
import numpy as np

# import sys

# def parallel_call(params):
#     """ a helper for calling 'remote' instances """
#     cls = getattr(sys.modules[__name__], params[0])  # get our class type
#     instance = cls.__new__(cls)  # create a new instance without invoking __init__
#     instance.__dict__ = params[1]  # apply the passed state to the new instance
#     method = getattr(instance, params[2])  # get the requested method
#     args = params[3] if isinstance(params[3], (list, tuple)) else [params[3]]
#     return method(args)  # expand arguments, call our method and return the result

class NBaye():
    """ Naive Bayesian object for trainning
        from the initial loading, with n category rule, the training sample will be splitted
        into n + 1 list, the dict will be flexible adapt as following:
        dict1 will hold all the dictionary records < rule 1
        dict2 will hold all the dictionary records < rule 2
        ....
        dict(n) will hold all the dictionary records < rule n
        dict(n+1) will hold all the dictionary records >= rule n
    """
    master_dict = {}
    pcx_set = {
        'dataset': 0
    }

    def __init__(self, setting, core):
        self.core = core
        self.lamda = int(setting['lambda'])
        self.rules = json.loads(setting['rule'])
        self.rules_len = len(self.rules) + 1

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
        for x in range(self.rules_len):
            tmp = Pbx_Cy[x] / Pbx_Cy - self.lamda * Pb_C / Pb_C[x]
            lambda_X = np.append(lambda_X, tmp)

        lambda_X = lambda_X.reshape(self.rules_len, self.rules_len)

        #select in lambda_X which row (cat) has the min >= 0
        min_lambda = np.amin(lambda_X, axis=1) #return array of min value of each row
        cat_pos = np.argmax(min_lambda) #since we only have 1 solution which min is >=0

        for x, rule in enumerate(self.rules):
            if float(data[1]) <= rule:
                cat_rule = x
                break
        else:
            cat_rule = len(self.rules)

        return [cat_pos == cat_rule, lambda_X]

    # def prepare_call(self, name, args):
    #     """ creates a 'remote call' package for each argument """
    #     for arg in args:
    #         yield [self.__class__.__name__, self.__dict__, name, arg]


    # def run(self, dataset):
    #     """ Process the dataset by dividing the data stream into multiple process to speed
    #     up the validiton
    #     In: [[w1, val1], [w2, val2], ...]
    #     Sub-Out: [[True/False, score], ......]
    #     Out: % of sub-out is True and list of item is False and it's score
    #     """
    #     # process_pool = Pool(processes=self.core)

    #     start = time.time()
    #     # result = process_pool.map(self.validate, dataset)

    #     # process_pool.close()

    #     result = []
    #     for ele in dataset:
    #         j = Process(target=self.validate, args=(ele,))
    #         result.append(j)

    #     rs_counter = Counter([x[0] for x in result])

    #     print(rs_counter)
    #     print('Validated', len(dataset), 'records - finish in', round(time.time() - start, 2))
    #     print(round(rs_counter[True] / len(dataset) * 100, 1),
    #           '% predicted correct - dictionary size of ', len(self.master_dict))

    #     #output into log

# For internal testing
if __name__ == '__main__':
    settng = {
        'rule': '[20, 30, 50]',
        'lambda': 1
    }

    import csv
    with open('C:\\Users\\Truong Le Nguyen\\Desktop\\ML\\data_feed\\machine_feed.csv', 'r', encoding='utf-8') as rd:
        csvreader = csv.reader(rd)
        dt = [[x[1], x[4]] for x in list(csvreader)[1:1000]]

    a = NBaye(settng, 4)

    a.training(dt)

    pass
