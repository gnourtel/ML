# pylint: disable=C0103
""" Naive Bayessian Classifer """

from multiprocessing import Pool
from collections import Counter
import time

class NBaye():
    """ N Bayesian object for trainning
        the trainin sample will be splitted into 2 list, one that < than the pre-set rule and the
        other one > rule
        dict1 will hold all the dictionary records < rule
        dict2 will hold all the dictionary records > rule
    """
    master_dict = {}
    dataset = 0
    pc1_set = 0
    pc2_set = 0

    def __init__(self, setting, core):
        self.core = core
        self.lamda = int(setting['lambda'])
        self.rule = int(setting['rule'])

    def training(self, dataset):
        """ data receive in a list of list
        In: [[w1, val1], [w2, val2], ...]
        Out: {'a': [1, 2], 'b': [0, 1], .....} | to update to master_dict

        The first value in each key value is number of freqs of words appear in Cat 1
        the second value in each key value is number of freqs of words appear in Cat 2
        """
        start = time.time()

        list_gen_pc1 = (x[0].split() for x in dataset if float(x[1]) <= self.rule)
        list_word_pc1 = [words for row in list_gen_pc1 for words in row]
        list_gen_pc2 = (x[0].split() for x in dataset if float(x[1]) > self.rule)
        list_word_pc2 = [words for row in list_gen_pc2 for words in row]

        #Building the dict
        pc1_dict = Counter(list_word_pc1)
        pc2_dict = Counter(list_word_pc2)

        #Update the dict into master dict
        for k, v in pc1_dict.items():
            if k in self.master_dict:
                self.master_dict[k][0] += v
            else:
                self.master_dict[k] = [v, 0]

        for k, v in pc2_dict.items():
            if k in self.master_dict:
                self.master_dict[k][1] += v
            else:
                self.master_dict[k] = [0, v]

        #Update the sample amount]
        self.dataset += len(dataset)
        self.pc1_set += sum(1 for x in dataset if float(x[1]) <= self.rule)
        self.pc2_set = self.dataset - self.pc1_set

        print('Training of', len(dataset), 'words takes', round(time.time() - start, 5), 's')

    def validate(self, data):
        """ Calculate the data input according to master dict and calculate the division
        of P(data | C1) / P(data | C2) and compare to P(C2) / P(C1)
        In: [word, value]
        Out: True / False
        """
        words_list = data[0].split()
        px_c1 = 1
        px_c2 = 1
        for k, v in self.master_dict.items():
            if k in words_list:
                px_c1 *= (v[0] + 1) / (self.pc1_set + 1)
                px_c2 *= (v[1] + 1) / (self.pc2_set + 1)
            else:
                px_c1 *= 1 - (v[0] + 1) / (self.pc1_set + 1)
                px_c2 *= 1 - (v[1] + 1) / (self.pc2_set + 1)

        lambda_X = px_c1 / px_c2
        machine_rs = True if (lambda_X > self.lamda * self.pc2_set / self.pc1_set) else False
        rule_compare = float(data[1]) <= self.rule
        return [machine_rs == rule_compare, lambda_X]

    def run(self, dataset):
        """ Process the dataset by dividing the data steam into multiple process to speed
        up the validiton
        In: [[w1, val1], [w2, val2], ...]
        Sub-Out: [[True/False, score], ......]
        Out: % of sub-out is True and list of item is False and it's score
        """
        process_pool = Pool(processes=self.core)

        start = time.time()
        result = process_pool.map(self.validate, [x for x in dataset])

        process_pool.close()

        rs_counter = Counter(result)

        print('Validated', len(dataset), 'records - finish in', round(time.time() - start, 2))
        print(round(rs_counter / len(dataset) * 100, 1),
              '% predicted correct - dictionary size of ', len(self.master_dict))

        #output into log

###testing
import csv
with open('C:\\Users\\Truong Le Nguyen\\Desktop\\ML\\data_feed\\machine_feed.csv', encoding='utf-8') as rd:
    csvreader = csv.reader(rd)
    dt1 = list(csvreader)[1:2000]
    dt = [[x[1], x[4]] for x in dt1]

stng = {
    'lambda': '1',
    'rule': '20'
}

test = NBaye(stng, 2)
test.training(dt[:1000])

b = time.time()

a = []
for x in range(990):
    x += 1000
    a.append(test.validate(dt[x]))

# p = Pool(processes=4)

# rs = p.map(test.validate, dt[1000:])

print(time.time() - b)
