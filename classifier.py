""" v1.00
Building a dictionary with 11000 SKU name and taking its word's occurent
"""

from multiprocessing import Pool
import time
import csv
import collections

#Pre-define rule, C1 <- x, C2 > x
RULES = 20

def run(methods, core):
    pass

class CHECKER():
    """ Create an class to build object for each time doing search"""
    l = 5
    def __init__(self, collection, pc1, pc2, pxc1, pxc2, rules):
        self.pc1 = pc1
        self.pc2 = pc2
        self.pxc1 = pxc1
        self.pxc2 = pxc2
        self.collection = collection
        self.rules = rules
    def cal_prob(self, info):
        """ Take the text loop over to calculate P(xx|C1) and P(xx|C2)"""
        words = info[0].split()
        px_c1 = 1
        px_c2 = 1
        for word in words:
            px_c1 *= self.pxc1[word]/self.collection if word in self.pxc1 else 1
            px_c2 *= self.pxc2[word]/self.collection if word in self.pxc2 else 1
        px_c1 = px_c1 if px_c1 != 1 else 0
        px_c2 = px_c2 if px_c2 != 1 else 0
        prob_c1 = px_c1 * self.pc1 / (px_c1 * self.pc1 + px_c2 * self.pc2)
        prob_c2 = px_c2 * self.pc2 / (px_c1 * self.pc1 + px_c2 * self.pc2)
        machine_rs = True if (prob_c1 / prob_c2 >= self.l * self.pc1 / self.pc2) else False
        rule_compare = float(info[1]) <= self.rules
        return machine_rs == rule_compare

def splitword(info):
    """ check if length is < RULES and output the string with split
        if the length (4th column in data) smaller than RULES it will
        return with data split and (1 / 0) if RULES is satisfied """
    checker = 1 if float(info[1]) < RULES else 0
    return [info[0].split(), checker]

if __name__ == '__main__':
    with open('machine_feed.csv', encoding='utf-8') as rd:
        RD_FILE = csv.reader(rd, delimiter=',', quotechar='"')
        DATA_FEED = list(RD_FILE)[1:]

    START = time.time()

    FIRST_POOL = Pool(processes=4)
    RAW_SPLIT = FIRST_POOL.map(splitword, [[x[1], x[3]] for x in DATA_FEED])

    LIST_GEN_PC1 = (x[0] for x in RAW_SPLIT if x[1] == 1)
    LIST_WORD_PC1 = [words for row in LIST_GEN_PC1 for words in row]
    LIST_GEN_PC2 = (x[0] for x in RAW_SPLIT if x[1] == 0)
    LIST_WORD_PC2 = [words for row in LIST_GEN_PC2 for words in row]

    PC1_DICT = collections.Counter(LIST_WORD_PC1)
    PC2_DICT = collections.Counter(LIST_WORD_PC2)

    print('Training takes', time.time() - START, 's with', len(DATA_FEED),
          '\nPC1:', len(PC1_DICT), 'records & PC2:', len(PC2_DICT), 'records')

    START = time.time()

    PC1 = len([x for x in RAW_SPLIT if x[1] == 1]) / len(DATA_FEED)
    PC2 = 1 - PC1

    VAL = CHECKER(len(DATA_FEED), PC1, PC2, PC1_DICT, PC2_DICT, RULES)

    with open('machine_validation.csv', encoding='utf-8') as rd:
        RD_FILE = csv.reader(rd, delimiter=',', quotechar='"')
        DATA_VAL = list(RD_FILE)[1:]

    #SECOND_POOL = Pool(processes=4)
    VAL_CHECK = FIRST_POOL.map(VAL.cal_prob, [[x[1], x[3]] for x in DATA_VAL])
    FIRST_POOL.close()

    # VAL_CHECK = []
    # for x in DATA_VAL:
    #     VAL_CHECK.append(VAL.cal_prob((x[1], x[3])))

    TRUE_COUNTER = collections.Counter(VAL_CHECK)

    print('Validationg takes:', time.time() - START, 's')
    print('Validated', len(DATA_VAL), 'records | lamda =', VAL.l,
          '| correct estimation ', TRUE_COUNTER[True])
