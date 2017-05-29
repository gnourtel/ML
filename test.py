from multiprocessing import Pool

class A():
    def __init__(self, vl):
        self.vl = vl
    def cal(self, nb):
        return nb * self.vl
    def run(self, dt):
        t = Pool(processes=4)
        rs = t.map(self.cal, dt)
        t.close()
        return t

a = A(2)

a.run(list(range(10)))