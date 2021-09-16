""" Data and feature wise normalization """

import numpy
from methods import get_methods
from matplotlib import pyplot as plt
from optimizers import optimizers,evaluation

class plots:
    def make_plot(self,conv,base_errs,methods= 'ALL', include_un = True):
        f, axes = plt.subplots(1,1)
        plt.rcParams["figure.figsize"] = (6, 3)
        mths = get_methods(mtype = methods, include_un = include_un)
        mapping = dict(zip(range(1, len(mths) + 1),mths))
        mn = numpy.argmin(base_errs)
        mx = numpy.argmax(base_errs)
        mins = numpy.repeat(base_errs[mn], len(conv))
        maxs = numpy.repeat(base_errs[mx], len(conv))
        idx = -1
        for k,i in enumerate(list(mapping.values())):
            if(i == 'UN'):
                idx = k
                break
        base = numpy.repeat(base_errs[idx], len(conv))
        plt.plot(conv, '-', color = 'black', label='FWN')
        plt.plot(mins, '-', color = 'red', label='$DWN_{' + mapping.get(mn+1) + '}$')
        plt.plot(maxs, '-', color = 'blue', label='$DWN_{' + mapping.get(mx+1) + '}$')
        if(idx != -1):
            plt.plot(base, '-', color = 'green', label='$DWN_{UN}$')
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15), frameon=False, ncol=4)
        # plt.show()
        plt.tight_layout()
        f.savefig('./sample.png',format='png',bbox_inches = "tight",pad_inches = 0,dpi=600)
        f.clf()
        

class DWN:
    def __init__(self, methods = 'ALL', include_un = True):
        self.methods = methods
        self.include_un = include_un
        
    def fitcv(self, X, y, clf, cv):
        mths = get_methods(mtype = self.methods, include_un = self.include_un)
        mapping = dict(zip(range(1, len(mths) + 1),mths))
        ev = evaluation()
        feat = X.shape[1]
        errs = numpy.zeros(len(mths))
        for i in range(1, len(mths) + 1):
            n_vec = i * numpy.ones((1,feat),dtype = int)
            errs[i-1] = ev.evaluate(X, y, clf, n_vec, cv, mapping, None)
        return errs
    
    def fit(self, X_train, y_train, X_test, y_test, clf):
        mths = get_methods(mtype = self.methods, include_un = self.include_un)
        mapping = dict(zip(range(1, len(mths) + 1),mths))
        ev = evaluation()
        feat = X_train.shape[1]
        errs = numpy.zeros(len(mths))
        for i in range(1, len(mths) + 1):
            n_vec = i * numpy.ones(feat, dtype = int)
            errs[i-1] = ev.predict(X_train, y_train, X_test, y_test, clf, n_vec, mapping, None)
        return errs

class FWN:
    def __init__(self, methods = 'ALL', include_un = True, optimizer = 'ALO', Population = 10, Iterations = 100, cpus = None, viewi = 0):
        self.methods = methods
        self.include_un = include_un
        self.optimizer = optimizer
        self.Population = Population
        self.Iterations = Iterations
        self.cpus = cpus
        self.viewi = viewi
        
    def fit(self, X, y, clf, cv):
        mths = get_methods(mtype = self.methods, include_un = self.include_un)
        mapping = dict(zip(range(1, len(mths) + 1),mths))
        H = optimizers(Population = self.Population, Iterations = self.Iterations, cpus = self.cpus, viewi = self.viewi)
        best, sol, conv = H.search(X, y, clf, cv, mapping)
        return best, sol, conv, mapping
    
    def predict(self, X_train, y_train, X_test, y_test, clf, sol, mapping):
        ev = evaluation()
        err = ev.predict(X_train, y_train, X_test, y_test, clf, sol, mapping, None)
        return err