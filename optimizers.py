""" Optimizer for searching the optimal normalization methods """

import numpy
from methods import wrapper
import os
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score

class evaluation:    
    def evaluate(self, X, y, clf, sol, cv, mapping, cpus):
        P = sol.shape[0]
        err = numpy.zeros(P)
        wr = wrapper()
        for i in  range(0, P):
            Xn, _ = wr.ndata(X, X, sol[i,:], mapping)
            out = cross_validate(clf, Xn, y, cv=cv, n_jobs = cpus)
            err[i] = 1 - numpy.mean(out['test_score'])
        return err
    
    def predict(self, X_train, y_train, X_test, y_test, clf, sol, mapping, cpus):
        wr = wrapper()
        Xn_train, Xn_test = wr.ndata(X_train, X_test, sol, mapping)
        clf.fit(Xn_train, y_train)
        pred_label = clf.predict(Xn_test)
        err = 1 - accuracy_score(y_test, pred_label)
        return err

class initializer:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub
    def initialization(self, P):
        dim = self.lb.shape
        pos = numpy.zeros([P, dim[0]]);
        for i in range(0, P):
            for j in range(0, dim[0]):
                pos[i,j]=numpy.random.uniform(self.lb[j], self.ub[j], 1)
        return pos

class ALO:
    def __init__(self, P = 10, I = 100, cpus = None, viewi = 0):
        self.P = P
        self.I = I
        self.cpus = cpus
        self.viewi = viewi

    def sort_pos(self,sort_index,antlions,P):
        npos = numpy.copy(antlions)
        for i in range(0,P):
            #print(i)
            npos[i,:] = antlions[int(sort_index[i]),:]
        return npos
    
    def RWSelection(self, acc, P):
        acc = [1/x for x in acc]
        cumulative_sum = numpy.cumsum(acc)
        p = numpy.random.rand()*cumulative_sum[P-1]
        idx = -1
        for i in range(0,P):
            if(cumulative_sum[i] > p):
                idx = i
                break
        if(idx == -1):
            idx = 0
        return idx
    
    def RW(self,dim,lb,ub,Iter,current_iter,AL):
        I = 1
        ratio = current_iter/Iter
        if (current_iter > Iter*0.95):
            I=1+50*ratio
        elif (current_iter > Iter*0.90):
            I=1+20*ratio
        elif (current_iter > Iter*0.75):
            I=1+10*ratio
        elif (current_iter > Iter*0.50):
            I=1+5*ratio
        
        lb=lb / I
        ub=ub / I
        if(numpy.random.rand() < 0.5):
            lb = lb + AL
        else:
            lb = -lb + AL
        
        if(numpy.random.rand()>0.5):
            ub = ub + AL
        else:
            ub = -ub + AL
        RW=[]
        for i in range(0,dim):
            X = numpy.cumsum(2*(numpy.random.rand(Iter,1)>0.5)-1)
            a = min(X)
            b = max(X)
            c = lb[i]
            d = ub[i]
            tmp1 = (X-a)/(b-a)
            X_norm = c + tmp1*(d-c)
            RW.append(X_norm)
        return RW
        
    def ALO(self, X, y, clf, cv, mapping):
        dim = X.shape[1]
        lb = numpy.ones(dim)
        ub = len(mapping) * numpy.ones(dim)
        inz = initializer(lb, ub)
        antlions = inz.initialization(self.P)
        antlions = numpy.rint(antlions)
        ev = evaluation()
        err = ev.evaluate(X, y, clf, antlions, cv, mapping, self.cpus)
        # print(err)
        sort_index = numpy.argsort(numpy.array(err))
        err.sort()
        sorted_antlions = self.sort_pos(sort_index,antlions, self.P)
        
        Elite_err = numpy.copy(err[0])
        Elite_pos = numpy.copy(sorted_antlions[0,:])
        CC = []
        CC.append(Elite_err)
        current_iter = 2
        while current_iter <= self.I:
            if(self.viewi):
                print(current_iter)
            ant_pos = numpy.zeros([self.P,dim])
            for i in range(0, self.P):
                idx = self.RWSelection(err, self.P)
                RW_EL = numpy.array(self.RW(dim, lb, ub, self.I, current_iter, Elite_pos))
                RW_RWS = numpy.array(self.RW(dim, lb, ub, self.I, current_iter, sorted_antlions[idx,:]))
                         
                pos = (RW_EL[:,current_iter-1] + RW_RWS[:,current_iter-1]) / 2
                pos = numpy.rint(pos)
                for j in range(0,dim):
                    ant_pos[i,j] = max(min(pos[j],ub[j]),lb[j])
            err1 = ev.evaluate(X, y, clf, ant_pos, cv, mapping, self.cpus)
            total_fitness = numpy.append(err,err1)
            total_pos = numpy.concatenate((sorted_antlions,ant_pos),axis=0)
            sort_index = numpy.argsort(numpy.array(total_fitness))
            total_fitness.sort()
            tmp_sorted_antlions = self.sort_pos(sort_index,total_pos, 2 * self.P)
            err=numpy.copy(total_fitness[0:self.P])   
            sorted_antlions = numpy.copy(tmp_sorted_antlions[0:self.P,:])
            Elite_err = numpy.copy(err[0])
            Elite_pos = numpy.copy(sorted_antlions[0,:])
            CC.append(Elite_err)
            current_iter = current_iter + 1
        return Elite_err,Elite_pos,CC
    
class optimizers:
    def __init__(self, optimizer= 'ALO', Population = 10, Iterations = 100, cpus = None, viewi = 0):
        self.optimizer = optimizer
        self.Population = Population
        self.Iterations = Iterations
        self.cpus = cpus
        self.viewi = viewi
    
    def search(self, X, y, clf, cv, mapping):
        if(self.cpus != None and self.cpus > os.cpu_count()):
                print('CPUs request exceed the system cores. Resetting cpus = Max. Cores')
                self.cpus = os.cpu_count()
        if(self.optimizer ==  'ALO'):
            Opt = ALO(self.Population, self.Iterations, self.cpus, self.viewi)
            best, sol, conv = Opt.ALO(X, y, clf, cv, mapping)
        return best, sol, conv
        