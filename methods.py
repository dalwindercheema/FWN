""" Normalization methods """

import numpy
from math import sqrt
from numpy import amin
from numpy import amax

class get_methods:
    def __new__(self, mtype = 'ALL', include_un = True):
        if(type(mtype) == list):
            return mtype
        elif(type(mtype) == str):
            if(mtype == 'ALL'):
                mths = ['MC','ZS','PS','VSS','PT','MM','MX','DS','MD','TH','VTH','SG']
            elif(mtype == 'MS'):
                mths = ['MC','ZS','PS','VSS','PT','TH','VTH','SG']
            elif(mtype == 'MM'):
                mths = ['MM','MX']
            elif(mtype == 'SC'):
                mths = ['MC','ZS','PS','VSS','MM','MX','DS','MD']
            elif(mtype == 'TS'):
                mths = ['PT','TH','VTH','SG']
            else:
                print('Unknown type!!!')
                return
            if(include_un == True):
                return ['UN'] + mths
            else:
                return mths
        else:
            print('Error. Wrong input')

class wrapper:    
    def meancenter(self,train,test):
        mn = numpy.mean(train)
        n_train = train - mn
        n_test = test - mn
        return n_train,n_test
    
    def zscore(self,train,test):
        mn = numpy.mean(train)
        st = numpy.std(train)
        if(st == 0):
            st = numpy.finfo(numpy.float).eps
        n_train = (train - mn) / st
        n_test = (test - mn) / st
        return n_train,n_test
    
    def paretoscaling(self,train,test):
        mn = numpy.mean(train)
        st = numpy.std(train)
        if(st == 0):
            st = numpy.finfo(numpy.float).eps
        n_train = (train - mn) / sqrt(st)
        n_test = (test - mn) / sqrt(st)
        return n_train,n_test

    def vss(self,train,test):
        mn = numpy.mean(train)
        st = numpy.std(train)
        if(st == 0):
            st = numpy.finfo(numpy.float).eps
        cv = mn / st
        x1 = (train - mn) / st
        x2 = (test - mn) / st
        n_train = x1 * cv
        n_test = x2 * cv
        return n_train,n_test
    
    def power(self,train, test):
        train = train - amin(train, axis=0)
        test = test - amin(test, axis=0)
        train = numpy.sqrt(train)
        test = numpy.sqrt(test)
        n_train,n_test = self.meancenter(train, test)
        return n_train,n_test

    def minmax(self,train,test,ntype):
        mn = amin(train,axis=0)
        mx = amax(train,axis=0)
        x1 = (train - mn) / (numpy.finfo(numpy.float).eps + mx - mn)
        x2 = (test - mn) / (numpy.finfo(numpy.float).eps + mx - mn)
        if(ntype == 1):
            n_train = x1
            n_test = x2
        elif(ntype == -1):
            n_train = x1 * (mx - mn) - mn
            n_test = x2 * (mx - mn) - mn                
        return n_train,n_test
    
    def maxnorm(self,train,test):
        mx = amax(test,axis=0)
        n_train = train/(numpy.finfo(numpy.float).eps + mx)
        n_test = test/(numpy.finfo(numpy.float).eps + mx)
        return n_train,n_test

    def decscale(self,train, test):
        train = train - amin(train, axis=0)
        test = test - amin(test, axis=0)
        mx = amax(train,axis=0)
        f_range = numpy.ceil(numpy.log10(max(mx,numpy.finfo(numpy.float).eps)))
        n_train = train/numpy.power(10,f_range)
        n_test = test/numpy.power(10,f_range)
        return n_train,n_test

    def mmad(self,train, test):        
        med = numpy.median(train)
        x1 = abs(train - med)
        mad_value = numpy.median(x1)
        if(mad_value == 0):
            mad_value = numpy.finfo(numpy.float).eps
        n_train = (train - med) / mad_value
        n_test = (test - med) / mad_value
        return n_train,n_test
    
    def hampel(self,train, test):  
        med = numpy.median(train)
        y1 = train - med
        abs_y1 = abs(y1)
        a = numpy.quantile(abs_y1, 0.70)
        b = numpy.quantile(abs_y1, 0.85)
        c = numpy.quantile(abs_y1, 0.95)
        x1 = numpy.zeros(train.shape)
        for j in range(0,train.shape[0]):
            if (abs_y1[j] >= 0 and abs_y1[j] <= a):
                x1[j] = y1[j]
            elif(abs_y1[j] > a and abs_y1[j] <= b):
                x1[j] = a * numpy.sign(y1[j])
            elif(abs_y1[j] > b and abs_y1[j] <= c):
                tmp = a * numpy.sign(y1[j])
                x1[j] = tmp * ((c - abs_y1[j]) / (c - b))
            elif(abs_y1[j] > c):
                x1[j] = 0
        mn = numpy.mean(x1)
        st = numpy.std(x1)
        n_train = 0.5 * (numpy.tanh(0.01 * ((train - mn) / (st + numpy.finfo(numpy.float).eps))) + 1)
        n_test = 0.5 * (numpy.tanh(0.01 * ((test - mn) / (st + numpy.finfo(numpy.float).eps))) + 1)
        return n_train,n_test
      
    def hampelsimple(self,train, test):
        mn = numpy.mean(train)
        st = numpy.std(train)
        if(st == 0):
            st = numpy.finfo(numpy.float).eps
        n_train = 0.5 * (numpy.tanh(0.01 * ((train - mn) / st)) + 1)
        n_test = 0.5 * (numpy.tanh(0.01 * ((test - mn) / st)) + 1)
        return n_train,n_test
    
    def signorm(self,train, test):
        mn = numpy.mean(train)
        st = numpy.std(train)
        if(st == 0):
            st = numpy.finfo(numpy.float).eps
        x1 = (train - mn) / st
        x2 = (test - mn) / st
        n_train = (1 - numpy.exp(-x1)) / (1+numpy.exp(-x1))
        n_test = (1 - numpy.exp(-x2)) / (1+numpy.exp(-x2))
        return n_train,n_test
    
    def ndata(self, train, test, solution, d_dict):
        n_train = numpy.zeros(train.shape)
        n_test = numpy.zeros(test.shape)
        feat = train.shape[1]
        # print(d_dict)
        # print(solution.shape)
        for i in range(0,feat):
            # print(solution[i])
            method = d_dict.get(int(solution[i]))
            if(method == None):
                print('Method not specified')
                return
            if(method == 'UN'):
                n_train[:,i], n_test[:,i] = train[:,i], test[:,i]
            elif(method == 'MC'):
                n_train[:,i], n_test[:,i] = self.meancenter(train[:,i], test[:,i])
            elif(method == 'ZS'):
                n_train[:,i], n_test[:,i] = self.zscore(train[:,i], test[:,i])
            elif(method == 'PS'):
                n_train[:,i], n_test[:,i] = self.paretoscaling(train[:,i], test[:,i])
            elif(method == 'VSS'):
                n_train[:,i], n_test[:,i] = self.vss(train[:,i], test[:,i])
            elif(method == 'PT'):
                n_train[:,i], n_test[:,i] = self.power(train[:,i], test[:,i])
            elif(method == 'MM'):
                n_train[:,i], n_test[:,i] = self.minmax(train[:,i], test[:,i], 1)
            elif(method == 'MX'):
                n_train[:,i], n_test[:,i] = self.maxnorm(train[:,i], test[:,i])
            elif(method == 'DS'):
                n_train[:,i], n_test[:,i] = self.decscale(train[:,i], test[:,i])
            elif(method == 'MD'):    
                n_train[:,i], n_test[:,i] = self.mmad(train[:,i], test[:,i])
            elif(method == 'TH'):
                n_train[:,i], n_test[:,i] = self.hampel(train[:,i], test[:,i])
            elif(method == 'VTH'):
                n_train[:,i], n_test[:,i] = self.hampelsimple(train[:,i], test[:,i])
            elif(method == 'SG'):
                n_train[:,i], n_test[:,i] = self.signorm(train[:,i], test[:,i])
        return n_train, n_test
