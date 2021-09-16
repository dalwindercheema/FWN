""" Demo for the implemetation of FWN and its comparison with the conventional
    data wise normalization """
    
""" Inputs:
    Normalization Methods: The methods have been categorized into several sets 
    for easy selection. The possible inputs are:
    <methods> :
    "ALL": ALL 12 methods
    "MS" : Mean and Standard deviation based methods ['MC','ZS','PS','VSS',
                                                      'PT','TH','VTH','SG']
    "MM" : Minimum and Maximum value based methods ['MM','MX']
    "SC" : Scaling methods ['MC','ZS','PS','VSS','MM','MX','DS','MD']
    "TS" : Transformation methods ['PT','TH','VTH','SG'] 
    OR
    A list of handpicked methods can also be supplied as a list.
    For instance : ['ZS','MM']
    <include_un> = Include original feature in search (True or False), 
                    Default: False 
    
    <Population> : Population of optimizer
    <Iterations> : Total iterations
    <cpus>      : CPUs to perform computations: None for serial and >1 for 
                  parallel, Default: None
    <viewi> :     To view Iterations (True or False), Default: False 
    
    
    Call main_cv() for the cross-validation or main_split() for the holdout
    style evaluation.
    """

import sklearn.datasets as dt
from normalization import DWN,FWN, plots
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import neighbors,svm
import numpy

def main_cv():
    X, y = dt.load_breast_cancer(return_X_y=True)
    cv = StratifiedKFold(n_splits = 10,shuffle = True,random_state = 0)
    clf = neighbors.KNeighborsClassifier(7)
    # clf = svm.SVC(kernel='rbf')
    d = DWN(methods = 'MS')
    f = FWN(methods = 'MS', Population = 10, Iterations = 100, cpus = None, verbosity = 1)
    best, sol, conv, mapping = f.fit(X, y, clf, cv)
    d_errs = d.fitcv(X, y, clf, cv)
    print('Cross-validation Errors:')
    for i,j in zip(mapping.values(),numpy.around(d_errs,decimals=3)):
        print(i,'-->',j)
    print('FWN','-->',numpy.around(best,decimals=3))
    p = plots()
    p.make_plot(conv,d_errs)

def main_split():
    X, y = dt.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,shuffle = True,test_size = 0.2, random_state = 0)
    
    cv = StratifiedKFold(n_splits = 10,shuffle = True, random_state = 0)
    clf = neighbors.KNeighborsClassifier(7)
    # clf = svm.SVC(kernel='rbf')
    d = DWN(methods = 'MS')
    f = FWN(methods = 'MS', Population = 10, Iterations = 100, cpus = None, viewi = 1)
    best, sol, conv, mapping = f.fit(X_train, y_train, clf, cv)
    d_errs = d.fitcv(X_train, y_train, clf, cv)
    print('Cross-validation Errors:')
    for i,j in zip(mapping.values(),numpy.around(d_errs,decimals=3)):
        print(i,'-->',j)
    print('FWN','-->',numpy.around(best,decimals=3))
    p = plots()
    p.make_plot(conv,d_errs)
    
    print('Prediction Errors:')
    best = f.predict(X_train, y_train, X_test, y_test, clf, sol, mapping)
    d_errs = d.fit(X_train, y_train, X_test, y_test, clf)
    for i,j in zip(mapping.values(),numpy.around(d_errs,decimals=3)):
        print(i,'-->',j)
    print('FWN','-->',numpy.around(best,decimals=3))
    
    
if __name__ == "__main__":
    main_split()