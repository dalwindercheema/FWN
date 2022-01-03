# FWN - Feature Wise Normalization

This code performs the data normalization feature-wise using a wrapper based approach. It is implemented in python 3 and searches for the optimal normalization technique for each feature individually. Implementation of FWN, conventional data wise normalization (DWN) and the sample codes have been included in this repository.

How to Use
-----------

1. Download the code and install the REQUIRED PACKAGES
```
* numpy == 1.19.5
* scikit_learn == 0.24.2
* matplotlib == 3.4.2
```
2. Select the pool of normalization methods to normalize features.
3. Select the classifier and data evaluation procedure. During the search, Cross-Validation is recommended to avoid overfitting. It is worth mentioning that the scaling methods are insensitive to the TREE-based classifiers such as CART, Random forests, xgboost and other similar variants.

Inputs
-----------
```
Abbreviations of normalization methods:
UN  : Un-normalized (or original) feature
MN  : Mean Centered Normalization
ZS  : Z-score Normalization
PS  : Pareto Scaling
VSS : Variable Stability Scaling
PT  : Power Transformation
MM  : Min-Max
MX  : Max
DS  : Decimal Scaling
MD  : Median and Median Absolute Deviation
TN  : Tanh
VTN : Variant Tanh
HT  : Hyperbolic Tangent

PARAMETERS

<methods> : The normalization methods have been categorized into several sets for easy selection. The possible inputs are:
    "ALL": ALL 12 methods
    "MS" : Mean and Standard deviation based methods ['MC','ZS','PS','VSS','PT','TH','VTH','SG']
    "MM" : Minimum and Maximum value based methods ['MM','MX']
    "SC" : Scaling methods ['MC','ZS','PS','VSS','MM','MX','DS','MD']
    "TS" : Transformation methods ['PT','TH','VTH','SG'] 
    OR
    A list of handpicked methods can also be supplied. For instance : ['ZS','MM']
<include_un> = Include original feature in search (True or False), Default: False 
<Population> : Population of optimizer
<Iterations> : Total iterations
<cpus>       : CPUs to perform computations: None for serial and >1 for parallel, Default: None
<viewi>      : To view Iterations during search (True or False), Default: False
```
Demo
-----------
main.py provides the sample code for FWN with cross-validation and holdout data evaluation procedures. The sklearn Breast Cancer dataset is used for demonstration. Call ***main_cv()*** for the ***cross-validation*** or ***main_split()*** for the ***holdout*** style evaluation.

Citations
-----------
Dalwinder Singh and Birmohan Singh, *[Feature wise normalization: An effective way of normalizing data](https://www.sciencedirect.com/science/article/pii/S0031320321004878)*, Pattern Recognition, Volume 122, 2022

Dalwinder Singh and Birmohan Singh, *[Investigating the impact of data normalization on classification performance](https://doi.org/10.1016/j.asoc.2019.105524)*, Applied Soft Computing, Volume 97, Part B, 2020
