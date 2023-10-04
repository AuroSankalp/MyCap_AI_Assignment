# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 08:36:38 2023

@author: Gowtham S
"""

"""
How it learns ??

Cost Function and Gradient Descent

Example

Features (bmi, bp, age) -> Diabetes

20, 40, 25 -> 1 - wrong

Cost Function -> It figures out the how wrong the model is
Gradient Descent -> Will help it overcome the error

Another example
1st iteration(time) - 50 -> 30 - wrong
30 to 0 ?? - No (cannot get 100%)

30 to 15 ?? - Yes

How to achieve it?
Cost Function -> It figures out the how wrong the model is - wrong is 30
Evaluate a model using Cost function -> Wrong it Is

Gradient Descent -> Will help it overcome the error - helps 30 to 15
"""

"""
What are Python Packages ??
- Already present code For you to use
- math, sqrt -> scratch, sqrt() From math
"""

# Pandas and Scikit learn
# save the dataset in the same folder as your code

import pandas as pd

# You can look into websites but do NOT copy paste rather Type
# csv - Comma Seperated Values

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

pima = pd.read_csv("diabetes.csv", header=0, names=col_names)

"""
Dataframe - Df - Tabular data format used for data science
Why ? - Speed
Inbuilt functions that makes ur life very easy
"""

feature_cols = ['pregnant', 'insulin', 'bmi', 'age' ,'glucose', 'bp', 'pedigree']

X = pima[feature_cols]

y = pima.label


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state=16)

logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)


from sklearn import metrics

cnf = metrics.confusion_matrix(y_test, y_pred)

cnf


#array([[115,  10],
#       [ 24,  43]], dtype=int64)

"""
[[ True Positive, False Positive],
  [False Negative, True Negative]]


positive = 1
negative = 0

To DO
cnf -> Find Accuracy of this model From confusion matrix
"""

"""
KMeans

Bunch of data without labels
These data get grouped together and form clusters
These clusters gives you some insight

768 rows of diabetes data without label

# IDEALISTIC
400 forms one group - cluster 1 - diabetes patient
368 forms one group - cluster 2 - no diabetes


1. All Data 
2. Clusters - 3
3. Random centroid
4. Centroid relocates and updates
"""































