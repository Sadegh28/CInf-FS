from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

def classify(X_train, y_train, X_test, y_test, classifier, mtrs): 
    results = {}
    
    
    if classifier == 'SVM':               
        clf  = SVC(random_state=42)
    
    
    prediction = clf.fit(X_train, y_train).predict(X_test)
    for m in mtrs: 
            

        if m == 'average precision score':
            results[m] = metrics.average_precision_score(y_test, prediction)

        if m == 'f1_score':
            results[m] = metrics.f1_score(y_test, prediction, average = 'weighted')

        if m == 'accuracy_score':
            results[m] = metrics.accuracy_score(y_test, prediction)

            
        
    return results