import loading_data
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier


"""DECISION TREE """

dtree_model = DecisionTreeClassifier(max_depth = 6).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 

print(accuracy_score(y_true=y_test,y_pred=dtree_predictions))
print(classification_report(y_test,dtree_predictions)) 
# creating a confusion matrix 
print(confusion_matrix(y_test, dtree_predictions) )

"""Accuracy : 65.95744680851063
 precision    recall  f1-score   support

       angry       0.74      0.74      0.74        90
       happy       0.57      0.53      0.55        94
     neutral       0.58      0.68      0.62        44
         sad       0.71      0.69      0.70       101

    accuracy                           0.66       329
   macro avg       0.65      0.66      0.65       329
weighted avg       0.66      0.66      0.66       329

[[67 21  0  2]
 [17 50  8 19]
 [ 4  3 30  7]
 [ 3 14 14 70]] """


"""SUPPORT VECTOR MACHINE"""


svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 


print(accuracy_score(y_true=y_test,y_pred=svm_predictions))
print(classification_report(y_test,svm_predictions)) 
# creating a confusion matrix 
print(confusion_matrix(y_test, svm_predictions) )




""" Accuracy :
0.7507598784194529
              precision    recall  f1-score   support

       angry       0.78      0.79      0.78        90
       happy       0.73      0.70      0.71        94
     neutral       0.65      0.73      0.69        44
         sad       0.80      0.77      0.78       101

    accuracy                           0.75       329
   macro avg       0.74      0.75      0.74       329
weighted avg       0.75      0.75      0.75       329

[[71 13  4  2]
 [12 66  4 12]
 [ 3  3 32  6]
 [ 5  9  9 78]]"""
 
 
 """Random Forest"""
 
 
classifier = RandomForestClassifier(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
classifier.fit(X_train, y_train)   

c_p = classifier.predict(X_test) 


print(accuracy_score(y_true=y_test,y_pred=c_p))
print(classification_report(y_test,c_p)) 
# creating a confusion matrix 
print(confusion_matrix(y_test,c_p) )

"""Accuracy : 
0.7142857142857143
              precision    recall  f1-score   support

       angry       0.79      0.86      0.82        90
       happy       0.68      0.61      0.64        94
     neutral       0.72      0.59      0.65        44
         sad       0.68      0.74      0.71       101

    accuracy                           0.71       329
   macro avg       0.72      0.70      0.70       329
weighted avg       0.71      0.71      0.71       329

[[77 10  0  3]
 [14 57  4 19]
 [ 0  4 26 14]
 [ 7 13  6 75]]"""
 
"""Note : The highest accuracy obtained models are uploaded here.....Refer Speechemotion_ml_algorithms ipynb file from my colab"""
