import loading_data
#FIRST MODEL
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
model.fit(X_train, y_train)

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
 """Obtained accuracy is Accuracy: 82.07%"""
"""Printing classification report and confusion matrix"""
 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
"""
precision    recall  f1-score   support

       angry       0.94      0.86      0.90        90
       happy       0.81      0.80      0.80        94
     neutral       0.70      0.73      0.71        44
         sad       0.81      0.86      0.83       101

    accuracy                           0.82       329
   macro avg       0.81      0.81      0.81       329
weighted avg       0.83      0.82      0.82       329

[[77  9  2  2]
 [ 5 75  5  9]
 [ 0  2 32 10]
 [ 0  7  7 87]]"""

"""Second mlp model after tuning the parameters"""

m_params = {
    'alpha': 0.01,
    'batch_size': 200,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
m1 = MLPClassifier(**m_params)

# train the model
print("[*] Training the model...")
m1.fit(X_train, y_train)

# predict 25% of data to measure how good we are
y_p = m1.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_p)

print("Accuracy: {:.2f}%".format(accuracy*100))

# now we save the model
# make result directory if doesn't exist yet
"""Accuracy:82%"""
m_params = {
    'alpha': 0.01,
    'batch_size': 200,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
m1 = MLPClassifier(**m_params)

# train the model
print("[*] Training the model...")
m1.fit(X_train, y_train)

# predict 25% of data to measure how good we are
y_p = m1.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_p)

print("Accuracy: {:.2f}%".format(accuracy*100))

# now we save the model
# make result directory if doesn't exist yet
#Accuracy:80%
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
print(classification_report(y_test,y_p))
print(confusion_matrix(y_test,y_p))

"""precision    recall  f1-score   support

       angry       0.93      0.89      0.91        90
       happy       0.69      0.91      0.79        94
     neutral       0.76      0.59      0.67        44
         sad       0.86      0.71      0.78       101

    accuracy                           0.80       329
   macro avg       0.81      0.78      0.78       329
weighted avg       0.82      0.80      0.80       329

[[80  9  1  0]
 [ 4 86  1  3]
 [ 0  9 26  9]
 [ 2 21  6 72]]"""



"""NOTE : Refer Speechemotion_mlp.ipynb...my colab notebook for better understanding"""
