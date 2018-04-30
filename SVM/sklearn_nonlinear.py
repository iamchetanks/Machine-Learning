from sklearn import svm
import numpy as np
import pandas as pd


df = pd.read_csv('data/nonlinsep.txt', sep = ',', header = None)
feats = df.as_matrix(columns = [0,1])
labels = df.as_matrix(columns = [2])
labels = np.ravel(labels)

clf = svm.SVC(kernel='rbf')
clf.fit(feats, labels)
#print (len(feats))
#print ("weight", clf.coef_)

preds = clf.predict([[0.003218655523771541,0.8500212633219006]])

print ("Predicted Label", preds)

print ("Support Vectors",clf.support_vectors_)
print ("bias", clf.intercept_)
