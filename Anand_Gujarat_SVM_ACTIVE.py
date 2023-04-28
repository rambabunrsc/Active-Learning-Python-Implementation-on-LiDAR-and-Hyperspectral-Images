# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:15:24 2023

@author: RAMBABU
"""

import os
os.chdir('D:/rambabu_r@nrsc.gov.in/STUDY/MTECH/IIST GeoInformatics/2 SEM/HyperSpectral RS/LAB/Python/Anand_Gujarat')
#------------------------------------------------------------

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modAL.uncertainty import uncertainty_sampling
from modAL.models import ActiveLearner
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import rasterio
from sklearn.decomposition import PCA
#-----------------------------------------


data = rasterio.open('Anand.tif')
data=data.read()
new_data = data.reshape(data.shape[0],-1).T

new_data.shape
#(76176, 372)

new_data[0].shape
#(372,)
def applyPCA(X, numComponents=75):
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(X)
    #newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

K=6
data_matrix,pca = applyPCA(new_data,numComponents=K)

data_matrix[0].shape
#(6,)
new_data_pca = data_matrix.reshape(276,276,6)

new_data_pca.shape
#(276, 276, 6)
new_data_pca[:,:,0].shape
#(276, 276)

# to visualise the hyperspectral data we need to normalize the bands

def normalize(band):
    mx,mn = band.max(),band.min()
    return (band - mn)/(mx - mn)
r = normalize(new_data_pca[:,:,1])
g = normalize(new_data_pca[:,:,2])
b = normalize(new_data_pca[:,:,3])

fcc = np.dstack((r,g,b))

plt.imshow(fcc, cmap='jet')
plt.colorbar()
plt.show()
#----------------------------------------------------------------
gt_matrix = rasterio.open('Anand_gt.tif').read()
gt_matrix.shape
#(1, 276, 276)

gt_matrix[0].shape
#(276, 276)
plt.imshow(gt_matrix[0])

gt_matrix.min()#0
gt_matrix.max()#12


# gt_matrix_pred=gt_matrix.reshape(276*276)
#=======================================================
train = []
for i in range(new_data_pca[:,:,0].shape[0]):
    for j in range((new_data_pca[:,:,0].shape[0])):
                   if gt_matrix[0][i][j] != 0:
                       tem = list(new_data_pca[i,j,:])
                       tem.append(gt_matrix[0][i][j])
                       train.append(tem)
                       

train = np.array(train)

X = train[:,:6]
y = train[:,6]
#----------------------------------------------------
# np.unique(gt_matrix)
# #array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16],
# #       dtype=uint8)

# data_matrix=data_matrix.reshape(-1, data_matrix.shape[2])


# gt_matrix=gt_matrix.reshape(276*276)


# split the dataset into a labeled set and an unlabeled pool set
# X_labeled, X_pool, y_labeled, y_pool = train_test_split(
#     X, y, test_size=0.3,stratify=gt_matrix)
#----------------------------------------------------------
X_labeled, X_pool, y_labeled, y_pool = train_test_split(
    X, y, test_size=0.3)


svm = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
learner = ActiveLearner(
    estimator=svm,
    X_training=X_labeled, y_training=y_labeled,
    query_strategy=uncertainty_sampling)

#--------------------------------------------------------------
# initialize a random forest classifier as the learner
# learner = ActiveLearner(
#     estimator=RandomForestClassifier(n_estimators=100),
#     X_training=X_labeled, y_training=y_labeled,
#     query_strategy=uncertainty_sampling)

# set the number of iterations for the active learning loop
#--------------------------------------------------------------
n_iterations = 10

for i in range(n_iterations):
    # query the learner for the most informative instance to label
    query_idx, query_instance = learner.query(X_pool)

    # label the queried instance and add it to the labeled set
    X_labeled = np.vstack((X_labeled, X_pool[query_idx]))
    y_labeled = np.append(y_labeled, y_pool[query_idx])

    # remove the queried instance from the pool set
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx)

    # reinitialize the learner with the updated labeled set
    learner = ActiveLearner(
        estimator=RandomForestClassifier(n_estimators=100),
        X_training=X_labeled, y_training=y_labeled,
        query_strategy=uncertainty_sampling)

    # train the learner on the updated labeled set
    learner.fit(X_labeled, y_labeled)

    # use the trained learner to predict the labels of the remaining unlabeled data
    y_pred = learner.predict(X_pool)

    # calculate the accuracy of the learner on the remaining unlabeled data
    accuracy = accuracy_score(y_pool, y_pred)

    print(f"Iteration {i+1}: Accuracy = {accuracy:.3f}")
#--------------------------------------------------------------
# calculate the accuracy of the learner on the entire dataset
new_data_pca = new_data_pca.reshape(-1,6)

new_data_pca.shape
#(76176, 6)
new_data_pca[0].shape
#(6,)
y_pred = learner.predict(new_data_pca)
y_pred.shape
#(76176,)
y_pred_v = y_pred.reshape(276,276)
plt.imshow(y_pred_v,cmap='jet')
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
y_pred1 = learner.predict(X)
gt_matrix_pred=y


print(f'Test Accuracy: {accuracy_score(gt_matrix_pred, y_pred1)}')
print(f'Confusion Matrix:\n{confusion_matrix(gt_matrix_pred, y_pred1)}')


# accuracy = accuracy_score(gt_matrix, y_pred)
# print(f"Final Accuracy: {accuracy:.3f}")


# Visualize the point cloud
target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 
                'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                'Stone-Steel-Towers']

print(classification_report(gt_matrix_pred, y_pred1, labels=np.unique(y), zero_division=1))

print(f'Test Accuracy: {accuracy_score(gt_matrix_pred, y_pred1)}')
print(f'Confusion Matrix:\n{confusion_matrix(gt_matrix_pred, y_pred1)}')


from sklearn.metrics import roc_auc_score,roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

# assume y_test and y_pred are the true and predicted labels respectively
# convert the labels to one-hot encoded form
lb = LabelBinarizer()
lb.fit(gt_matrix_pred)
y_test_one_hot = lb.transform(gt_matrix_pred)
y_pred_one_hot = lb.transform(y_pred1)

# compute the AUROC score
auroc = roc_auc_score(y_test_one_hot, y_pred_one_hot, multi_class='ovo')
print(f"AUROC score: {auroc}")

# compute the ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_one_hot.shape[1]
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_one_hot[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_one_hot.ravel(), y_pred_one_hot.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# plot the ROC curves
plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"], color='darkorange', lw=lw, label='micro-average ROC curve (area = {0:0.2f})'
                                               ''.format(roc_auc["micro"]))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
#===============================================