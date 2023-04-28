# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:32:46 2023

@author: RAMBABU
"""


import os 
os.chdir('D:/rambabu_r@nrsc.gov.in/STUDY/MTECH/IIST GeoInformatics/2 SEM/PRML/Project/PYDAL')

#===============================================================
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modAL.uncertainty import uncertainty_sampling
from modAL.models import ActiveLearner
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import seaborn as sns
from plyfile import PlyData, PlyElement

# Load the PLY file
ply_data = PlyData.read('toronto_segmented.ply')

# Print the attributes
for element in ply_data.elements:
    print('Element name:', element.name)
    print('Element properties:')
    for property in element.properties:
        print('-', property.name)

points_element = ply_data['vertex']
points = np.vstack([points_element['x'], points_element['y'], points_element['z']]).T

import pandas  as pd
df = pd.DataFrame(points)
df.info()

print(df)

fig, axes = plt.subplots(1, 3, figsize=(15, 15))
s=sns.boxplot(y=df[0], ax=axes[0],color="#FC9803")
axes[0].set_title('X Coordinate')
s=sns.boxplot(df[1], ax=axes[1],color="#FC9803")
axes[1].set_title('Y Coordinate')
s=sns.boxplot(df[2], ax=axes[2],color="#FC9803")
axes[2].set_title('Z Coordinate')

pair2=sns.pairplot(df,diag_kind= 'kde',corner=True,plot_kws=dict(s=7, edgecolor="r", linewidth=1))
pair2.savefig('pairplot.png')


# Get the labels
labels = points_element['scalar_Label']


np.unique(labels)
#array([0., 1., 2., 3., 4., 5., 6., 7.], dtype=float32)

pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(points)

import matplotlib.pyplot as plt

# (Optional) Set the colors for each point using intensity values from the PLY file
colors = plt.get_cmap('jet')(labels/np.max(labels))
pcd1.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd1])


# split the dataset into a labeled set and an unlabeled pool set
X_labeled, X_pool, y_labeled, y_pool = train_test_split(
    points, labels, test_size=0.3, stratify=labels)

# initialize a random forest classifier as the learner
learner = ActiveLearner(
    estimator=RandomForestClassifier(n_estimators=100),
    X_training=X_labeled, y_training=y_labeled,
    query_strategy=uncertainty_sampling)

# set the number of iterations for the active learning loop
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

# calculate the accuracy of the learner on the entire dataset
y_pred = learner.predict(points)
accuracy = accuracy_score(labels, y_pred)
print(f"Final Accuracy: {accuracy:.3f}")

label_colors = np.zeros((len(points), 3))
label_colors[y_pred == 0] = [1, 0, 0]  # class 0 in red
label_colors[y_pred == 1] = [0, 1, 0]  # class 1 in green
label_colors[y_pred == 2] = [0, 0, 1]  # class 2 in blue
label_colors[y_pred == 3] = [1, 1, 0]  # class 3 in yellow
label_colors[y_pred == 4] = [1, 0, 1]  # class 4 in magenta
label_colors[y_pred == 5] = [0, 1, 1]  # class 5 in cyan
label_colors[y_pred == 6] = [1, 1, 1]  # class 6 in white
label_colors[y_pred == 7] = [0, 0, 0]
# Create an Open3D point cloud object for the test set
test_pcd = o3d.geometry.PointCloud()
test_pcd.points = o3d.utility.Vector3dVector(points)
test_pcd.colors = o3d.utility.Vector3dVector(label_colors)


# Visualize the point cloud
o3d.visualization.draw_geometries([test_pcd])

target_names = ['Unclassified 0', 'Ground 1', 'Road_markings 2','Natural 3','Building 4',
                'Utility_line 5','Pole 6','Car 7','Fence 8']

print(classification_report(labels, y_pred, target_names=target_names, zero_division=1))

print(f'Test Accuracy: {accuracy_score(labels, y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(labels, y_pred)}')

from sklearn.metrics import roc_auc_score,roc_curve, auc
from sklearn.preprocessing import LabelBinarizer

# assume y_test and y_pred are the true and predicted labels respectively
# convert the labels to one-hot encoded form
lb = LabelBinarizer()
lb.fit(labels)
y_test_one_hot = lb.transform(labels)
y_pred_one_hot = lb.transform(y_pred)

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


