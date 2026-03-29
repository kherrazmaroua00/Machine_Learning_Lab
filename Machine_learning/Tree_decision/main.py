# decision_tree_tennis.py

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus

# -------- Step 1: Function to convert categorical data to numeric --------
def data2vector(data):
    for col in data.columns[:-1]:  
        data[col] = pd.Categorical(data[col]).codes
    return data

# -------- Step 2: Function to create a decision tree --------
def createTree(data):
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, -1].values    
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    return clf

# -------- Step 3: Function to save tree as PDF --------
def showtree2pdf(trainedTree, feature_names, filename):
    dot_data = export_graphviz(trainedTree, out_file=None,
                               feature_names=feature_names,
                               class_names=['No', 'Yes'],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf(filename)
    print(f"Decision tree saved as {filename}")

# -------- Step 4: Load the dataset --------

data = pd.read_table("tennis.txt", header=None, sep="\t")
data = data2vector(data)  

# -------- Step 5: Build the decision tree --------
decisionTree = createTree(data)
showtree2pdf(decisionTree, feature_names=['Weather', 'Temperature', 'Humidity', 'Wind'], filename='tennis.pdf')

# -------- Step 6: Predict a new sample --------
# مثال: Weather=sunny(0), Temperature=low(0), Humidity=high(1), Wind=strong(1)
testVec = np.array([0, 0, 1, 1]).reshape(1, -1)
prediction = decisionTree.predict(testVec)
print("Predicted label for testVec:", prediction)