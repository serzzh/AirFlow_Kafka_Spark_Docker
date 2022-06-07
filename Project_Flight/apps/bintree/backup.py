from sklearn import tree
import numpy as np
import json, os, sys

rule = {
    'n_features_in_': 2,
    'feature_names': ['Cu50', 'pulp_level'],
    'n_classes_': 2,
    'classes_': np.array([0, 1]), #ID рекомендации
    'tree_': {
        'node_count': 3,
        'children_left':  [1, -1, -1,],
        'children_right': [2, -1, -1,],
        'feature': [0, -2, -2,],
        'threshold': [0.5, -2, -2],
        'class': [-2, 0, 1,],
        #'value': [0]
        # value : array of double, shape [node_count, n_outputs, max_n_classes]
        # Contains the constant prediction value of each node.
    }
}

rule = {
    'n_features_in_': 3,
    'feature_names': ['Cu50', 'pulp_level', 'bulb_torn'],
    'n_classes_': 11,
    'classes_': np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]), #ID рекомендации
    'node_count': 21,
    'tree_': {
        'node_count': 21,
        'children_left':  [1, 3, 5, 7, 13, 16, 19, 10, 14, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        'children_right': [2, 4, 6, 12, 8, 9, 20, 11, 15, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        'feature': [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        'threshold': [20.5, 19.5, 21.5, 19, 72, 72, 72, 18.5, 0.5, 0.5, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        'class': [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    }
}



with open("rule.json", "w") as write_file:
    json.dump(rule, write_file, indent=4)

nrows = 10
X = np.ones((nrows, rule['n_features_in_']))
y = np.ones((nrows,1))

'''
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
'''

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)


clf.tree_.value[0] = [[[50, 50]], [[50, 0]], [[0, 50]]]
#clf.tree_.value[1] = [[0,50]]

print('values', clf.tree_.value)


# Set parameters manually 3 nodes
for attr in rule:
    if isinstance(rule[attr], dict):
        for nested_attr in rule[attr]:
            if isinstance(rule[attr][nested_attr], list):
                rng = slice(0, len(rule[attr][nested_attr]))
                getattr(clf.tree_, nested_attr)[rng] = rule[attr][nested_attr][rng]
            else:
                setattr(clf.tree_, nested_attr, rule[attr][nested_attr])
    else:
        setattr(clf, attr, rule[attr])

text_representation = tree.export_text(clf, feature_names=rule['feature_names'])
print(text_representation)


print('0.51', clf.predict([[0.51, 0.51]]))
print('0.49', clf.predict([[0.49, 0.49]]))


#python -c "from sklearn import tree; help(tree._tree.Tree)"
#json_data = json.d.umps(clf.tree_.children_left)
#print(json_data)