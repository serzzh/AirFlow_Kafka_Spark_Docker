from sklearn import tree
import numpy as np
import json, os, sys

rule = {
    'n_features_in_': 3,
    'feature_names': ['Cu50', 'pulp_level', 'bulb_torn'],
    'n_classes_': 11,
    'classes_': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'node_count': 21,
    'tree_': {
        'node_count': 21,
        'children_left':  [1, 3, 5, 7, 13, 16, 19, 10, 14, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        'children_right': [2, 4, 6, 12, 8, 9, 20, 11, 15, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        'feature': [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        'threshold': [20.5, 19.5, 21.5, 19, 72, 72, 72, 18.5, 0.5, 0.5, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        'class': [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
}

with open("my.json", "w") as f:
    json.dump(rule, f)

rule["classes_"] = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

def dataset_generate(tree, n_features_in_, n_classes_):
    X = dict()
    X[0] = np.zeros(n_features_in_)
    features = list()
    target = list()
    tree['value'] = np.zeros((tree['node_count'], 1, n_classes_))
    for i in range(tree['node_count']):
        if tree['children_left'][i] > 0:
            X[tree['children_left'][i]] = X[i].copy()
            X[tree['children_left'][i]][tree['feature'][i]] = tree['threshold'][i] - abs(tree['threshold'][i])*0.0001

        if tree['children_right'][i] > 0:
            X[tree['children_right'][i]] = X[i].copy()
            X[tree['children_right'][i]][tree['feature'][i]] = tree['threshold'][i] + abs(tree['threshold'][i])*0.0001

        if tree['class'][i]>=0:
            features.append(X[i])
            target.append(tree['class'][i])
            tree['value'][i][0] = np.zeros(n_classes_)
            tree['value'][i][0][tree['class'][i]] = 3

    return features, target, tree

X, y, rule['tree_'] = dataset_generate(rule['tree_'], rule['n_features_in_'], rule['n_classes_'])

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

def rule_setattr(rule):
    # Set parameters manually
    for attr in rule:
        if isinstance(rule[attr], dict):
            for nested_attr in rule[attr]:
                if not hasattr(clf.tree_, nested_attr):
                    continue
                if isinstance(rule[attr][nested_attr], (list, type(rule['tree_']['value']))):
                    rng = slice(0, len(rule[attr][nested_attr]))
                    getattr(clf.tree_, nested_attr)[rng] = rule[attr][nested_attr][rng]
                else:
                    setattr(clf.tree_, nested_attr, rule[attr][nested_attr])
        else:
            setattr(clf, attr, rule[attr])

rule_setattr(rule)
text_representation = tree.export_text(clf, feature_names=rule['feature_names'])

print(text_representation)
print('test 20.43, 71, 0', clf.predict([[20.43, 71, 0]]))
print('test 22.43, 74, 0', clf.predict([[22.43, 74, 0]]))
print('test 21.43, 74, 0', clf.predict([[21.43, 74, 0]]))
print('test 21.43, 74, 1', clf.predict([[21.43, 74, 1]]))


#python -c "from sklearn import tree; help(tree._tree.Tree)"
#json_data = json.dumps(clf.tree_.children_left)
#print(json_data)
