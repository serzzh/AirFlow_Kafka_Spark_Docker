from sklearn import tree
import numpy as np

from typing import List
from typing import Any
from dataclasses import dataclass
import json

@dataclass
class Tree:
    node_count: int
    children_left: List[int]
    children_right: List[int]
    feature: List[int]
    threshold: List[float]
    leaf_class_: List[int]

    @staticmethod
    def from_dict(obj: Any) -> 'Tree':
        _node_count = int(obj.get("node_count"))
        _children_left = [y for y in obj.get("children_left")]
        _children_right = [y for y in obj.get("children_right")]
        _feature = [y for y in obj.get("feature")]
        _threshold = [y for y in obj.get("threshold")]
        _leaf_class_ = [y for y in obj.get("leaf_class_")]
        return Tree(_node_count, _children_left, _children_right, _feature, _threshold, _leaf_class_)

@dataclass
class Rule:
    n_features_in_: int
    feature_names: List[str]
    n_classes_: int
    classes_: List[int]
    node_count: int
    tree_: Tree

    @staticmethod
    def from_dict(obj: Any) -> 'Rule':
        _n_features_in_ = int(obj.get("n_features_in_"))
        _feature_names = [y for y in obj.get("feature_names")]
        _n_classes_ = int(obj.get("n_classes_"))
        _classes_ = [y for y in obj.get("classes_")]
        _node_count = int(obj.get("node_count"))
        _tree_ = Tree.from_dict(obj.get("tree_"))
        return Rule(_n_features_in_, _feature_names, _n_classes_, _classes_, _node_count, _tree_)

# Example Usage
jsonstring = json.loads('''{"n_features_in_": 3, "feature_names": ["Cu50", "pulp_level", "bulb_torn"], "n_classes_": 11, "classes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "node_count": 21, "tree_": {"node_count": 21, "children_left": [1, 3, 5, 7, 13, 16, 19, 10, 14, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], "children_right": [2, 4, 6, 12, 8, 9, 20, 11, 15, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], "feature": [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2], "threshold": [20.5, 19.5, 21.5, 19, 72, 72, 72, 18.5, 0.5, 0.5, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2], "leaf_class_": [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}}''')
rule = Rule.from_dict(jsonstring)

'''
rule = {
    'n_features_in_': 3,
    'feature_names': .Cu50', 'pulp_level', 'bulb_torn,
    'n_classes_': 11,
    'classes_': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'node_count': 21,
    'tree_': {
        'node_count': 21,
        'children_left':  [1, 3, 5, 7, 13, 16, 19, 10, 14, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        'children_right': [2, 4, 6, 12, 8, 9, 20, 11, 15, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        'feature': [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        'threshold': [20.5, 19.5, 21.5, 19, 72, 72, 72, 18.5, 0.5, 0.5, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        'leaf_class_': [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
}

with open("rule.json", "w") as f:
    json.dump(rule, f)
'''

rule.classes_ = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print("rule:", rule)

def dataset_generate(tree, n_features_in_, n_classes_):
    X = dict()
    X[0] = np.zeros(n_features_in_)
    features = list()
    target = list()
    tree.value = np.zeros((tree.node_count, 1, n_classes_))
    for i in range(tree.node_count):
        if tree.children_left[i] > 0:
            X[tree.children_left[i]] = X[i].copy()
            X[tree.children_left[i]][tree.feature[i]] = tree.threshold[i] - abs(tree.threshold[i])*0.0001

        if tree.children_right[i] > 0:
            X[tree.children_right[i]] = X[i].copy()
            X[tree.children_right[i]][tree.feature[i]] = tree.threshold[i] + abs(tree.threshold[i])*0.0001

        if tree.leaf_class_[i]>=0:
            features.append(X[i])
            target.append(tree.leaf_class_[i])
            tree.value[i][0] = np.zeros(n_classes_)
            tree.value[i][0][tree.leaf_class_[i]] = 3

    return features, target, tree

X, y, rule.tree_ = dataset_generate(rule.tree_, rule.n_features_in_, rule.n_classes_)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

def rule_setattr(rule):
    # Set parameters manually
    rule_attributes = [x for x in dir(rule) if not x.startswith('_')]
    tree_attributes = [x for x in dir(rule.tree_) if not x.startswith('_')]

    for attr in rule_attributes:
        if isinstance(getattr(rule, attr), Tree):
            for tree_attr in tree_attributes:
                if not hasattr(clf.tree_, tree_attr):
                    continue
                if isinstance(getattr(rule.tree_, tree_attr), (list, type(rule.tree_.value))):
                    rng = slice(0, len(getattr(rule.tree_, tree_attr)))
                    getattr(clf.tree_, tree_attr)[rng] = getattr(rule.tree_, tree_attr)[rng]
                else:
                    setattr(clf.tree_, tree_attr, getattr(rule.tree_, tree_attr))
        else:
            setattr(clf, attr, getattr(rule, attr))

rule_setattr(rule)
text_representation = tree.export_text(clf, feature_names=rule.feature_names)
print(text_representation)

print('test 20.43, 71, 0', clf.predict([[20.43, 71, 0]]))
print('test 22.43, 74, 0', clf.predict([[22.43, 74, 0]]))
print('test 21.43, 74, 0', clf.predict([[21.43, 74, 0]]))
print('test 21.43, 74, 1', clf.predict([[21.43, 74, 1]]))


#python -c "from sklearn import tree; help(tree._tree.Tree)"
#json_data = json.dumps(clf.tree_.children_left)
#print(json_data)
