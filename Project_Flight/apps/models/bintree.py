from sklearn import tree
import numpy as np
from pydantic import BaseModel, conlist
from typing import List, Any

class PredictionQuery(BaseModel):
    model_id: int = 1
    data: List[conlist(float, min_items=3, max_items=3)]

class PredictionResponse(BaseModel):
    model_id: int = 1
    prediction: int

class ModelResponse(BaseModel):
    model_id: int = 1
    tree: Any


class Tree(BaseModel):
    node_count: int
    children_left: List[int]
    children_right: List[int]
    feature: List[int]
    threshold: List[float]
    leaf_class_: List[int]
    value: Any

    @staticmethod
    def from_dict(obj: Any) -> 'Tree':
        _node_count = int(obj.get("node_count"))
        _children_left = [y for y in obj.get("children_left")]
        _children_right = [y for y in obj.get("children_right")]
        _feature = [y for y in obj.get("feature")]
        _threshold = [y for y in obj.get("threshold")]
        _leaf_class_ = [y for y in obj.get("leaf_class_")]
        return Tree(_node_count, _children_left, _children_right, _feature, _threshold, _leaf_class_)

class Model(BaseModel):
    model_id: int
    n_features_in_: int
    feature_names: List[str]
    n_classes_: int
    classes_: List[int]
    node_count: int
    tree_: Tree
    features: Any
    target: Any
    clf: Any

    def __call__(self, q: str = ""):
        if q:
            return self.fixed_content in q
        return False


    def dataset_generate(self):
        tree = self.tree_
        X = dict()
        X[0] = np.zeros(self.n_features_in_)
        features = list()
        target = list()
        tree.value = np.zeros((tree.node_count, 1, self.n_classes_))
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
                tree.value[i][0] = np.zeros(self.n_classes_)
                tree.value[i][0][tree.leaf_class_[i]] = 3

        self.features = features
        self.target = target

    def build_tree(self):
        # Precalc parameters
        self.dataset_generate()
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(self.features , self.target)
        clf_attr = dir(clf).copy()
        # Set parameters manually
        rule_attributes = [x for x in dir(self) if not x.startswith('_')]
        tree_attributes = [x for x in dir(self.tree_) if not x.startswith('_')]
        self.classes_ = np.array(self.classes_)

        for attr in rule_attributes:
            if isinstance(getattr(self, attr), Tree):
                for tree_attr in tree_attributes:
                    if not hasattr(clf.tree_, tree_attr):
                        continue
                    if isinstance(getattr(self.tree_, tree_attr), (list, type(self.tree_.value))):
                        rng = slice(0, len(getattr(self.tree_, tree_attr)))
                        getattr(clf.tree_, tree_attr)[rng] = getattr(self.tree_, tree_attr)[rng]
                    else:
                        setattr(clf.tree_, tree_attr, getattr(self.tree_, tree_attr))
            else:
                setattr(clf, attr, getattr(self, attr))

        del_attr = list(set(dir(clf))-set(clf_attr))
        [delattr(clf, x) for x in del_attr]
        self.clf = clf
        text_representation = tree.export_text(clf, feature_names=self.feature_names)
        return text_representation

    @staticmethod
    def from_dict(obj: Any) -> 'Model':
        _n_features_in_ = int(obj.get("n_features_in_"))
        _feature_names = [y for y in obj.get("feature_names")]
        _n_classes_ = int(obj.get("n_classes_"))
        _classes_ = np.array([y for y in obj.get("classes_")])
        _node_count = int(obj.get("node_count"))
        _tree_ = Tree.from_dict(obj.get("tree_"))
        return Model(_n_features_in_, _feature_names, _n_classes_, _classes_, _node_count, _tree_)
