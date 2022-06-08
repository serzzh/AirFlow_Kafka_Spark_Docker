import json
from model import Rule
from sklearn import tree

jsonstring = json.loads('''
{
    "n_features_in_": 3,
    "feature_names": ["Cu50", "pulp_level", "bulb_torn"],
    "n_classes_": 11,
    "classes_": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "node_count": 21,
    "tree_": {
        "node_count": 21,
        "children_left":  [1, 3, 5, 7, 13, 16, 19, 10, 14, 17, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "children_right": [2, 4, 6, 12, 8, 9, 20, 11, 15, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        "feature": [0, 0, 0, 0, 1, 1, 1, 0, 2, 2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        "threshold": [20.5, 19.5, 21.5, 19, 72, 72, 72, 18.5, 0.5, 0.5, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        "leaf_class_": [-2, -2, -2, -2, -2, -2, -2, -2, -2, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
}
''')

rule = Rule.from_dict(jsonstring)
rule.build_tree()

text_representation = tree.export_text(rule.clf, feature_names=rule.feature_names)
print(text_representation)

print('test 20.43, 71, 0', rule.clf.predict([[20.43, 71, 0]]))
print('test 22.43, 74, 0', rule.clf.predict([[22.43, 74, 0]]))
print('test 21.43, 74, 0', rule.clf.predict([[21.43, 74, 0]]))
print('test 21.43, 74, 1', rule.clf.predict([[21.43, 74, 1]]))


#python -c "from sklearn import tree; help(tree._tree.Tree)"
#json_data = json.dumps(clf.tree_.children_left)
#print(json_data)
