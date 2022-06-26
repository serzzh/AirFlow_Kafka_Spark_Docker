from pydantic import BaseModel, conlist
from typing import List, Literal, Union, Any
import math

class RuleChainResponse(BaseModel):
    model_id: int
    tree: Any
    error_message: str

class Operator(BaseModel):
    operator_id: int
    operator_type: Literal['condition', 'delay', 'recommendation']
    parent_id: Union[int, None] #только если есть родитель, иначе null (пусто)
    feature_id: Union[int, None] #только для типа condition, иначе null (пусто)
    threshold_left: Union[float, None] #только для типа condition, если есть порог слева, иначе null (пусто)
    threshold_right: Union[float, None] #только для типа condition, если есть порог справа, threshold_right>threshold_left, иначе null (пусто)
    children_id: Union[List[int], None] #только если есть потомки, иначе null (пусто)
    leaf_class_: Union[int, None] #только для типа recommendation, иначе null (пусто)
    delay_sec: Union[float, None] #только для типа delay, иначе null (пусто)

class RuleChain(BaseModel):
    model_id: int
    n_features_in_: int
    feature_names: List[str]
    n_classes_: int
    classes_: List[int]
    node_count: int
    tree_: List[Operator]
    current_node: Union[int, None] # внутренний параметр - на вход не подается

    def get_children(self, operator_ids, features):
        children = []
        for operator in operator_ids:
            op = self.tree_[operator]
            if op.threshold_left is None:
                op.threshold_left = -math.inf
            if op.threshold_right is None:
                op.threshold_right = math.inf
            if op.operator_type == 'condition':
                if features[op.feature_id] > op.threshold_left and features[op.feature_id] < op.threshold_right:
                    children = children + op.children_id
                else:
                    #print(children)
                    pass
            else:
                children = op.children_id
        return children

    def test_predict(self, features: List[float]):
        self.current_node = [0, ]
        result = []
        while self.get_children(self.current_node, features[0]) is not None and len(self.get_children(self.current_node, features[0])) > 0:
            self.current_node = self.get_children(self.current_node, features[0])
        for node in self.current_node:
            if self.tree_[node].operator_type == 'recommendation':
                result.append(node)
        if len(result)>0:
            prediction = result[0]
        else:
            prediction = 0
        return prediction

    def to_lambda(self):
        return self

if __name__ == "__main__":
    op0 = Operator(
        operator_id=0,
        operator_type='condition',
        parent_id=None,
        feature_id=0,
        threshold_left=19.5,
        threshold_right=20.5,
        children_id=[1,],
        leaf_class_=None,
        delay_sec=None,
    )

    op1 = Operator(
        operator_id=1,
        operator_type='delay',
        parent_id=0,
        feature_id=None,
        threshold_left=None,
        threshold_right=None,
        children_id=[2, 3,],
        leaf_class_=None,
        delay_sec=3600
    )

    op2 = Operator(
        operator_id=2,
        operator_type='condition',
        parent_id=1,
        feature_id=1,
        threshold_left=None,
        threshold_right=72,
        children_id=[4,],
        leaf_class_=None,
        delay_sec=None,
    )

    op3 = Operator(
        operator_id=3,
        operator_type='condition',
        parent_id=1,
        feature_id=1,
        threshold_left=72,
        threshold_right=None,
        children_id=[5, 6,],
        leaf_class_=None,
        delay_sec=None,
    )

    op4 = Operator(
        operator_id=4,
        operator_type='recommendation',
        parent_id=2,
        feature_id=None,
        threshold_left=None,
        threshold_right=None,
        children_id=None,
        leaf_class_=3,
        delay_sec=None,
    )

    op5 = Operator(
        operator_id=5,
        operator_type='condition',
        parent_id=3,
        feature_id=2,
        threshold_left=None,
        threshold_right=0.5,
        children_id=[7,],
        leaf_class_=None,
        delay_sec=None,
    )

    op6 = Operator(
        operator_id=6,
        operator_type='condition',
        parent_id=3,
        feature_id=2,
        threshold_left=0.5,
        threshold_right=None,
        children_id=[8,],
        leaf_class_=None,
        delay_sec=None,
    )

    op7 = Operator(
        operator_id=7,
        operator_type='recommendation',
        parent_id=5,
        feature_id=None,
        threshold_left=None,
        threshold_right=None,
        children_id=None,
        leaf_class_=41,
        delay_sec=None,
    )

    op8 = Operator(
        operator_id=8,
        operator_type='recommendation',
        parent_id=6,
        feature_id=None,
        threshold_left=None,
        threshold_right=None,
        children_id=None,
        leaf_class_=21,
        delay_sec=None,
    )

    rule = RuleChain(
        model_id=5,
        n_features_in_=3,
        feature_names=["fm50_Cu", "fm50_pulp_level", "fm50_bubble"],
        n_classes_=3,
        classes_=[3, 41, 21],
        node_count = 9,
        tree_=[op0, op1, op2, op3, op4, op5, op6, op7, op8,]
    )

    print('predict[19.7, 71, 0]: ', rule.test_predict([19.7, 71, 0]))
    print('predict[0.7, 71, 1]: ', rule.test_predict([0.7, 71, 1]))
    print('predict[19.7, 73, 0]: ', rule.test_predict([19.7, 73, 0]))
    print('predict[19.7, 73, 1]: ', rule.test_predict([19.7, 73, 1]))
    print('predict[19.7, 76, 0]: ', rule.test_predict([19.7, 76, 0]))

    with open('db/rule_new.json', 'w') as outfile:
        outfile.write(rule.json(indent=4))

