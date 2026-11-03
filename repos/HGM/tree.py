import os
import pickle
from statistics import stdev

import numpy as np

import hgm_utils
from hgm_utils import eval_agent, sample_child

FIX_DESCENDANT_EVALS = os.getenv("HGM_FIX_DESCENDANT_EVALS", "1") == "1"


def get_num_total_evals():
    return hgm_utils.nodes[0].get_sum(lambda node: node.num_evals)


class Node:
    def __init__(
        self,
        commit_id,
        utility_measures=None,
        parent_id=None,
        id=None,
    ):
        self.commit_id = commit_id
        self.children = []
        if utility_measures:
            self.utility_measures = utility_measures
        else:
            self.utility_measures = []
        self.parent_id = parent_id
        if id is None:  #
            self.id = len(hgm_utils.nodes)
        else:
            self.id = id
        hgm_utils.nodes[self.id] = self

    def get_sub_tree(self, fn=lambda self: self):
        if len(self.children) == 0:
            return [fn(self)]
        else:
            nodes_list = [fn(self)]
            for child in self.children:
                nodes_list.extend(child.get_sub_tree(fn))
            return nodes_list

    def get_pseudo_decendant_evals(self, num_pseudo):
        if self.num_evals < num_pseudo:
            # Return a copy: get_decendant_evals extends the result in place, and
            # extending self.utility_measures itself would mix descendants' evals
            # into this node's own record. HGM_FIX_DESCENDANT_EVALS=0 restores
            # the upstream behaviour for comparison runs.
            if FIX_DESCENDANT_EVALS:
                return list(self.utility_measures)
            return self.utility_measures
        return [self.mean_utility] * num_pseudo
        

    def get_decendant_evals(self, num_pseudo=10):
        decendant_evals = self.get_pseudo_decendant_evals(num_pseudo)
        for decendant in self.get_sub_tree()[1:]:
            decendant_evals += decendant.utility_measures

        return decendant_evals

    @property
    def num_evals(self):
        return len(self.utility_measures)

    @property
    def mean_utility(self):
        if self.num_evals == 0:
            return np.inf
        return np.sum(self.utility_measures) / self.num_evals

    def add_child(self, child):
        self.children.append(child)

    def save_as_dict(self):
        return {
            "commit_id": self.commit_id,
            "id": self.id,
            "parent_id": self.parent_id,
            "mean_utility": self.mean_utility,
            "num_evals": self.num_evals,
        }
