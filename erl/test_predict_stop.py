"""Tests for the prediction-driven stopping replay.

Run from this directory:  python -m unittest test_predict_stop -v
"""

import unittest

import predict_stop as ps


def state(results, costs=None, **static):
    st = ps.NodeState({"parent_rate": 0.2, "parent_n": 10, "depth": 1, **static})
    st.results = list(results)
    st.costs = list(costs or [100.0] * len(results))
    return st


class TestFeatures(unittest.TestCase):
    def test_prefix_uses_only_the_first_k_results(self):
        st = state([1, 0, 0, 0], [100.0, 3600.0, 100.0, 100.0], main_removed=7)
        f = dict(zip(ps.FEATURES, st.features(2)))
        self.assertEqual((f["k"], f["solved"], f["rate"]), (2, 1, 0.5))
        self.assertEqual((f["mean_eval_s"], f["long_share"]), (1850.0, 0.5))
        self.assertEqual(f["main_removed"], 7)
        self.assertEqual(len(st.features()), len(ps.FEATURES))

    def test_rows_skip_short_records_and_weight_each_node_once(self):
        X, y, w = ps.training_rows([state([1, 0, 0, 0]), state([1])], min_label=2)
        self.assertEqual(len(X), 5)                  # prefixes k = 0 .. 4 of the first node only
        self.assertEqual(set(y), {0.25})
        self.assertAlmostEqual(sum(w), 1.0)

    def test_stopped_results_left_out_of_the_target(self):
        st = state([1, 0, 0, 1, 0, 0])
        st.censored = [False, True, True, False, False, False]
        X, y, _ = ps.training_rows([st], min_label=2)
        self.assertEqual(set(y), {0.5})                 # 2 solved of the 4 real results
        self.assertEqual(len(X), 7)                     # prefixes still cover all 6 results

    def test_paired_features_compare_with_the_parent(self):
        parent = state([1, 0, 1])
        parent.tasks = ["a", "b", "c"]
        child = ps.NodeState({"parent_rate": 0.5, "parent_n": 3, "depth": 2}, parent=parent, paired=True)
        child.results, child.tasks, child.costs = [0, 1, 1, 0], ["a", "b", "c", "z"], [1.0] * 4
        f = dict(zip(ps.FEATURES + ps.PAIRED_KEYS, child.features()))
        self.assertEqual((f["reg_n"], f["reg_lost"], f["reg_lost_rate"]), (2, 1, 0.5))   # lost a, kept c
        self.assertEqual((f["gain_n"], f["gain_won"]), (1, 1))                           # won b; z unknown
        self.assertEqual(child.features(1)[-5:], [1, 1, 1.0, 0, 0])
        root = ps.NodeState({})
        root.task_map = {"a": 0}
        child.parent = root
        self.assertEqual(child.features()[-5:], [0, 0, 0.0, 1, 0])


class TestAutoWeight(unittest.TestCase):
    def test_uninformative_predictor_earns_little_weight(self):
        states = {f"n{i}": state([1] * r + [0] * (20 - r)) for i, r in enumerate([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])}
        flat = {n: (0.45, 0) for n in states}                 # predicts the mean only
        exact = {n: (sum(st.results) / 20, 0) for n, st in states.items()}
        self.assertLess(ps.auto_weight(flat, states, 5), 3.0)       # about 2.3 tasks
        self.assertEqual(ps.auto_weight(exact, states, 5), ps.W_MAX)

    def test_needs_enough_labelled_nodes(self):
        states = {"a": state([1, 0, 0, 0, 0])}
        self.assertEqual(ps.auto_weight({"a": (0.2, 0)}, states, 5), 0.0)

    def test_only_results_after_the_prediction_count(self):
        # every node was predicted after 18 of its 20 results: 2 later results < min_label 5
        states = {f"n{i}": state([1] * 4 + [0] * 16) for i in range(12)}
        self.assertEqual(ps.auto_weight({n: (0.2, 18) for n in states}, states, 5), 0.0)


if __name__ == "__main__":
    unittest.main()
