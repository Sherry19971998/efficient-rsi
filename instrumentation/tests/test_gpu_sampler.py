import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gpu_sampler  # noqa: E402


def fake_proc(root, tree):
    """tree: pid -> (parent pid, process name)"""
    for pid, (ppid, name) in tree.items():
        os.makedirs(os.path.join(root, str(pid)))
        with open(os.path.join(root, str(pid), "stat"), "w") as f:
            f.write(f"{pid} ({name}) S {ppid} 1 1 0 -1\n")
    os.makedirs(os.path.join(root, "self"))   # non-numeric entries are skipped


class TestWithDescendants(unittest.TestCase):
    def test_children_of_the_vllm_server_are_expected(self):
        with tempfile.TemporaryDirectory() as root:
            fake_proc(root, {1: (0, "init"), 100: (1, "python"), 101: (100, "VLLM::EngineCore"),
                             102: (101, "worker (x) y"), 200: (1, "python")})
            self.assertEqual(gpu_sampler.with_descendants([100], proc=root), {100, 101, 102})

    def test_without_proc_only_the_pids(self):
        self.assertEqual(gpu_sampler.with_descendants([5, 6], proc="/nonexistent"), {5, 6})


if __name__ == "__main__":
    unittest.main()
