"""Tests for the patch-file race fix in repos/HGM/polyglot/harness.py.

Upstream, every evaluation rewrote each patch of the agent's lineage in place with open(path, "w");
concurrent evaluations of one agent could read a just-truncated file and write it back empty.
harness() now filters each patch into a private copy and only reads the shared files.

The harness module imports docker and HGM's helpers, so harness() and remove_patch_by_files() are
executed on their own here, with _harness() replaced by a stub that reads the copies it is given.
Run from the instrumentation directory:  python -m unittest discover -s tests -v
"""

import os
import random
import re
import shutil
import tempfile
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
HGM = os.path.join(HERE, "..", "..", "repos", "HGM")
HARNESS = os.path.join(HGM, "polyglot", "harness.py")
GIT_UTILS = os.path.join(HGM, "utils", "git_utils.py")


def load_function(path, name):
    with open(path) as f:
        src = f.read()
    # up to the next top-level line (a line starting with ")" still closes the signature)
    m = re.search(rf"^def {name}\(.*?(?=^(?![\s)])|\Z)", src, re.S | re.M)
    return m.group(0)


def read(path):
    with open(path) as f:
        return f.read()


PATCH_TOOL = ("diff --git a/tools/code_validator.py b/tools/code_validator.py\n"
              "new file mode 100644\n--- /dev/null\n+++ b/tools/code_validator.py\n@@ -0,0 +1,3 @@\n"
              + "".join(f"+line {i}\n" for i in range(3)))
PATCH_MIXED = ("diff --git a/coding_agent.py b/coding_agent.py\n--- a/coding_agent.py\n+++ b/coding_agent.py\n"
               "@@ -1 +1 @@\n-old\n+new\n"
               "diff --git a/polyglot_helper.py b/polyglot_helper.py\nnew file mode 100644\n"
               "--- /dev/null\n+++ b/polyglot_helper.py\n@@ -0,0 +1 @@\n+removed by the filter\n")


@unittest.skipUnless(os.path.exists(HARNESS), "HGM repo not present")
class TestPatchCopies(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.lock = threading.Lock()

        def stub(model_patch_paths=None, **kwargs):
            seen = []
            for p in model_patch_paths or []:
                time.sleep(random.random() * 0.002)          # let the threads interleave
                seen.append((p, read(p)))
            with self.lock:
                self.calls.append(seen)
            return []

        self.ns = {"os": os, "tempfile": tempfile, "_harness": stub}
        exec(load_function(GIT_UTILS, "remove_patch_by_files"), self.ns)
        exec(load_function(HARNESS, "harness"), self.ns)
        self.dir = tempfile.mkdtemp()
        self.paths = []
        for i, text in enumerate([PATCH_TOOL * 2000, PATCH_MIXED]):      # a large one and a mixed one
            p = os.path.join(self.dir, f"{i}_model_patch.diff")
            with open(p, "w") as f:
                f.write(text)
            self.paths.append(p)
        self.original = [read(p) for p in self.paths]

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def run_concurrently(self, n_threads):
        start = threading.Barrier(n_threads)

        def worker():
            start.wait()
            self.ns["harness"](test_task_list=["t"], model_patch_paths=self.paths)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_shared_files_untouched_under_concurrency(self):
        mtimes = [os.stat(p).st_mtime_ns for p in self.paths]
        for _ in range(20):
            self.run_concurrently(20)
        self.assertEqual([read(p) for p in self.paths], self.original)
        self.assertEqual([os.stat(p).st_mtime_ns for p in self.paths], mtimes)

    def test_every_call_sees_the_filtered_patches_in_order(self):
        self.run_concurrently(20)
        expected = [self.ns["remove_patch_by_files"](t) for t in self.original]
        self.assertEqual(len(self.calls), 20)
        for seen in self.calls:
            self.assertEqual([text for _, text in seen], expected)
        self.assertNotIn("polyglot_helper.py", expected[1])
        self.assertIn("coding_agent.py", expected[1])

    def test_private_copies_are_removed(self):
        self.run_concurrently(5)
        for seen in self.calls:
            for p, _ in seen:
                self.assertFalse(os.path.exists(p))
                self.assertNotIn(self.dir, p)

    def test_no_patches_passes_through(self):
        self.ns["harness"](test_task_list=["t"], model_patch_paths=None)
        self.ns["harness"](test_task_list=["t"], model_patch_paths=[])
        self.assertEqual(self.calls, [[], []])


if __name__ == "__main__":
    unittest.main()
