"""Prefix features for early outcome prediction (EarlyEval-style, behavioural features only).

A prefix of length k is the first k steps of a trajectory; its features use only
what the agent has done and seen so far, so they are available at run time.
"""

import re

LANGUAGES = ("cpp", "go", "java", "javascript", "python", "rust")
TEST_CMD = re.compile(r"pytest|go test|cargo test|npm test|npx jest|gradle|mvn|ctest|make\b|cmake|unittest|\btest\b", re.I)
ERROR_TXT = re.compile(r"Traceback|Exception|error:|Error:|FAILED|FAIL\b|panic:|undefined reference|cannot find", re.M)
PASS_TXT = re.compile(r"\bpassed\b|All tests passed|\bok\b|PASS\b|test result: ok|BUILD SUCCESSFUL", re.I)
FAIL_TXT = re.compile(r"\bfailed\b|failures?:|test result: FAILED|BUILD FAILED|AssertionError|Assertion failed", re.I)

FEATURE_NAMES = (
    ["k", "tokens_so_far", "n_view", "n_edit", "n_create", "n_bash", "n_files_edited",
     "steps_since_edit", "n_test_runs", "n_error_results", "n_pass_signals", "n_fail_signals",
     "last_is_edit", "last_is_bash", "last_is_test", "last_result_len", "last_error", "last_pass",
     "last_fail", "edited_after_last_test"]
    + [f"lang_{l}" for l in LANGUAGES]
)


def _editor_command(step):
    m = re.search(r"'command': '(\w+)'", step.input)
    return m.group(1) if m else ""


def _edited_path(step):
    m = re.search(r"'path': '([^']+)'", step.input)
    return m.group(1) if m else ""


def prefix_features(traj, k):
    """Feature vector for the first k steps (k >= 1) of a trajectory."""
    steps = traj.steps[:k]
    n_view = n_edit = n_create = n_bash = n_test = n_err = n_pass = n_fail = 0
    files, last_edit, last_test = set(), -1, -1
    tokens = traj.first_call_tokens[0]
    for i, s in enumerate(steps):
        tokens += s.input_tokens
        if s.tool == "editor":
            cmd = _editor_command(s)
            if cmd == "view":
                n_view += 1
            elif cmd in ("edit", "str_replace", "insert"):
                n_edit += 1; last_edit = i; files.add(_edited_path(s))
            elif cmd == "create":
                n_create += 1; last_edit = i; files.add(_edited_path(s))
        elif s.tool == "bash":
            n_bash += 1
            if TEST_CMD.search(s.input):
                n_test += 1; last_test = i
        n_err += bool(ERROR_TXT.search(s.result))
        n_pass += bool(PASS_TXT.search(s.result))
        n_fail += bool(FAIL_TXT.search(s.result))
    last = steps[-1]
    last_cmd = _editor_command(last) if last.tool == "editor" else ""
    lang = traj.task.split("__")[0]
    return [
        k, tokens, n_view, n_edit, n_create, n_bash, len(files),
        (k - 1 - last_edit) if last_edit >= 0 else k, n_test, n_err, n_pass, n_fail,
        int(last_cmd in ("edit", "str_replace", "insert", "create")), int(last.tool == "bash"),
        int(last.tool == "bash" and bool(TEST_CMD.search(last.input))), len(last.result),
        int(bool(ERROR_TXT.search(last.result))), int(bool(PASS_TXT.search(last.result))),
        int(bool(FAIL_TXT.search(last.result))), int(last_edit > last_test),
    ] + [int(lang == l) for l in LANGUAGES]
