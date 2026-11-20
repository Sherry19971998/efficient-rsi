# This file is adapted from https://github.com/jennyzzt/dgm.

import argparse
import datetime
import json
import os
import re
import tempfile
from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                as_completed)
from enum import Enum
from pathlib import Path

import docker
from datasets import load_dataset

from polyglot.constants import MAP_REPO_VERSION_TO_SPECS, TEST_COMMANDS
from polyglot.docker_build import (build_container, build_env_images,
                                   cleanup_container)
from polyglot.test_spec import hide_answers, make_test_spec
from prompts.testrepo_prompt import get_test_description
from swe_bench.utils import (copy_from_container, copy_to_container,
                             log_container_output, remove_existing_container,
                             safe_log, setup_logger)
from utils.git_utils import filter_patch_by_files, remove_patch_by_files
import rsi_ledger
import lc_integration
import task_records
import shutil

llm = ""
timeout = 1800  # seconds


def get_eval_script(commands):
    return "\n".join(["#!/bin/bash", "set -uxo pipefail"] + commands) + "\n"


def process_entry(
    entry,
    out_dname,
    model_name_or_path,
    model_patch_paths,
    skip_existing=True,
    init_agent_path=".",
):
    """
    Process a single dataset entry. This function encapsulates the main processing logic
    for each entry to make it suitable for parallel execution.
    """
    if lc_integration.enabled():     # harness-rsi: little-coder agent, coupled evaluation
        return process_entry_lc(entry, out_dname, model_name_or_path, model_patch_paths,
                                skip_existing=skip_existing, init_agent_path=init_agent_path)
    instance_id = entry["instance_id"]
    problem_statement = entry["problem_statement"]
    base_commit = entry["base_commit"]
    chat_history_file = out_dname / (instance_id + ".md")
    out_fname = out_dname / (instance_id + ".json")
    eval_file = out_dname / f"{instance_id}_eval.sh"
    eval_result_file = out_dname / f"{instance_id}_eval.md"

    # Skip if output result already exists
    if out_fname.exists() and skip_existing:
        print(f"Skipping existing entry {instance_id}")
        with open(out_fname) as f:
            result = json.loads(f.read())
        return result

    # Initialize container as None to avoid UnboundLocalError
    container, client, logger = None, None, None
    card = task_records.TaskRecord(out_dname, entry, model_name_or_path, llm, "original",
                                   timeout, model_patch_paths, init_agent_path)
    result = {"instance_id": instance_id, "eval_result": "incomplete", "success": False}

    phases = rsi_ledger.Phases("eval", node=model_name_or_path, task=instance_id, attempt_id=card.id)
    phases.phase("setup")
    try:
        # Create and start the Docker container
        client = docker.from_env()
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Set up thread-specific logger
        logger = setup_logger(str(out_dname / f"{instance_id}_docker.log"))
        nocache = True
        test_spec = make_test_spec(entry)
        # Remove any existing container with the same name
        container_name = test_spec.get_instance_container_name(run_id)
        remove_existing_container(client, container_name)
        # Now create and start the container
        container = build_container(
            test_spec, client, run_id, logger, nocache, force_rebuild=False
        )
        container.start()
        phases.attach(container)

        # Copy the necessary files and requirements to the container
        copy_to_container(
            container,
            os.path.join(init_agent_path, "coding_agent_polyglot.py"),
            "/hgm/coding_agent.py",
        )
        copy_to_container(
            container,
            os.path.join(init_agent_path, "requirements.txt"),
            "/hgm/requirements.txt",
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "pytest.ini"), "/hgm/pytest.ini"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "tools/"), "/hgm/tools/"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "utils/"), "/hgm/utils/"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "tests/"), "/hgm/tests/"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "prompts/"), "/hgm/prompts/"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "llm.py"), "/hgm/llm.py"
        )
        copy_to_container(
            container,
            os.path.join(init_agent_path, "llm_withtools.py"),
            "/hgm/llm_withtools.py",
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "LICENSE"), "/hgm/LICENSE"
        )
        copy_to_container(
            container, os.path.join(init_agent_path, "README.md"), "/hgm/README.md"
        )
        chat_history_file_container = f"/hgm/{chat_history_file.name}"

        # See the checked repo
        exec_result = container.exec_run("ls -R /testbed", workdir="/")
        log_container_output(exec_result)

        # Get test description
        eval_cmd = MAP_REPO_VERSION_TO_SPECS[entry["language"]]["test_cmd"]
        test_description = get_test_description(eval_cmd, polyglot=True)

        # Apply model patch
        if model_patch_paths:
            safe_log("Applying model patches")
            for model_patch_path in model_patch_paths:
                copy_to_container(container, model_patch_path, "/hgm/parent_patch.txt")
                exec_result = container.exec_run(
                    "/bin/sh -c 'patch -p1 < /hgm/parent_patch.txt'", workdir="/hgm"
                )
                log_container_output(exec_result)
                exec_result = container.exec_run(
                    "rm /hgm/parent_patch.txt", workdir="/hgm"
                )
                log_container_output(exec_result)

        # Install this repo requirements
        safe_log("Installing more requirements")
        exec_result = container.exec_run(
            "python -m pip install -r /hgm/requirements.txt", workdir="/"
        )
        log_container_output(exec_result)

        # Run the agent
        env_vars = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "AWS_REGION": os.getenv("AWS_REGION"),
            "AWS_REGION_NAME": os.getenv("AWS_REGION_NAME"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OpenRouter_API_KEY": os.getenv("OpenRouter_API_KEY"),
        }
        safe_log("Running the agent")
        cmd = [
            "timeout",
            str(timeout),
            "python",
            "/hgm/coding_agent.py",
            "--problem_statement",
            problem_statement,
            "--git_dir",
            "/testbed/",
            "--chat_history_file",
            chat_history_file_container,
            "--base_commit",
            base_commit,
            "--outdir",
            "/hgm/",
            "--test_description",
            test_description,
            "--language",
            entry["language"],
            "--model",
            llm,
            "--timeout",
            str(timeout),
        ]
        env_vars.update(rsi_ledger.proxy_env("eval", model_name_or_path, instance_id))
        phases.phase("agent_run")
        card.phase("solve")
        exec_result = container.exec_run(cmd, environment=env_vars, workdir="/testbed/")
        card.agent_exit(exec_result)
        log_container_output(exec_result, raise_error=False)
        card.collect(container, copy_from_container)
        _publish_task_artifacts(card, out_dname)
        model_patch = card.patch_for_grading()

        # Additional proposed model patches
        proposed_model_patches = []

        phases.phase("test")
        # Directly do eval
        eval_result = ""
        if not model_patch:
            eval_result = "empty_patch"
            result = {
                "instance_id": instance_id,
                "model_name_or_path": model_name_or_path,
                "model_patch": model_patch,
                "proposed_model_patches": proposed_model_patches,
                "eval_result": eval_result,
                "success": True,
            }
            out_fname.write_text(json.dumps(result, indent=4))
            return {
                "success": True,
                "instance_id": instance_id,
                "eval_result": eval_result,
            }

        card.phase("grade")
        exec_result = _score_in_container(container, entry, eval_file)
        log_container_output(exec_result, raise_error=False)
        eval_result_file.write_text(exec_result.output.decode(errors="replace"))
        card.grade(exec_result)
        if exec_result.exit_code == 0:
            eval_result = "resolved"
        else:
            eval_result = "unresolved"

        # Write result to file
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            "proposed_model_patches": proposed_model_patches,
            "eval_result": eval_result,
            "success": True,
        }
        out_fname.write_text(json.dumps(result, indent=4))

        return {"success": True, "instance_id": instance_id, "eval_result": eval_result}

    except Exception as e:
        card.error(e)
        # Check if eval_result exists in local scope
        if "eval_result" not in locals():
            eval_result = "incomplete"
        else:
            eval_result = "error"
        if "model_patch" not in locals():
            model_patch = ""
        if "proposed_model_patches" not in locals():
            proposed_model_patches = ""

        # Write result to file
        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            "proposed_model_patches": proposed_model_patches,
            "eval_result": eval_result,
            "success": False,
        }
        out_fname.write_text(json.dumps(result, indent=4))

        # print(f"Error processing entry {instance_id}: {str(e)}")
        if logger is not None:
            logger.error(f"Error processing entry {instance_id}: {str(e)}")
        return {
            "success": False,
            "instance_id": instance_id,
            "eval_result": eval_result,
        }

    finally:
        _finish_task_record(card, container, result, out_dname)
        phases.done()
        # Clean up docker container
        try:
            if container is not None:
                cleanup_container(client, container, logger)
        except Exception as e:
            print(f"Error cleaning up Docker container for {instance_id}: {e}")


def _publish_task_artifacts(card, out_dir, record_prefix=None):
    """Keep legacy readers working; the attempt directory retains the original evidence."""
    for name, legacy in (("trajectory.md", f"{card.task}.md"), ("agent_attempt.json", f"{card.task}_rsi.json")):
        source = card.dir / name
        if source.exists():
            shutil.copy2(source, Path(out_dir) / legacy)
    for source in card.data["evidence"].get("additional_trace_paths", []):
        shutil.copy2(source, Path(out_dir) / Path(source).name)
    # Only completed attempts supply replay caches; incomplete records stay in the card directory.
    if record_prefix is not None and card.data["execution"]["exit_code"] == 0:
        for suffix in (".jsonl", ".req.jsonl.gz", ".replay", ".replay.req.jsonl.gz", ".side.jsonl"):
            source = card.dir / ("record" + suffix)
            if source.exists():
                shutil.copy2(source, Path(str(record_prefix) + suffix))


def _finish_task_record(card, container, result, out_dir):
    try:
        if not card.collected and card.data["execution"].get("stage") == "solve":
            card.collect(container, copy_from_container)
            _publish_task_artifacts(card, out_dir)
        docker_log = Path(out_dir) / f"{card.task}_docker.log"
        if docker_log.exists():
            shutil.copy2(docker_log, card.dir / "runner.log")
            card.data["evidence"]["runner_log_path"] = str(card.dir / "runner.log")
        card.finish(result)
    except Exception as exc:
        # Logging failure must not turn an original timeout into a different execution outcome.
        safe_log(f"Task record could not be finalized ({card.id}): {exc}")


def _score_in_container(container, entry, eval_file):
    """efficient-rsi's grading, unchanged: stop the agent's processes, bring in the test commit,
    keep only the solution files, run the language's test command (polyglot/harness.py)."""
    if hide_answers():
        container.exec_run("sh -c 'kill -9 -1 || true'", workdir="/")
        copy_to_container(container, Path(entry["repo"]) / ".git", "/tmp/answer_repo/.git")
        container.exec_run("git config --global --add safe.directory '*'", workdir="/")
        log_container_output(container.exec_run("git -C /testbed fetch --quiet /tmp/answer_repo HEAD", workdir="/"))
    log_container_output(container.exec_run(
        ["git", "--literal-pathspecs", "-C", "/testbed", "stash", "push", "--include-untracked", "--",
         *entry["files"]["solution"]], workdir="/"))
    log_container_output(container.exec_run(f"git -C /testbed reset --hard {entry['test_commit']}", workdir="/"))
    log_container_output(container.exec_run("git -C /testbed clean -fd", workdir="/"))
    log_container_output(container.exec_run("git -C /testbed stash pop", workdir="/"))
    eval_file.write_text(get_eval_script(TEST_COMMANDS[entry["language"]]))
    copy_to_container(container, eval_file, "/testbed/eval.sh")
    container.exec_run("chmod +x /testbed/eval.sh", workdir="/")
    return container.exec_run("timeout 120 ./eval.sh", workdir="/testbed")


def process_entry_lc(entry, out_dname, model_name_or_path, model_patch_paths,
                     skip_existing=True, init_agent_path="."):
    """harness-rsi (lc_integration.py): one evaluation of the little-coder agent on one task.

    Differences from process_entry:
      - the agent is <init_agent_path>/lc, patched in /hgm/lc, run as /hgm/lc/hgm_agent.py;
      - HGM_TEST_PROTOCOL=visible_ro: the task's tests are copied into /testbed before the agent
        runs, read-only, and hashed before and after; a changed hash scores the attempt unresolved;
      - the agent records every model call; with HGM_COUPLED=1 and a parent record of this task it
        replays the parent's trajectory and calls the model only from the divergence point on;
      - web tools are off (LC_DISABLE_TOOLS); the records go to <node>/records/, the attempt
        summary to <task>_rsi.json, and an lc_eval event (route, live model seconds) to the ledger.
    Grading is efficient-rsi's (_score_in_container)."""
    instance_id = entry["instance_id"]
    chat_history_file = out_dname / (instance_id + ".md")
    out_fname = out_dname / (instance_id + ".json")
    eval_file = out_dname / f"{instance_id}_eval.sh"
    eval_result_file = out_dname / f"{instance_id}_eval.md"
    attempt_file = out_dname / f"{instance_id}_rsi.json"
    if out_fname.exists() and skip_existing:
        return json.loads(out_fname.read_text())

    node = model_name_or_path[:-2] if model_name_or_path.endswith("_0") else model_name_or_path
    node_dir = lc_integration.node_dir_from_outdir(out_dname)
    pdir = lc_integration.parent_dir(node_dir)
    replay = lc_integration.coupled() and pdir is not None and lc_integration.has_record(pdir, instance_id)
    rec_host = lc_integration.record_prefix(node_dir, instance_id)
    rec_host.parent.mkdir(parents=True, exist_ok=True)

    container, client, logger = None, None, None
    eval_result, model_patch, attempt, tests_modified = "incomplete", "", {}, False
    card = task_records.TaskRecord(out_dname, entry, model_name_or_path, llm, "littlecoder",
                                   timeout, model_patch_paths, init_agent_path)
    phases = rsi_ledger.Phases("eval", node=model_name_or_path, task=instance_id, attempt_id=card.id)
    phases.phase("setup")
    try:
        client = docker.from_env()
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logger = setup_logger(str(out_dname / f"{instance_id}_docker.log"))
        test_spec = make_test_spec(entry)
        container_name = test_spec.get_instance_container_name(run_id)
        remove_existing_container(client, container_name)
        container = build_container(test_spec, client, run_id, logger, True, force_rebuild=False)
        container.start()
        phases.attach(container)

        copy_to_container(container, os.path.join(init_agent_path, lc_integration.LC_DIR) + "/",
                          lc_integration.CONTAINER_LC + "/")
        for model_patch_path in model_patch_paths or []:
            copy_to_container(container, model_patch_path, "/hgm/parent_patch.txt")
            log_container_output(container.exec_run(
                "/bin/sh -c 'patch -p1 < /hgm/parent_patch.txt'", workdir=lc_integration.CONTAINER_LC))
            container.exec_run("rm /hgm/parent_patch.txt", workdir="/hgm")
        log_container_output(container.exec_run(
            f"python -m pip install -r {lc_integration.CONTAINER_LC}/requirements.txt", workdir="/"))

        protected, hashes_before = [], {}
        if lc_integration.protocol() == "visible_ro":
            for rel, data in lc_integration.test_file_contents(entry).items():
                container.exec_run(["mkdir", "-p", str(Path(f"/testbed/{rel}").parent)], workdir="/")
                with tempfile.NamedTemporaryFile(delete=False) as tf:
                    tf.write(data)
                copy_to_container(container, Path(tf.name), f"/testbed/{rel}")
                os.unlink(tf.name)
                container.exec_run(["chmod", "444", f"/testbed/{rel}"], workdir="/")
                protected.append(rel)
            hashes_before = lc_integration.sha256_map(container, protected)

        if replay:
            container.exec_run(["mkdir", "-p", "/hgm/parent_record"], workdir="/")
            for suffix in (".jsonl", ".req.jsonl.gz"):
                copy_to_container(container, Path(str(lc_integration.record_prefix(pdir, instance_id)) + suffix),
                                  "/hgm/parent_record/rec" + suffix)

        eval_cmd = MAP_REPO_VERSION_TO_SPECS[entry["language"]]["test_cmd"]
        env_vars = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "HGM_TASK_ID": instance_id,
            "HGM_PROTECTED_FILES": ",".join(protected),
            "LC_DISABLE_TOOLS": "WebSearch,WebFetch",
            "LC_FIXED_DATE": os.getenv("LC_FIXED_DATE", ""),
            "LC_RSI_MODE": "replay" if replay else "record",
            "LC_RECORD_TO": "/hgm/record/rec",
            "LC_REPLAY_FROM": "/hgm/parent_record/rec" if replay else "",
            "LC_ON_DIVERGE": "live",
        }
        env_vars.update(rsi_ledger.proxy_env("eval", model_name_or_path, instance_id))
        cmd = ["timeout", str(timeout), "python", f"{lc_integration.CONTAINER_LC}/hgm_agent.py",
               "--problem_statement", entry["problem_statement"], "--git_dir", "/testbed/",
               "--chat_history_file", f"/hgm/{chat_history_file.name}", "--base_commit", entry["base_commit"],
               "--outdir", "/hgm/", "--test_description", get_test_description(eval_cmd, polyglot=True),
               "--language", entry["language"], "--model", llm, "--timeout", str(timeout)]
        phases.phase("agent_run")
        card.phase("solve")
        agent_result = container.exec_run(cmd, environment=env_vars, workdir="/testbed/")
        card.agent_exit(agent_result)
        log_container_output(agent_result, raise_error=False)
        card.collect(container, copy_from_container)
        _publish_task_artifacts(card, out_dname, rec_host)
        attempt = card.data["evidence"].get("agent_report", {})
        model_patch = card.patch_for_grading()
        if protected:
            tests_modified = lc_integration.sha256_map(container, protected) != hashes_before
            # the visible copies are untracked; grading checks out the real tests from test_commit
            container.exec_run(["rm", "-f"] + [f"/testbed/{rel}" for rel in protected], workdir="/")

        phases.phase("test")
        if not model_patch:
            eval_result = "empty_patch"
        else:
            card.phase("grade")
            r = _score_in_container(container, entry, eval_file)
            card.grade(r, tests_modified)
            eval_result_file.write_text(r.output.decode())
            eval_result = "resolved" if r.exit_code == 0 and not tests_modified else "unresolved"
        success = True
    except Exception as e:
        card.error(e)
        if logger is not None:
            logger.error(f"Error processing entry {instance_id}: {e}")
        eval_result = "incomplete" if not model_patch else "error"
        success = False
    finally:
        _finish_task_record(card, container, {"instance_id": instance_id, "model_name_or_path": model_name_or_path,
                                            "eval_result": eval_result, "success": locals().get("success", False)}, out_dname)
        phases.done()
        try:
            if container is not None:
                cleanup_container(client, container, logger)
        except Exception as e:
            print(f"Error cleaning up Docker container for {instance_id}: {e}")

    route = lc_integration.attempt_route(attempt, replay)
    cost = lc_integration.record_cost(rec_host)
    lc_integration.ledger_event(rsi_ledger, node, instance_id, route, attempt, cost, tests_modified, eval_result)
    result = {"instance_id": instance_id, "model_name_or_path": model_name_or_path,
              "model_patch": model_patch, "proposed_model_patches": [], "eval_result": eval_result,
              "success": success, "tests_modified": tests_modified, "lc_route": route,
              "lc_parent": pdir.name if pdir is not None else None, "lc_cost": cost,
              "task_record": str(card.path.relative_to(out_dname)), "attempt_id": card.id}
    out_fname.write_text(json.dumps(result, indent=4))
    return {"success": success, "instance_id": instance_id, "eval_result": eval_result}


def _harness(
    dataset_path="polyglot/polyglot_benchmark_metadata.json",
    test_task_list=None,
    max_workers=4,
    model_name_or_path=None,
    model_patch_paths=None,
    pred_dname="./polyglot/predictions",
    output_dir="./polyglot/predictions",
    skip_existing=True,
    init_agent_path=".",
):
    """
    Parallel processing harness using ThreadPoolExecutor.

    Args:
        test_task_list: List of task IDs to process (None for all)
        num_samples: Number of samples to process (-1 for all)
        max_workers: Maximum number of concurrent threads
        model_name_or_path: Model name or path
        model_patch_paths: Paths to the model patches for hgm (already filtered by harness())
        num_evals: Repeated number of swe evaluations
    """
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)

    # Ensure that necessary directories exist
    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_or_path = f"{timestamp}--claude-3-5-sonnet-20241022"
    pred_dname = Path(pred_dname)
    pred_dname.mkdir(exist_ok=True)
    out_dnames = []

    # Prepare the dataset entries
    entries = list(dataset)
    entries = [entry for entry in entries if entry["instance_id"] in test_task_list]

    # Build the environment images
    client = docker.from_env()
    build_env_images(
        client, dataset=entries, max_workers=max_workers, force_rebuild=False
    )

    # Define a function to handle a single evaluation for all specified issues
    model_name_or_path_inst = f"{model_name_or_path}_{0}"
    out_dname = pred_dname / model_name_or_path_inst
    out_dname.mkdir(exist_ok=True)

    print(f"Starting evaluation {0} for model {model_name_or_path}")

    # Process entries in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_entry = {
            executor.submit(
                process_entry,
                entry,
                out_dname,
                model_name_or_path_inst,
                model_patch_paths,
                skip_existing=skip_existing,
                init_agent_path=init_agent_path,
            ): entry
            for entry in entries
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_entry):
            result = future.result()
            results.append(future.result())
            if result["success"]:
                print(
                    f"Successfully processed entry {result['instance_id']} for eval {0}"
                )
            else:
                print(
                    f"Failed to process entry {result['instance_id']} for eval {0}: {result.get('error', 'Unknown error')}"
                )
        # Get final results from completed futures

    print(f"All evaluations completed for model {model_name_or_path}")

    # Directly generate report
    # write report to file
    incomplete_ids = [
        result["instance_id"] for result in results if not result["success"]
    ]
    completed_ids = [result["instance_id"] for result in results if result["success"]]
    # Get resolved/unresolved/error/empty patch IDs from results
    resolved_ids = []
    unresolved_ids = []
    error_ids = []
    empty_patch_ids = []
    unstopped_containers = []
    unremoved_images = []

    for result in results:
        if result["success"]:
            if result.get("eval_result") == "resolved":
                resolved_ids.append(result["instance_id"])
            elif result.get("eval_result") == "unresolved":
                unresolved_ids.append(result["instance_id"])
            elif result.get("eval_result") == "empty_patch":
                empty_patch_ids.append(result["instance_id"])
            else:
                error_ids.append(result["instance_id"])

    report = {
        "total_instances": len(dataset),
        "submitted_instances": len(results),
        "completed_instances": len(completed_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "unstopped_instances": len(unstopped_containers),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(result["instance_id"] for result in results)),
        "resolved_ids": list(sorted(resolved_ids)),
        "unresolved_ids": list(sorted(unresolved_ids)),
        "error_ids": list(sorted(error_ids)),
        "unstopped_containers": list(sorted(unstopped_containers)),
        "unremoved_images": list(sorted(unremoved_images)),
        "schema_version": 2,
    }

    print(report)
    report_file = output_dir / Path(
        model_name_or_path.replace("/", "__")
        + " "
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + ".json"
    )
    with open(report_file, "w") as f:
        print(json.dumps(report, indent=4), file=f)
    print(f"Report written to {report_file}")

    return out_dnames


def harness(
    dataset_path="polyglot/polyglot_benchmark_metadata.json",
    test_task_list=None,
    max_workers=4,
    model_name_or_path=None,
    model_patch_paths=None,
    pred_dname="./polyglot/predictions",
    output_dir="./polyglot/predictions",
    skip_existing=True,
    init_agent_path=".",
):
    """Evaluate an agent on test_task_list; see _harness for the arguments.

    efficient-RSI fix: each patch in model_patch_paths is filtered (remove_patch_by_files) into a
    private copy that only this call uses, and the files on disk are only read. Upstream wrote the
    filtered text back over the shared file with open(path, "w"), which first truncates it. HGM
    evaluates the same agent in several threads at once (each call is one task), so a thread could
    read the file between another thread's truncate and write, and write it back empty: the agent
    then silently ran as its parent from that point on. In v1 this emptied 8 first-generation
    patches, among them the final agent of hgm-fixed-r1.
    """
    kwargs = dict(dataset_path=dataset_path, test_task_list=test_task_list, max_workers=max_workers,
                  model_name_or_path=model_name_or_path, pred_dname=pred_dname, output_dir=output_dir,
                  skip_existing=skip_existing, init_agent_path=init_agent_path)
    if not model_patch_paths:
        return _harness(model_patch_paths=model_patch_paths, **kwargs)
    with tempfile.TemporaryDirectory(prefix="hgm_patches_") as tmp:
        private_paths = []
        for i, model_patch_path in enumerate(model_patch_paths):
            with open(model_patch_path, "r") as f:
                patch_content = remove_patch_by_files(f.read())
            private_path = os.path.join(tmp, f"{i:03d}_model_patch.diff")
            with open(private_path, "w") as f:
                f.write(patch_content)
            private_paths.append(private_path)
        return _harness(model_patch_paths=private_paths, **kwargs)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples", type=int, default=-1, help="Number of samples to process"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of concurrent threads",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default=None, help="Model name or path"
    )
    parser.add_argument(
        "--model_patch_paths", type=str, default=None, help="Paths to the model patches"
    )
    parser.add_argument(
        "--num_evals", type=int, default=1, help="Repeated number of swe evaluations"
    )
    parser.add_argument(
        "--num_evals_parallel",
        type=int,
        default=1,
        help="Number of parallel repeated evaluations",
    )
    args = parser.parse_args()

    with open("polyglot/polyglot_benchmark_metadata.json") as f:
        metadata = json.loads(f.read())
        language_task_list = [
            entry["instance_id"]
            for entry in metadata
            if entry["instance_id"].startswith("python")
        ]
        # Create a list of all tasks from metadata
        all_task_list = [entry["instance_id"] for entry in metadata]

    from utils.common_utils import load_json_file

    swe_issues_med = load_json_file("./polyglot/subsets/medium.json")
    model_patch_paths = (
        args.model_patch_paths.split(",")
        if args.model_patch_paths is not None
        else None
    )
    # Run the parallel harness

    harness(
        dataset_path="polyglot/polyglot_benchmark_metadata.json",
        test_task_list=all_task_list,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=model_patch_paths,
        num_evals=args.num_evals,
        num_evals_parallel=args.num_evals_parallel,
    )


if __name__ == "__main__":
    main()
