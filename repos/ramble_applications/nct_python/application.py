# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ramble.appkit import *
from ramble.expander import Expander


class NctPython(ExecutableApplication):
    """Python environment tests: package health checks, OSU's Python MPI benchmarks
    (mpi4py, NumPy or CuPy buffers) and CuPy GPU checks.

    NCAR fork: scripts come from the NCT pytests install ({nct_app_dir}); the Python
    env ({py}) and MPI launcher ({py_launch}) are chosen by the experiment, so the
    same tests run on a deployed env (npl-2025b) or an NCT-built one.
    """

    name = "nct-python"

    maintainers("neumanbrett")

    tags("python", "mpi4py", "cupy", "mpi")

    required_package("nct-python")

    osu = (
        "{py_setup} {py_launch} {gpu_bind} {py} {nct_app_dir}/osu-python/run.py "
        "--benchmark {osu_bench} --buffer {buffer} {osu_args}"
    )
    executable(
        "env-health",
        "{py_setup} {py} {nct_app_dir}/checks/env_health.py "
        "--workdir {experiment_run_dir} {health_args}",
        use_mpi=False,
    )
    executable("osu", osu, use_mpi=False)
    executable(
        "cupy-check",
        "{py} {nct_app_dir}/checks/cupy_check.py --expect {n_gpus}",
        use_mpi=False,
    )

    workload("env-health", executables=["env-health"])
    workload("osu-latency", executables=["osu"])
    workload("osu-bw", executables=["osu"])
    workload("osu-allreduce", executables=["osu"])
    workload("cupy-check", executables=["cupy-check"])

    all_workloads = [
        "env-health",
        "osu-latency",
        "osu-bw",
        "osu-allreduce",
        "cupy-check",
    ]
    # Not `name`: a class-level loop variable would replace the app's name.
    for var, default, desc in (
        ("nct_app_dir", "", "NCT pytests install dir"),
        ("py_setup", "", "Env activation prefixed to the test's command line"),
        ("health_args", "", "Extra env_health.py arguments"),
        ("py", "python", "Python interpreter of the env under test"),
        ("py_launch", "mpiexec", "MPI launcher for the env under test"),
        ("gpu_bind", "", "Per-rank GPU binding wrapper (GPU runs only)"),
        ("osu_bench", "latency", "OSU Python benchmark name"),
        ("buffer", "numpy", "OSU buffer type: numpy or cupy"),
        ("osu_args", "", "Extra OSU arguments"),
    ):
        workload_variable(
            var, default=default, description=desc, workloads=all_workloads
        )

    log_file = os.path.join(
        Expander.expansion_str("experiment_run_dir"),
        Expander.expansion_str("experiment_name") + ".out",
    )

    # Each workload prints its own pass line (OSU: its header; its exit code and
    # results are checked by NCT's job_completed and the FOMs).
    success_criteria(
        "completed",
        mode="string",
        match=r"^(# OMB(-Py| Python) MPI|All checks passed|CuPy checks passed)",
        file=log_file,
    )

    number = r"[0-9]+\.[0-9]+"
    # OSU rows: "<size> <value>" (latency in us, bandwidth in MB/s).
    figure_of_merit_context(
        "msg_size",
        regex=rf"^(?P<msg_size>[0-9]+)\s+{number}\s*$",
        output_format="Message Size: {msg_size}",
    )
    figure_of_merit(
        "OSU result",
        log_file=log_file,
        fom_regex=rf"^(?P<msg_size>[0-9]+)\s+(?P<value>{number})\s*$",
        group_name="value",
        units="us or MB/s",
        contexts=["msg_size"],
    )
    # env_health: per-package import time.
    figure_of_merit_context(
        "package",
        regex=r"^Check (?P<package>\S+): PASSED",
        output_format="Package: {package}",
    )
    figure_of_merit(
        "Import time",
        log_file=log_file,
        fom_regex=rf"^Check (?P<package>\S+): PASSED \(import (?P<t>{number}) s",
        group_name="t",
        units="s",
        contexts=["package"],
    )
    # cupy_check: per-GPU results.
    figure_of_merit_context(
        "Device",
        regex=r"^Device index:\s+(?P<device>[0-9]+)",
        output_format="Device: {device}",
    )
    figure_of_merit(
        "CuPy triad bandwidth",
        log_file=log_file,
        fom_regex=rf"^Triad bandwidth:\s+(?P<bw>{number}) GB/s",
        group_name="bw",
        units="GB/s",
        contexts=["Device"],
    )
    figure_of_merit(
        "CuPy SGEMM rate",
        log_file=log_file,
        fom_regex=rf"^SGEMM rate:\s+(?P<rate>{number}) GFLOP/s",
        group_name="rate",
        units="GFLOP/s",
        contexts=["Device"],
    )
