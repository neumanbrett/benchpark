# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ramble.appkit import *
from ramble.expander import Expander


class NctStream(ExecutableApplication):
    """STREAM (C, OpenMP) built by NCT; array size and iterations set at run time.

    NCAR fork: the binary comes from {nct_app_dir}, not Spack. One process uses
    all its OpenMP threads, so it runs without mpiexec.
    """

    name = "nct-stream"

    maintainers("neumanbrett")

    tags("memorybenchmark", "microbenchmark", "openmp", "c")

    required_package("nct-stream")

    executable("run", "{nct_app_dir}/bin/stream -n {n} -s {s} -o 0", use_mpi=False)

    workload("stream", executables=["run"])

    workload_variable(
        "nct_app_dir",
        default="",
        description="NCT install dir of the STREAM build",
        workloads=["stream"],
    )
    workload_variable("n", default="20", description="Iterations", workloads=["stream"])
    workload_variable(
        "s", default="400000000", description="Elements per array", workloads=["stream"]
    )

    log_file = os.path.join(
        Expander.expansion_str("experiment_run_dir"),
        Expander.expansion_str("experiment_name") + ".out",
    )

    success_criteria(
        "valid",
        mode="string",
        match=r"Solution Validates: avg error less than 1.000000e-13 on all three arrays",
        file=log_file,
    )

    number = r"[0-9]+\.[0-9]*"
    for op in ("Copy", "Scale", "Add", "Triad"):
        figure_of_merit(
            f"{op} best rate",
            log_file=log_file,
            fom_regex=rf"^{op}:\s+(?P<rate>{number})\s+{number}\s+{number}\s+{number}",
            group_name="rate",
            units="MB/s",
        )
    figure_of_merit(
        "Threads",
        log_file=log_file,
        fom_regex=r"Number of Threads counted\s+=\s+(?P<threads>[0-9]+)",
        group_name="threads",
        units="",
    )
