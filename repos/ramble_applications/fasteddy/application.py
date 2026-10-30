# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ramble.appkit import *
from ramble.expander import Expander


class Fasteddy(ExecutableApplication):
    """FastEddy: NCAR's GPU-resident large-eddy simulation model (CUDA + MPI).

    NCAR fork: built by NCT (nct/build.py), not Spack. Each run starts from an
    upstream tutorial input in {nct_app_dir}/examples, shortened to {fe_steps}
    timesteps with the domain decomposed over one MPI rank per GPU.
    """

    name = "fasteddy"

    maintainers("neumanbrett")

    tags("atmosphere", "les", "gpu", "cuda", "mpi")

    required_package("fasteddy")

    executable(
        "prepare",
        [
            "cp {nct_app_dir}/examples/{fe_case}.in fasteddy.in",
            "sed -i -E "
            "-e 's|^numProcsX = [0-9]+|numProcsX = {n_ranks}|' "
            "-e 's|^numProcsY = [0-9]+|numProcsY = 1|' "
            "-e 's|^Nt = [0-9]+|Nt = {fe_steps}|' "
            "-e 's|^NtBatch = [0-9]+|NtBatch = {fe_steps}|' "
            "-e 's|^frqOutput = [0-9]+|frqOutput = {fe_steps}|' "
            "-e 's|^outPath = [^#]*|outPath = ./output/ |' "
            "fasteddy.in",
            "mkdir -p output",
        ],
        use_mpi=False,
    )
    executable("run", "{nct_app_dir}/bin/FastEddy fasteddy.in", use_mpi=True)

    workload("fasteddy", executables=["prepare", "run"])

    workload_variable(
        "nct_app_dir",
        default="",
        description="NCT install dir of the FastEddy build",
        workloads=["fasteddy"],
    )
    workload_variable(
        "fe_case",
        default="Example02_CBL",
        description="Upstream tutorial input (tutorials/examples/<case>.in)",
        workloads=["fasteddy"],
    )
    workload_variable(
        "fe_steps",
        default="200",
        description="Timesteps to run (one batch; state is dumped at the end)",
        workloads=["fasteddy"],
    )

    log_file = os.path.join(
        Expander.expansion_str("experiment_run_dir"),
        Expander.expansion_str("experiment_name") + ".out",
    )
    number = r"[0-9]*\.?[0-9]+"

    # TIMESTEP PERFORMANCE row: total | batch steps | time/step | comp./step | IO.
    # The last table (0 batch steps) is the final dump, so match only full batches.
    figure_of_merit(
        "Time per step",
        log_file=log_file,
        fom_regex=(
            rf"^\s*(?P<total>{number})\s+\|\s+(?P<steps>[1-9][0-9]*)\s+\|"
            rf"\s+(?P<per_step>{number})\s+\|\s+(?P<comp>{number})\s+\|"
            rf"\s+(?P<io>{number})"
        ),
        group_name="per_step",
        units="s",
    )
    figure_of_merit(
        "Compute time per step",
        log_file=log_file,
        fom_regex=(
            rf"^\s*(?P<total>{number})\s+\|\s+(?P<steps>[1-9][0-9]*)\s+\|"
            rf"\s+(?P<per_step>{number})\s+\|\s+(?P<comp>{number})\s+\|"
            rf"\s+(?P<io>{number})"
        ),
        group_name="comp",
        units="s",
    )

    success_criteria(
        "complete",
        mode="string",
        match=r"Your FastEddy simulation is complete!",
        file=log_file,
    )
