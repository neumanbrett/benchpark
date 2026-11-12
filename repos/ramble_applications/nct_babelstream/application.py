# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ramble.appkit import *
from ramble.expander import Expander


class NctBabelstream(ExecutableApplication):
    """BabelStream (CUDA or HIP) built by NCT, run on each GPU of the job in turn.

    NCAR fork: the binary comes from {nct_app_dir}, not Spack. Results are per GPU
    (FOM context "Device"), so one slow or failing GPU stands out. The run fails if
    any GPU's run fails or BabelStream reports a validation failure.
    """

    name = "nct-babelstream"

    maintainers("neumanbrett")

    tags("memorybenchmark", "microbenchmark", "cuda", "rocm", "c++")

    required_package("nct-babelstream")

    # A subshell, so Ramble's output redirect covers the whole loop and $? is the
    # loop's result (Ramble expands {...}, so no shell braces here).
    executable(
        "run",
        "( rc=0; for d in $(seq 0 $(( {bs_devices} - 1 ))); do "
        'echo "Device index: $d"; '
        "{nct_app_dir}/bin/babelstream --device $d -s {s} -n {n} || rc=1; "
        "done; exit $rc )",
        use_mpi=False,
    )

    workload("babelstream", executables=["run"])

    workload_variable(
        "nct_app_dir",
        default="",
        description="NCT install dir of the BabelStream build",
        workloads=["babelstream"],
    )
    workload_variable(
        "bs_devices", default="1", description="GPUs to test", workloads=["babelstream"]
    )
    workload_variable(
        "n", default="100", description="Iterations", workloads=["babelstream"]
    )
    workload_variable(
        "s",
        default="268435456",
        description="Elements per array (3 arrays of doubles, 6.4 GB in all)",
        workloads=["babelstream"],
    )

    log_file = os.path.join(
        Expander.expansion_str("experiment_run_dir"),
        Expander.expansion_str("experiment_name") + ".out",
    )

    figure_of_merit_context(
        "Device",
        regex=r"Device index:\s+(?P<device>[0-9]+)",
        output_format="Device: {device}",
    )
    number = r"[0-9]+\.?[0-9]*(?:[eE][-+]?[0-9]+)?"
    for op in ("Copy", "Mul", "Add", "Triad", "Dot"):
        figure_of_merit(
            f"{op} bandwidth",
            log_file=log_file,
            fom_regex=rf"^{op}\s+(?P<rate>{number})\s+{number}\s+{number}\s+{number}",
            group_name="rate",
            units="MB/s",
            contexts=["Device"],
        )

    # Ramble checks string criteria line by line. A GPU that can't run (e.g. a missing
    # device index) exits non-zero and fails the loop (NCT's job_completed);
    # BabelStream's own validation only prints, so it gets an anti-match.
    success_criteria("triad_ran", mode="string", match=r"^Triad\s+[0-9]", file=log_file)
    success_criteria(
        "validated", mode="string", anti_match=r".*Validation failed", file=log_file
    )
