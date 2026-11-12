# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ramble.appkit import *
from ramble.expander import Expander


class NctMdtest(ExecutableApplication):
    """mdtest (from the NCT IOR build): file create/stat/read/remove rates, one
    directory per rank.

    NCAR fork: the binary comes from {nct_app_dir}, not Spack. mdtest removes its
    files; rmdir then fails (and so the job) if any were left behind.
    """

    name = "nct-mdtest"

    maintainers("neumanbrett")

    tags("synthetic", "i-o", "metadata", "mpi", "c")

    required_package("nct-mdtest")

    executable("mkdir", "mkdir -p {io_dir}", use_mpi=False)
    executable("df", "df -hT {io_dir}", use_mpi=False)
    executable(
        "mdtest",
        "{nct_app_dir}/bin/mdtest -F -u -i 1 -n {files_per_rank} -d {io_dir}",
        use_mpi=True,
    )
    executable("rmdir", "rmdir {io_dir}", use_mpi=False)

    workload("files", executables=["mkdir", "df", "mdtest", "rmdir"])

    # Not `name`: a class-level loop variable would replace the app's name.
    for var, default, desc in (
        ("nct_app_dir", "", "NCT install dir of the IOR build (has mdtest)"),
        ("io_dir", "", "Directory the test writes in"),
        ("files_per_rank", "1000", "Files each rank creates (-n)"),
    ):
        workload_variable(var, default=default, description=desc, workloads=["files"])

    log_file = os.path.join(
        Expander.expansion_str("experiment_run_dir"),
        Expander.expansion_str("experiment_name") + ".out",
    )

    # mdtest prints a rate table and then a time table with the same row names.
    figure_of_merit_context(
        "Summary", regex=r"SUMMARY (?P<kind>rate|time)", output_format="Summary: {kind}"
    )
    number = r"[0-9]+\.[0-9]+"
    for op in ("File creation", "File stat", "File read", "File removal"):
        figure_of_merit(
            op,
            log_file=log_file,
            fom_regex=rf"\s*{op}\s+(?P<max>{number})\s+{number}\s+(?P<mean>{number})",
            group_name="mean",
            units="",
            contexts=["Summary"],
        )
    success_criteria(
        "removal_completed",
        mode="string",
        match=r"\s*File removal\s+[0-9]",
        file=log_file,
    )
