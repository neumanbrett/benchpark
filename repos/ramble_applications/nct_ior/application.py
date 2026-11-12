# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ramble.appkit import *
from ramble.expander import Expander


class NctIor(ExecutableApplication):
    """IOR built by NCT: POSIX write then read in one job, with O_DIRECT so reads
    come from the filesystem, not the page cache.

    NCAR fork: the binary comes from {nct_app_dir}, not Spack. The test directory is
    created before and removed after; rmdir fails (and so the job) if IOR left
    files behind.
    """

    name = "nct-ior"

    maintainers("neumanbrett")

    tags("synthetic", "i-o", "mpi", "c")

    required_package("nct-ior")

    executable("mkdir", "mkdir -p {io_dir}", use_mpi=False)
    # Which filesystem the test hit, and how full it is.
    executable("df", "df -hT {io_dir}", use_mpi=False)
    common = (
        "{nct_app_dir}/bin/ior -a POSIX --posix.odirect -w -r -e -C "
        "-Q {n_ranks_per_node} -i 1 -t {transfer} -b {block} -o {io_dir}/ior"
    )
    executable("ior-fpp", common + " -F", use_mpi=True)
    executable("ior-shared", common, use_mpi=True)
    executable("rmdir", "rmdir {io_dir}", use_mpi=False)

    workload("fpp", executables=["mkdir", "df", "ior-fpp", "rmdir"])
    workload("shared", executables=["mkdir", "df", "ior-shared", "rmdir"])

    # Not `name`: a class-level loop variable would replace the app's name.
    for var, default, desc in (
        ("nct_app_dir", "", "NCT install dir of the IOR build"),
        ("io_dir", "", "Directory the test writes in"),
        ("transfer", "1m", "Transfer size (-t)"),
        ("block", "1g", "Data per rank (-b)"),
    ):
        workload_variable(
            var, default=default, description=desc, workloads=["fpp", "shared"]
        )

    log_file = os.path.join(
        Expander.expansion_str("experiment_run_dir"),
        Expander.expansion_str("experiment_name") + ".out",
    )

    # "Summary of all tests" rows: op, Max(MiB), Min, Mean, StdDev, Max(OPs), Min,
    # Mean, StdDev, Mean(s), then NA columns (per-iteration rows have no NA).
    # Ramble expands {...} in regexes, so the 8 columns are spelled out, not {8}.
    n = r"[0-9]+\.[0-9]+"
    rest = (r"\s+" + n) * 8 + r"\s+NA"
    for op, fom in (("write", "Write bandwidth"), ("read", "Read bandwidth")):
        figure_of_merit(
            fom,
            log_file=log_file,
            fom_regex=rf"^{op}\s+(?P<bw>{n})" + rest,
            group_name="bw",
            units="MiB/s",
        )
    success_criteria(
        "read_completed", mode="string", match=r"^read\s+" + n + rest, file=log_file
    )
