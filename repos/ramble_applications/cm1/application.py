# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ramble.appkit import *
from ramble.expander import Expander


class Cm1(ExecutableApplication):
    """CM1 (Cloud Model 1), NCAR's non-hydrostatic cloud-resolving model.

    NCAR fork: built by NCT (nct/build.py), not Spack. {nct_app_dir}/run holds
    cm1.exe, its lookup tables and the upstream test-case configs. Each run
    starts from an upstream config_files/<case>/namelist.input, shortened by
    overriding the run length and output intervals.
    """

    name = "cm1"

    maintainers("neumanbrett")

    tags("atmosphere", "cloud-model", "les", "mpi", "fortran")

    required_package("cm1")

    executable(
        "prepare",
        [
            "cp -r {nct_app_dir}/run/RRTMG_LW_DATA {nct_app_dir}/run/RRTMG_SW_DATA "
            "{nct_app_dir}/run/LANDUSE.TBL .",
            "cp {nct_app_dir}/run/config_files/{cm1_case}/namelist.input .",
            "sed -i -E "
            "-e 's/^( *timax *=).*/\\1 {timax},/' "
            "-e 's/^( *tapfrq *=).*/\\1 {timax},/' "
            "-e 's/^( *timestats *=).*/\\1 1,/' "
            "namelist.input",
        ],
        use_mpi=False,
    )
    executable("run", "{nct_app_dir}/run/cm1.exe", use_mpi=True)

    workload("cm1", executables=["prepare", "run"])

    workload_variable(
        "nct_app_dir",
        default="",
        description="NCT install dir of the CM1 build",
        workloads=["cm1"],
    )
    workload_variable(
        "cm1_case",
        default="supercell",
        description="Upstream test case in run/config_files",
        workloads=["cm1"],
    )
    workload_variable(
        "timax",
        default="600.0",
        description="Simulated time (s); output is written once, at the end",
        workloads=["cm1"],
    )

    log_file = os.path.join(
        Expander.expansion_str("experiment_run_dir"),
        Expander.expansion_str("experiment_name") + ".out",
    )

    # timestats=1 makes CM1 print, at each output, the total of its timed sections
    # on rank 0 ('Total time:'): solver time, not job wall time. The last one is
    # kept. Fortran list-directed output starts with a blank; Ramble matches from
    # the start of the line.
    figure_of_merit(
        "Solve time",
        log_file=log_file,
        fom_regex=r"\s*Total time:\s+(?P<time>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        group_name="time",
        units="s",
    )

    success_criteria(
        "terminated_normally",
        mode="string",
        # re.match from line start; gfortran list-directed output has a leading blank.
        match=r"\s*Program terminated normally",
        file=log_file,
    )
