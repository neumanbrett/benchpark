# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ramble.appkit import *
from ramble.expander import Expander


class Miniweather(ExecutableApplication):
    """miniWeather: a 2D compressible atmospheric dynamics mini-app.

    NCAR fork: binaries are built by NCT (nct/build.py), not Spack, and run from
    {nct_app_dir}. Problem size is compiled in: each executable has a *_test
    twin (100x50 grid, 400 s) used for the upstream correctness check.
    """

    name = "miniweather"

    maintainers("neumanbrett")

    tags("cfd", "atmosphere", "mini-app", "mpi", "openmp", "openacc", "cuda")

    required_package("miniweather")

    executable("run", "{nct_app_dir}/bin/{mw_exe}", use_mpi=True)

    workload("miniweather", executables=["run"])

    workload_variable(
        "nct_app_dir",
        default="",
        description="NCT install dir of the miniWeather build",
        workloads=["miniweather"],
    )
    workload_variable(
        "mw_exe",
        default="mpi_test",
        description="Executable: <model> or <model>_test",
        workloads=["miniweather"],
    )
    # Upstream check_output.sh tolerances: |d_mass| < mass_tol, -te_tol < d_te < 0.
    workload_variable(
        "mass_tol",
        default="1e-13",
        description="Relative mass change tolerance (Fortran 1e-13, C++ 1e-9)",
        workloads=["miniweather"],
    )
    workload_variable(
        "te_tol",
        default="4.5e-5",
        description="Relative total energy change tolerance",
        workloads=["miniweather"],
    )

    log_file = os.path.join(
        Expander.expansion_str("experiment_run_dir"),
        Expander.expansion_str("experiment_name") + ".out",
    )
    # Ramble matches FOM regexes from the start of each line; Fortran list-directed
    # output begins with a blank, hence the leading \s* in each pattern.
    number = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

    figure_of_merit(
        "Run time",
        log_file=log_file,
        fom_regex=rf"\s*CPU Time:\s+(?P<time>{number})",
        group_name="time",
        units="s",
    )
    figure_of_merit(
        "Relative mass change",
        log_file=log_file,
        fom_regex=rf"\s*d_mass:\s+(?P<d_mass>{number})",
        group_name="d_mass",
        units="",
    )
    figure_of_merit(
        "Relative total energy change",
        log_file=log_file,
        fom_regex=rf"\s*d_te:\s+(?P<d_te>{number})",
        group_name="d_te",
        units="",
    )

    # A NaN or missing value leaves the FOM undefined, which fails the check.
    success_criteria(
        "mass_conserved",
        mode="fom_comparison",
        fom_name="Relative mass change",
        formula="{value} < {mass_tol} and {value} > -{mass_tol}",
    )
    success_criteria(
        "energy_dissipated",
        mode="fom_comparison",
        fom_name="Relative total energy change",
        formula="{value} < 0 and {value} > -{te_tol}",
    )
