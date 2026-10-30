# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from benchpark.directives import maintainers, variant
from benchpark.experiment import Experiment
from benchpark.nct_app import NctBuiltApp
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType


class Cm1(
    NctBuiltApp,
    Experiment,
    ProgrammingModel(ProgrammingModelType.Mpionly),
):
    nct_version = "21.1"
    nct_builds = {
        "mpi-gcc": ("gcc", [], False),
    }

    variant(
        "workload",
        default="cm1",
        values=("cm1",),
        description="workloads available",
    )

    variant(
        "version",
        default="21.1",
        values=("21.1",),
        description="app version (see NCT apps/versions.yaml)",
    )

    variant(
        "build",
        default="mpi-gcc",
        values=tuple(nct_builds),
        description="NCT build to run",
    )

    variant(
        "case",
        default="supercell",
        description="Upstream test case (run/config_files/<case>)",
    )

    variant(
        "timax",
        default="600",
        description="Simulated seconds (upstream supercell runs 7200)",
    )

    variant(
        "n_nodes",
        default="1",
        multi=True,
        description="Node counts to run (one experiment each)",
    )

    variant(
        "ranks_per_node",
        default="16",
        description="MPI ranks per node",
    )

    maintainers("neumanbrett")

    def compute_applications_section(self):
        app_dir = self.nct_app_dir(self.spec.variants["build"][0])
        self.add_experiment_variable("nct_app_dir", app_dir)
        self.add_experiment_variable("cm1_case", self.spec.variants["case"][0], True)
        self.add_experiment_variable("timax", self.spec.variants["timax"][0])
        self.add_experiment_variable(
            "n_nodes", [int(n) for n in self.spec.variants["n_nodes"]], named=True
        )
        self.add_experiment_variable(
            "n_ranks_per_node", int(self.spec.variants["ranks_per_node"][0]), True
        )

        self.set_required_variables(
            n_resources="{n_ranks}",
            process_problem_size="",
            total_problem_size="",
        )

    def compute_package_section(self):
        self.add_package_spec(self.name, ["cm1"])
