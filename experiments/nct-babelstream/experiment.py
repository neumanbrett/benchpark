# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from benchpark.directives import maintainers, variant
from benchpark.error import BenchparkError
from benchpark.experiment import Experiment
from benchpark.nct_app import NctBuiltApp
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType


class NctBabelstream(
    NctBuiltApp,
    Experiment,
    ProgrammingModel(ProgrammingModelType.Mpionly),
):
    """GPU memory bandwidth: BabelStream on each GPU of one node, one at a time."""

    nct_app = "babelstream"
    nct_version = "5.0"
    nct_builds = {
        "cuda": ("gcc", ["cuda"], True),
        "hip": ("gcc", ["rocm"], True),
    }

    variant(
        "workload",
        default="babelstream",
        values=("babelstream",),
        description="workloads available",
    )

    variant(
        "version",
        default="5.0",
        values=("5.0",),
        description="app version (pinned commit; see NCT apps/versions.yaml)",
    )

    variant(
        "build",
        default="cuda",
        values=tuple(nct_builds),
        description="NCT build to run (cuda: NVIDIA profiles, hip: amdgpu-mi300a)",
    )

    variant(
        "gpus",
        default="all",
        description="GPUs to request and test (all: every GPU of the node profile)",
    )

    maintainers("neumanbrett")

    def compute_applications_section(self):
        app_dir = self.nct_app_dir(self.spec.variants["build"][0])
        per_node = self.system_spec.system.sys_gpus_per_node
        gpus = self.spec.variants["gpus"][0]
        gpus = per_node if gpus == "all" else int(gpus)
        if not 0 < gpus <= per_node:
            raise BenchparkError(f"gpus must be 1..{per_node} on this node profile")

        self.add_experiment_variable("nct_app_dir", app_dir)
        self.add_experiment_variable("n_nodes", 1, named=True)
        self.add_experiment_variable("n_ranks_per_node", 1, named=True)
        self.add_experiment_variable("n_gpus", gpus, named=True)
        self.add_experiment_variable("bs_devices", gpus)

        self.set_required_variables(
            n_resources="{n_gpus}",
            process_problem_size="",
            total_problem_size="",
        )

    def compute_package_section(self):
        self.add_package_spec(self.name, ["nct-babelstream"])
