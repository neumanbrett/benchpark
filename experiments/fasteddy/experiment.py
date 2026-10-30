# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from benchpark.directives import maintainers, variant
from benchpark.error import BenchparkError
from benchpark.experiment import Experiment
from benchpark.nct_app import NctBuiltApp
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType


class Fasteddy(
    NctBuiltApp,
    Experiment,
    ProgrammingModel(ProgrammingModelType.Mpionly),
):
    nct_version = "v5.0.1"
    nct_builds = {
        "cuda-intel": ("intel", ["netcdf", "cuda"], True),
    }

    variant(
        "workload",
        default="fasteddy",
        values=("fasteddy",),
        description="workloads available",
    )

    variant(
        "version",
        default="v5.0.1",
        values=("v5.0.1",),
        description="app version (see NCT apps/versions.yaml)",
    )

    variant(
        "build",
        default="cuda-intel",
        values=tuple(nct_builds),
        description="NCT build to run",
    )

    variant(
        "case",
        default="Example02_CBL",
        values=("Example02_CBL",),
        description="Upstream tutorial input (tutorials/examples/<case>.in)",
    )
    case_nx = 600  # Nx in Example02_CBL.in

    variant(
        "steps",
        default="200",
        description="Timesteps to run (upstream Example02_CBL runs 288000)",
    )

    variant(
        "n_nodes",
        default="1",
        multi=True,
        description="Node counts to run (one experiment each)",
    )

    variant(
        "gpus_per_node",
        default="1",
        description="GPUs (= MPI ranks) per node",
    )

    maintainers("neumanbrett")

    def compute_applications_section(self):
        app_dir = self.nct_app_dir(self.spec.variants["build"][0])
        n_nodes = [int(n) for n in self.spec.variants["n_nodes"]]
        gpus_per_node = int(self.spec.variants["gpus_per_node"][0])

        if gpus_per_node > self.system_spec.system.sys_gpus_per_node:
            raise BenchparkError(
                f"gpus_per_node must be <= {self.system_spec.system.sys_gpus_per_node}"
            )
        # The domain is split in x only: Nx must divide by the rank count.
        bad = [n for n in n_nodes if self.case_nx % (n * gpus_per_node)]
        if bad:
            raise BenchparkError(
                f"Nx={self.case_nx} is not divisible by n_nodes x gpus_per_node "
                f"for n_nodes={bad}"
            )

        self.add_experiment_variable("nct_app_dir", app_dir)
        self.add_experiment_variable("fe_case", self.spec.variants["case"][0], True)
        self.add_experiment_variable("fe_steps", self.spec.variants["steps"][0])
        self.add_experiment_variable("n_nodes", n_nodes, named=True)
        self.add_experiment_variable("n_ranks_per_node", gpus_per_node, named=True)
        self.add_experiment_variable("n_gpus", [n * gpus_per_node for n in n_nodes])

        self.set_required_variables(
            n_resources="{n_gpus}",
            process_problem_size="",
            total_problem_size="",
        )

    def compute_package_section(self):
        self.add_package_spec(self.name, ["fasteddy"])
