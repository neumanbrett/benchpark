# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from benchpark.directives import maintainers, variant
from benchpark.experiment import Experiment
from benchpark.nct_app import NctBuiltApp
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType


class NctStream(
    NctBuiltApp,
    Experiment,
    ProgrammingModel(ProgrammingModelType.Mpionly),
):
    """CPU memory bandwidth: one STREAM process with an OpenMP thread per core."""

    nct_app = "stream"
    nct_version = "5.10-caliper"
    nct_builds = {"omp-gcc": ("gcc", [], False)}

    variant(
        "workload",
        default="stream",
        values=("stream",),
        description="workloads available",
    )

    variant(
        "version",
        default="5.10-caliper",
        values=("5.10-caliper",),
        description="app version (pinned commit; see NCT apps/versions.yaml)",
    )

    variant(
        "build",
        default="omp-gcc",
        values=tuple(nct_builds),
        description="NCT build to run",
    )

    variant(
        "threads",
        default="all",
        description="OpenMP threads (all: one per core of the node profile)",
    )

    variant(
        "size",
        default="400000000",
        description="Elements per array (3 arrays of doubles; the default, 3.2 GB "
        "each, is over 4x the largest Casper node's total L3)",
    )

    maintainers("neumanbrett")

    def compute_applications_section(self):
        app_dir = self.nct_app_dir(self.spec.variants["build"][0])
        threads = self.spec.variants["threads"][0]
        if threads == "all":
            threads = self.system_spec.system.sys_cores_per_node
        threads = int(threads)

        self.add_experiment_variable("nct_app_dir", app_dir)
        self.add_experiment_variable("n_nodes", 1, named=True)
        self.add_experiment_variable("n_ranks_per_node", 1, named=True)
        self.add_experiment_variable("n_threads_per_proc", threads, named=True)
        self.add_experiment_variable("s", self.spec.variants["size"][0])
        self.set_environment_variable("OMP_NUM_THREADS", str(threads))
        self.set_environment_variable("OMP_PROC_BIND", "spread")
        self.set_environment_variable("OMP_PLACES", "cores")

        self.set_required_variables(
            n_resources="{n_ranks}",
            process_problem_size="{s}",
            total_problem_size="{s}",
        )

    def compute_package_section(self):
        self.add_package_spec(self.name, ["nct-stream"])
