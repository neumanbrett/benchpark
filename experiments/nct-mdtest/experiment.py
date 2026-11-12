# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from benchpark.directives import maintainers, variant
from benchpark.experiment import Experiment
from benchpark.nct_app import NctBuiltApp
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType


class NctMdtest(
    NctBuiltApp,
    Experiment,
    ProgrammingModel(ProgrammingModelType.Mpionly),
):
    """Filesystem metadata rates: mdtest file create/stat/read/remove."""

    nct_app = "ior"
    nct_version = "4.0.0"  # mdtest comes from the IOR build
    nct_builds = {"mpi-gcc": ("gcc", [], False)}
    # Per filesystem: (ranks per node, files per rank). One directory per rank, so
    # each stays under ~1000 entries.
    sizes = {"scratch": (16, 1000), "tmp": (16, 1000), "work": (4, 500)}

    variant(
        "workload",
        default="files",
        values=("files",),
        description="workloads available",
    )

    variant(
        "version",
        default="4.0.0",
        values=("4.0.0",),
        description="IOR version (pinned commit; see NCT apps/versions.yaml)",
    )

    variant(
        "build",
        default="mpi-gcc",
        values=tuple(nct_builds),
        description="NCT build to run",
    )

    variant(
        "io_path",
        default="scratch",
        values=tuple(sizes),
        description="scratch: Lustre; work: GPFS /glade/work (small); tmp: node-local",
    )

    variant(
        "ranks_per_node",
        default="default",
        description="MPI ranks per node (default: 16, or 4 on work)",
    )

    maintainers("neumanbrett")

    def compute_applications_section(self):
        io_path = self.spec.variants["io_path"][0]
        ranks, files = self.sizes[io_path]
        if self.spec.variants["ranks_per_node"][0] != "default":
            ranks = int(self.spec.variants["ranks_per_node"][0])

        self.add_experiment_variable(
            "nct_app_dir", self.nct_app_dir(self.spec.variants["build"][0])
        )
        self.add_experiment_variable("io_path", io_path, named=True)
        self.add_experiment_variable("io_dir", self.io_paths[io_path])
        self.add_experiment_variable("files_per_rank", files)
        self.add_experiment_variable("n_nodes", 1, named=True)
        self.add_experiment_variable("n_ranks_per_node", ranks, named=True)

        self.set_required_variables(
            n_resources="{n_ranks}",
            process_problem_size="{files_per_rank}",
            total_problem_size="",
        )

    def compute_package_section(self):
        self.add_package_spec(self.name, ["nct-mdtest"])
