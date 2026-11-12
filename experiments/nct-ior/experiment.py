# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from benchpark.directives import maintainers, variant
from benchpark.experiment import Experiment
from benchpark.nct_app import NctBuiltApp
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType


class NctIor(
    NctBuiltApp,
    Experiment,
    ProgrammingModel(ProgrammingModelType.Mpionly),
):
    """Filesystem bandwidth: IOR POSIX write+read, file-per-process or shared file."""

    nct_app = "ior"
    nct_version = "4.0.0"
    nct_builds = {"mpi-gcc": ("gcc", [], False)}
    # Per filesystem: (ranks per node, data per rank). /glade/work is small (Q4 of the
    # Phase 1b plan); /tmp's size isn't known for every node type.
    sizes = {"scratch": (16, "1g"), "tmp": (16, "256m"), "work": (4, "256m")}

    variant(
        "workload",
        default="fpp",
        values=("fpp", "shared"),
        description="fpp: one file per rank; shared: one file for all ranks",
    )

    variant(
        "version",
        default="4.0.0",
        values=("4.0.0",),
        description="app version (pinned commit; see NCT apps/versions.yaml)",
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

    variant(
        "block",
        default="default",
        description="Data per rank, IOR -b (default: 1g on scratch, else 256m)",
    )

    variant("n_nodes", default="1", description="Nodes")

    maintainers("neumanbrett")

    def compute_applications_section(self):
        io_path = self.spec.variants["io_path"][0]
        ranks, block = self.sizes[io_path]
        if self.spec.variants["ranks_per_node"][0] != "default":
            ranks = int(self.spec.variants["ranks_per_node"][0])
        if self.spec.variants["block"][0] != "default":
            block = self.spec.variants["block"][0]

        self.add_experiment_variable(
            "nct_app_dir", self.nct_app_dir(self.spec.variants["build"][0])
        )
        self.add_experiment_variable("io_path", io_path, named=True)
        self.add_experiment_variable("io_dir", self.io_paths[io_path])
        self.add_experiment_variable("block", block)
        self.add_experiment_variable(
            "n_nodes", int(self.spec.variants["n_nodes"][0]), named=True
        )
        self.add_experiment_variable("n_ranks_per_node", ranks, named=True)

        self.set_required_variables(
            n_resources="{n_ranks}",
            process_problem_size="{block}",
            total_problem_size="",
        )

    def compute_package_section(self):
        self.add_package_spec(self.name, ["nct-ior"])
