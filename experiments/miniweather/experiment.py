# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from benchpark.directives import maintainers, variant
from benchpark.error import BenchparkError
from benchpark.experiment import Experiment
from benchpark.nct_app import NctBuiltApp
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType


class Miniweather(
    NctBuiltApp,
    Experiment,
    ProgrammingModel(ProgrammingModelType.Mpionly),
):
    nct_version = "b001069"
    nct_builds = {
        "fortran-gcc": ("gcc", ["parallel-netcdf"], False),
        "fortran-nvhpc-gpu": ("nvhpc", ["parallel-netcdf", "cuda"], True),
        "cpp-gcc": ("gcc", ["parallel-netcdf"], False),
        "cpp-gcc-cuda": ("gcc", ["parallel-netcdf", "cuda"], True),
    }
    # Executables each build installs (NCT apps/miniweather/builds.yaml).
    build_executables = {
        "fortran-gcc": ("serial", "mpi", "openmp"),
        "fortran-nvhpc-gpu": ("mpi", "openacc", "openmp45", "do_concurrent"),
        "cpp-gcc": ("serial", "mpi", "parallelfor"),
        "cpp-gcc-cuda": ("parallelfor",),
    }

    variant(
        "workload",
        default="miniweather",
        values=("miniweather",),
        description="workloads available",
    )

    variant(
        "version",
        default="b001069",
        values=("b001069",),
        description="app version (pinned commit; see NCT apps/versions.yaml)",
    )

    variant(
        "build",
        default="fortran-gcc",
        values=tuple(nct_builds),
        description="NCT build to run (GPU builds need a GPU node profile)",
    )

    variant(
        "exe",
        default="mpi",
        values=(
            "serial",
            "mpi",
            "openmp",
            "openacc",
            "openmp45",
            "do_concurrent",
            "parallelfor",
        ),
        description="Executable (parallel model) from the build",
    )

    variant(
        "size",
        default="test",
        values=("test", "perf"),
        description="test: 100x50, 400 s (correctness); perf: 2000x1000, 100 s",
    )

    variant(
        "n_nodes",
        default="1",
        multi=True,
        description="Node counts to run (one experiment each)",
    )

    variant(
        "ranks_per_node",
        default="4",
        description="MPI ranks per node (GPU builds: one rank per GPU)",
    )

    variant(
        "threads",
        default="1",
        description="OpenMP threads per rank (exe=openmp)",
    )

    maintainers("neumanbrett")

    def compute_applications_section(self):
        build = self.spec.variants["build"][0]
        exe = self.spec.variants["exe"][0]
        app_dir = self.nct_app_dir(build)
        n_nodes = [int(n) for n in self.spec.variants["n_nodes"]]
        ranks_per_node = int(self.spec.variants["ranks_per_node"][0])
        gpu = self.nct_builds[build][2]

        if exe not in self.build_executables[build]:
            raise BenchparkError(
                f"build={build} has exe in {self.build_executables[build]}, not {exe}"
            )
        if exe == "serial" and (n_nodes != [1] or ranks_per_node != 1):
            raise BenchparkError("exe=serial needs n_nodes=1 ranks_per_node=1")
        gpus = self.system_spec.system.sys_gpus_per_node if gpu else 0
        if gpu and ranks_per_node > gpus:
            raise BenchparkError(
                f"GPU build: ranks_per_node (one GPU each) must be <= {gpus}"
            )

        suffix = "_test" if self.spec.satisfies("size=test") else ""
        self.add_experiment_variable("nct_app_dir", app_dir)
        self.add_experiment_variable("mw_build", build, named=True)
        self.add_experiment_variable("mw_exe", exe + suffix, named=True)
        self.add_experiment_variable(
            "mass_tol", "1e-9" if build.startswith("cpp") else "1e-13"
        )
        self.add_experiment_variable("n_nodes", n_nodes, named=True)
        self.add_experiment_variable("n_ranks_per_node", ranks_per_node, named=True)
        if gpu:
            self.add_experiment_variable(
                "n_gpus", [n * ranks_per_node for n in n_nodes]
            )
        threads = int(self.spec.variants["threads"][0])
        if threads != 1:
            self.add_experiment_variable("n_threads_per_proc", threads, named=True)
            self.set_environment_variable("OMP_NUM_THREADS", str(threads))

        self.set_required_variables(
            n_resources="{n_ranks}",
            process_problem_size="",
            total_problem_size="",
        )

    def compute_package_section(self):
        self.add_package_spec(self.name, ["miniweather"])
