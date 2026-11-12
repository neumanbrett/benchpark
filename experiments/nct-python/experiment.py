# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from benchpark.directives import maintainers, variant
from benchpark.error import BenchparkError
from benchpark.experiment import Experiment
from benchpark.nct_app import NctBuiltApp
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType

# Deployed conda envs (activate=path uses these directly).
NPL_ENVS = {
    "npl-2025b": "/glade/u/apps/opt/conda/envs/npl-2025b",
    "npl-2026a": "/glade/u/apps/opt/miniforge/envs/npl-2026a",
}


class NctPython(
    NctBuiltApp,
    Experiment,
    ProgrammingModel(ProgrammingModelType.Mpionly),
):
    """Python env tests on a deployed npl env (CPU, conda MPICH) or the NCT npl-gpu
    env (npl-2025b + CuPy, mpi4py on ncarenv OpenMPI)."""

    nct_app = "pytests"
    nct_version = "osu-7.5.1"
    nct_builds = {
        "scripts": ("gcc", [], False),
        "cupy-openmpi": ("gcc", [], False),  # the npl-gpu env
    }

    variant(
        "workload",
        default="env-health",
        values=("env-health", "osu-latency", "osu-bw", "osu-allreduce", "cupy-check"),
        description="cupy-check needs env=npl-gpu",
    )

    variant(
        "version",
        default="osu-7.5.1",
        values=("osu-7.5.1",),
        description="pytests version (OSU tarball; see NCT apps/versions.yaml)",
    )

    variant(
        "env",
        default="npl-2025b",
        values=("npl-2025b", "npl-2026a", "npl-gpu"),
        description="npl-2025b, npl-2026a: deployed envs as users get them (NumPy "
        "buffers, conda MPICH); npl-gpu: NCT's npl-2025b + CuPy (CuPy buffers, "
        "ncarenv OpenMPI)",
    )

    variant(
        "activate",
        default="path",
        values=("path", "module"),
        description="deployed envs: path runs the env's python/mpiexec by path; module "
        "runs `module load conda && conda activate <env>` first, as users do",
    )

    variant("n_nodes", default="1", description="Nodes (OSU p2p: 1 or 2)")

    variant(
        "ranks_per_node",
        default="default",
        description="MPI ranks per node (default: OSU p2p 2 on one node or 1 per "
        "node on two; allreduce every core, or every GPU with npl-gpu)",
    )

    variant(
        "gpus",
        default="all",
        description="cupy-check: GPUs to request and test (all: the node profile's)",
    )

    maintainers("neumanbrett")

    def compute_applications_section(self):
        system = self.system_spec.system
        workload = self.spec.variants["workload"][0]
        env = self.spec.variants["env"][0]
        activate = self.spec.variants["activate"][0]
        n_nodes = int(self.spec.variants["n_nodes"][0])
        gpus = getattr(system, "sys_gpus_per_node", 0)
        if env == "npl-gpu" and not getattr(system, "cuda_arch", None):
            raise BenchparkError("env=npl-gpu needs an NVIDIA GPU node profile")
        if workload == "cupy-check" and env != "npl-gpu":
            raise BenchparkError("workload=cupy-check needs env=npl-gpu")
        if activate == "module" and env not in NPL_ENVS:
            raise BenchparkError(f"activate=module needs one of {', '.join(NPL_ENVS)}")

        self.add_experiment_variable("nct_app_dir", self.nct_app_dir("scripts"))
        # Path-activated names are unchanged, so they compare with earlier runs.
        name = env if activate == "path" else f"{env}-module"
        self.add_experiment_variable("env", name, named=True)

        if env in NPL_ENVS:
            if activate == "module":
                # Runs in the same shell as the test (prefixed to its command line).
                self.add_experiment_variable(
                    "py_setup", f"module load conda && conda activate {env} &&"
                )
                self.add_experiment_variable("health_args", "--activated")
                bindir = ""
            else:
                bindir = f"{NPL_ENVS[env]}/bin/"
            self.add_experiment_variable("py", f"{bindir}python")
            # conda MPICH's hydra (not ncarenv's mpiexec). pbs_hosts (one line per
            # host, written by the job script) rather than $PBS_NODEFILE (one per rank).
            launch = (
                f"{bindir}mpiexec -n {{n_ranks}} -ppn {{n_ranks_per_node}} "
                "-f pbs_hosts"
            )
            # Which MPICH device/provider it picked goes in the log.
            self.set_environment_variable("MPIR_CVAR_DEBUG_SUMMARY", "1")
            self.add_experiment_variable("buffer", "numpy")
            self.add_experiment_variable("gpu_bind", "")
        else:
            env_dir = self.nct_app_dir("cupy-openmpi", app="npl-gpu", version="2025b")
            self.add_experiment_variable("py", f"{env_dir}/env/bin/python")
            # ncarenv OpenMPI (CUDA-aware); the env's own mpiexec is MPICH's.
            launch = "mpiexec -n {n_ranks} --map-by ppr:{n_ranks_per_node}:node"
            self.add_experiment_variable("buffer", "cupy")
            self.add_experiment_variable("gpu_bind", "{nct_app_dir}/checks/gpu_bind.sh")
        self.add_experiment_variable("py_launch", launch)

        if workload in ("osu-latency", "osu-bw"):
            if n_nodes not in (1, 2):
                raise BenchparkError("OSU p2p tests use n_nodes=1 or 2")
            ranks = 2 // n_nodes
            self.add_experiment_variable("osu_bench", workload.split("-")[1])
            self.add_experiment_variable("osu_args", "")
        elif workload == "osu-allreduce":
            ranks = gpus if env == "npl-gpu" else system.sys_cores_per_node
            self.add_experiment_variable("osu_bench", "allreduce")
            self.add_experiment_variable("osu_args", "--max 1048576")
        else:
            if n_nodes != 1:
                raise BenchparkError(f"workload={workload} runs on one node")
            ranks = 1
        if self.spec.variants["ranks_per_node"][0] != "default":
            ranks = int(self.spec.variants["ranks_per_node"][0])

        self.add_experiment_variable("n_nodes", n_nodes, named=True)
        self.add_experiment_variable("n_ranks_per_node", ranks, named=True)
        if env == "npl-gpu":
            n_gpus = ranks
            if workload == "cupy-check":
                n_gpus = self.spec.variants["gpus"][0]
                n_gpus = gpus if n_gpus == "all" else int(n_gpus)
            self.add_experiment_variable("n_gpus", n_nodes * n_gpus, named=True)

        self.set_required_variables(
            n_resources="{n_ranks}",
            process_problem_size="",
            total_problem_size="",
        )

    def compute_package_section(self):
        self.add_package_spec(self.name, ["nct-python"])
