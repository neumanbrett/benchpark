# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from benchpark.directives import maintainers, variant
from benchpark.error import BenchparkError
from benchpark.experiment import Experiment
from benchpark.programming_model import ProgrammingModel, ProgrammingModelType
from benchpark.scaling import Scaling, ScalingMode


class OsuMicroBenchmarks(
    Experiment,
    ProgrammingModel(
        ProgrammingModelType.Mpionly,
        ProgrammingModelType.Cuda,
        ProgrammingModelType.Rocm,
    ),
    Scaling(ScalingMode.Strong, ScalingMode.Throughput),
):

    two_rank_workloads = [
        "osu_bibw",
        "osu_bw",
        "osu_latency",
        "osu_get_acc_latency",
        "osu_get_bw",
        "osu_get_latency",
        "osu_put_bibw",
        "osu_put_bw",
        "osu_put_latency",
    ]

    scalable_workloads = [
        "osu_latency_mp",
        "osu_latency_mt",
        "osu_mbw_mr",
        "osu_multi_lat",
        "osu_allgather",
        "osu_allreduce_persistent",
        "osu_alltoallw",
        "osu_bcast_persistent",
        "osu_iallgather",
        "osu_ialltoallw",
        "osu_ineighbor_allgather",
        "osu_ireduce",
        "osu_neighbor_allgatherv",
        "osu_reduce_persistent",
        "osu_scatterv",
        "osu_allgather_persistent",
        "osu_alltoall",
        "osu_alltoallw_persistent",
        "osu_gather",
        "osu_iallgatherv",
        "osu_ibarrier",
        "osu_ineighbor_allgatherv",
        "osu_ireduce_scatter",
        "osu_neighbor_alltoall",
        "osu_reduce_scatter",
        "osu_scatterv_persistent",
        "osu_allgatherv",
        "osu_alltoall_persistent",
        "osu_barrier",
        "osu_gather_persistent",
        "osu_iallreduce",
        "osu_ibcast",
        "osu_ineighbor_alltoall",
        "osu_iscatter",
        "osu_neighbor_alltoallv",
        "osu_reduce_scatter_persistent",
        "osu_allgatherv_persistent",
        "osu_alltoallv",
        "osu_barrier_persistent",
        "osu_gatherv",
        "osu_ialltoall",
        "osu_igather",
        "osu_ineighbor_alltoallv",
        "osu_iscatterv",
        "osu_neighbor_alltoallw",
        "osu_scatter",
        "osu_allreduce",
        "osu_alltoallv_persistent",
        "osu_bcast",
        "osu_gatherv_persistent",
        "osu_ialltoallv",
        "osu_igatherv",
        "osu_ineighbor_alltoallw",
        "osu_neighbor_allgather",
        "osu_reduce",
        "osu_scatter_persistent",
        "osu_cas_latency",
        "osu_fop_latency",
        "osu_hello",
        "osu_init",
    ]

    variant(
        "workload",
        default="osu_latency",
        values=two_rank_workloads + scalable_workloads,
        multi=True,
        description="workloads available",
    )

    variant(
        "version",
        default="7.5",
        values=("latest", "7.5"),
        description="app version",
    )

    # NCAR fork: 2-rank workloads: both, 1 (intra-node) or 2 (inter-node);
    # scalable workloads (collectives): any node count, the nodes filled with ranks.
    variant(
        "n_nodes",
        default="both",
        description="2-rank workloads: both, 1 (intra-node) or 2 (inter-node); "
        "scalable workloads: the node count (both: 1)",
    )

    # NCAR fork: scalable workloads fill the node with ranks; on SMT nodes (MI300A:
    # 192 cpus = 96 cores x 2) use threads_per_rank=2 so each rank gets a whole core
    # (OpenMPI binds to cores and refuses more ranks than cores).
    variant(
        "threads_per_rank",
        default="1",
        description="cpus per rank for scalable workloads (ranks = cores / this)",
    )

    maintainers("nhanford")

    def compute_applications_section(self):

        if self.spec.satisfies("exec_mode=test"):
            num_nodes = {"n_nodes": 1, "n_ranks": 2}
            for pk, pv in num_nodes.items():
                self.add_experiment_variable(pk, pv, True)

        device = None
        if self.spec.satisfies("+rocm"):
            device = "rocm"
        if self.spec.satisfies("+cuda"):
            device = "cuda"
        if device:
            # NCAR fork: OSU's point-to-point tests keep both buffers in host memory
            # unless given "D D", even with -d (c/util/osu_util.c). Collectives take
            # no buffer arguments.
            two_rank = any(
                self.spec.satisfies(f"workload={wl}") for wl in self.two_rank_workloads
            )
            args = f" -d {device}" + (" D D" if two_rank else "")
            self.add_experiment_variable("additional_args", args, False)

        use_gpus = self.spec.satisfies("+cuda") or self.spec.satisfies("+rocm")
        resource_var = "n_gpus" if use_gpus else "n_ranks"
        # self.add_experiment_variable("n_nodes", 2 , True)
        if any(self.spec.satisfies(f"workload={wl}") for wl in self.two_rank_workloads):
            if any(
                self.spec.satisfies(f"workload={wl}") for wl in self.scalable_workloads
            ):
                raise BenchparkError(
                    "This experiment must contain eithor only 2-rank workloads or only scalable workloads."
                )
            # Test the intra-node case and the inter-node case. Ignore user-set scaling params.
            choice = self.spec.variants["n_nodes"][0]
            if choice not in ("both", "1", "2"):
                raise BenchparkError("2-rank workloads take n_nodes=both, 1 or 2")
            n_nodes = [1, 2] if choice == "both" else [int(choice)]
            self.add_experiment_variable(resource_var, [2] * len(n_nodes), True)
            self.add_experiment_variable("n_nodes", n_nodes, True)
            self.add_experiment_variable("scaling-iterations", 1, True)
        else:
            sys = self.system_spec.system
            # "Fill" the nodes with ranks, and scale according to user parameters.
            threads = int(self.spec.variants["threads_per_rank"][0])
            resources_per_node = (
                sys.sys_gpus_per_node if use_gpus else sys.sys_cores_per_node // threads
            )
            choice = self.spec.variants["n_nodes"][0]
            nodes = 1 if choice == "both" else int(choice)
            self.add_experiment_variable(resource_var, resources_per_node * nodes, True)
            if threads != 1:
                self.add_experiment_variable("n_threads_per_proc", threads, True)
            self.add_experiment_variable("n_nodes", nodes, True)
        self.register_scaling_config(
            {
                ScalingMode.Strong: {
                    "n_nodes": lambda var, itr, dim, scaling_factor: var.val(dim)
                    * scaling_factor,
                    resource_var: lambda var, itr, dim, scaling_factor: var.val(dim)
                    * scaling_factor,
                    "process_problem_size": "",
                    "total_problem_size": "",
                },
                ScalingMode.Throughput: {
                    "n_nodes": None,
                },
            }
        )

        n_resources = "{" + resource_var + "}"
        self.set_required_variables(
            n_resources=n_resources, process_problem_size="", total_problem_size=""
        )

    def compute_package_section(self):
        self.add_package_spec(
            self.name, [f"osu-micro-benchmarks{self.determine_version()}"]
        )
