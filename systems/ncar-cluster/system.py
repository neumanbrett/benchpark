# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

import os

from benchpark.cudasystem import CudaSystem
from benchpark.directives import maintainers, variant
from benchpark.openmpsystem import OpenMPCPUOnlySystem
from benchpark.rocmsystem import ROCmSystem
from benchpark.system import JobQueue, System


class NcarCluster(System):
    """NSF NCAR clusters (Casper today; extensible to Derecho, etc.).

    ``cluster=<name>`` picks the machine and ``node=<profile>`` picks the
    hardware profile within it. Profiles were derived from ``pbsnodes -a``
    and grouped by PBS Qlist + cpu/gpu type.

    Notes:
      * ``sys_cores_per_node`` reflects ``resources_available.ncpus`` (what
        PBS will actually schedule) rather than the physical ``pcpus`` — on
        several node classes PBS reserves cores for the OS.
      * ``max_nodes`` on each ``JobQueue`` is the count of currently-online
        nodes in that profile, i.e. the largest job you can size on that
        profile today. Multi-hardware queues (``htc``, ``nvgpu``, ``vis``)
        span multiple profiles, so a queue's total node count is larger
        than any single profile's ``max_nodes``.
    """

    maintainers("bneuman")

    # Two-level catalogue: cluster -> profile -> resources.
    # Populated from ``pbsnodes -a`` on 2026-09-14.
    clusters = {
        "casper": {
            # ---- CPU-only, high-throughput computing (queue: htc) ----
            "htc-cascadelake": {
                "cpu_arch": "cascadelake",
                "pbs_cpu_type": "cascadelake",
                "sys_cores_per_node": 34,
                # 300, not 354: one node has 329 GB (pbsnodes 2026-11-02), and a
                # full-node request must fit every node of the profile.
                "sys_mem_per_node_GB": 300,
                "queues": [JobQueue("htc", 1440, 48)],
            },
            "htc-genoa": {
                "cpu_arch": "zen4",
                "pbs_cpu_type": "genoa",
                "sys_cores_per_node": 62,
                "sys_mem_per_node_GB": 732,
                "queues": [JobQueue("htc", 1440, 64)],
            },
            "htc-skylake": {
                "cpu_arch": "skylake_avx512",
                "pbs_cpu_type": "skylake",
                "sys_cores_per_node": 36,
                "sys_mem_per_node_GB": 354,
                "queues": [JobQueue("htc", 1440, 4)],
            },
            # ---- Large-memory CPU-only (queue: largemem) ----
            "largemem-cascadelake": {
                "cpu_arch": "cascadelake",
                "pbs_cpu_type": "cascadelake",
                "sys_cores_per_node": 36,
                "sys_mem_per_node_GB": 1488,
                "queues": [JobQueue("largemem", 1440, 2)],
            },
            "largemem-genoa": {
                "cpu_arch": "zen4",
                "pbs_cpu_type": "genoa",
                "sys_cores_per_node": 64,
                "sys_mem_per_node_GB": 1488,
                "queues": [JobQueue("largemem", 1440, 6)],
            },
            # ---- Visualization nodes (queue: vis; do have GPUs) ----
            "vis-skylake": {
                "cpu_arch": "skylake_avx512",
                "pbs_cpu_type": "skylake",
                "cuda_arch": "60",
                "sys_cores_per_node": 36,
                "sys_gpus_per_node": 6,
                "sys_mem_per_node_GB": 354,
                "gpu_type": "gp100_16gb",
                "queues": [JobQueue("vis", 1440, 7)],
            },
            "vis-genoa": {
                "cpu_arch": "zen4",
                "pbs_cpu_type": "genoa",
                "cuda_arch": "89",
                "sys_cores_per_node": 48,
                "sys_gpus_per_node": 7,
                "sys_mem_per_node_GB": 732,
                "gpu_type": "l40_45gb",
                "queues": [JobQueue("vis", 1440, 6)],
            },
            # ---- NVIDIA compute-GPU nodes (queue: nvgpu) ----
            "nvgpu-v100-8way-cascadelake": {
                "cpu_arch": "cascadelake",
                "pbs_cpu_type": "cascadelake",
                "cuda_arch": "70",
                "sys_cores_per_node": 36,
                "sys_gpus_per_node": 8,
                "sys_mem_per_node_GB": 1110,
                "gpu_type": "v100_32gb",
                "queues": [JobQueue("nvgpu", 1440, 2)],
            },
            "nvgpu-v100-4way-cascadelake": {
                "cpu_arch": "cascadelake",
                "pbs_cpu_type": "cascadelake",
                "cuda_arch": "70",
                "sys_cores_per_node": 36,
                "sys_gpus_per_node": 4,
                "sys_mem_per_node_GB": 732,
                "gpu_type": "v100_32gb",
                "queues": [JobQueue("nvgpu", 1440, 2)],
            },
            "nvgpu-v100-8way-skylake": {
                "cpu_arch": "skylake_avx512",
                "pbs_cpu_type": "skylake",
                "cuda_arch": "70",
                "sys_cores_per_node": 36,
                "sys_gpus_per_node": 8,
                "sys_mem_per_node_GB": 1110,
                "gpu_type": "v100_32gb",
                "queues": [JobQueue("nvgpu", 1440, 3)],
            },
            "nvgpu-v100-4way-skylake": {
                "cpu_arch": "skylake_avx512",
                "pbs_cpu_type": "skylake",
                "cuda_arch": "70",
                "sys_cores_per_node": 36,
                "sys_gpus_per_node": 4,
                "sys_mem_per_node_GB": 732,
                "gpu_type": "v100_32gb",
                "queues": [JobQueue("nvgpu", 1440, 1)],
            },
            "nvgpu-a100": {
                "cpu_arch": "zen3",
                "pbs_cpu_type": "milan",
                "cuda_arch": "80",
                "sys_cores_per_node": 128,
                "sys_gpus_per_node": 4,
                "sys_mem_per_node_GB": 985,
                "gpu_type": "a100_80gb",
                "queues": [JobQueue("nvgpu", 1440, 8)],
            },
            "nvgpu-h100": {
                "cpu_arch": "sapphirerapids",
                "pbs_cpu_type": "sapphirerapids",
                "cuda_arch": "90",
                "sys_cores_per_node": 64,
                "sys_gpus_per_node": 4,
                "sys_mem_per_node_GB": 985,
                "gpu_type": "h100_80gb",
                "queues": [JobQueue("nvgpu", 1440, 2)],
            },
            # ---- AMD GPU nodes (queue: amdgpu) ----
            "amdgpu-mi300a": {
                "cpu_arch": "zen4",
                "pbs_cpu_type": "mi300a",
                "rocm_arch": "gfx942",
                "sys_cores_per_node": 192,
                "sys_gpus_per_node": 4,
                # 480, not 496: PBS GB is 1024^3, and 496GB (520,093,696 kb) is just
                # over casper60's 520,064,476 kb (2026-11-03).
                "sys_mem_per_node_GB": 480,
                "gpu_type": "mi300a",
                "queues": [JobQueue("amdgpu", 1440, 2)],
            },
        },
        # Future clusters (Derecho, etc.) go here as additional top-level keys.
    }

    variant(
        "cluster",
        default="casper",
        values=("casper",),
        description="Which NCAR cluster to target",
    )

    variant(
        "node",
        default="htc-genoa",
        values=(
            "htc-cascadelake",
            "htc-genoa",
            "htc-skylake",
            "largemem-cascadelake",
            "largemem-genoa",
            "vis-skylake",
            "vis-genoa",
            "nvgpu-v100-8way-cascadelake",
            "nvgpu-v100-4way-cascadelake",
            "nvgpu-v100-8way-skylake",
            "nvgpu-v100-4way-skylake",
            "nvgpu-a100",
            "nvgpu-h100",
            "amdgpu-mi300a",
        ),
        description="Which node profile within the selected cluster to target",
    )

    # Lmod load order (after ``ncarenv/<ver>``) used with
    # ``package_manager=environment-modules``, keyed by ncarenv version then
    # (compiler, mpi) variant values. Only combinations listed here are runnable.
    module_stacks = {
        "25.10": {
            ("gcc", "openmpi"): ["gcc/14.3.0", "openmpi/5.0.8"],
            ("intel", "openmpi"): ["intel/2025.2.1", "openmpi/5.0.8"],
            ("nvhpc", "openmpi"): ["nvhpc/25.9", "openmpi/5.0.8"],
        },
        # AMD GPU nodes only: ROCm-aware OpenMPI (no parallel-netcdf in this stack).
        "25.10-rocm": {
            ("gcc", "openmpi"): ["gcc/14.3.0", "openmpi/5.0.9"],
        },
    }

    # Library modules an experiment can request by name, loaded after the
    # compiler/MPI stack, per ncarenv version.
    library_modules = {
        "25.10": {
            "cuda": "cuda/12.9.0",
            "netcdf": "netcdf/4.9.3",
            "parallel-netcdf": "parallel-netcdf/1.14.1",
        },
        "25.10-rocm": {
            "rocm": "rocm/6.4.3",
        },
    }

    # Application package name -> module, per ncarenv version.
    app_modules = {
        "25.10": {
            "osu-micro-benchmarks": "osu-micro-benchmarks/7.5.1",
        },
        "25.10-rocm": {
            "osu-micro-benchmarks": "osu-micro-benchmarks/7.5.1",
        },
    }

    variant(
        "account",
        default="SCSG0001",
        description="PBS project code passed as -A",
    )

    variant(
        "ncarenv",
        default="25.10",
        values=("25.10", "25.10-rocm"),
        description="ncarenv module version to load for environment-modules runs",
    )

    variant(
        "mem",
        default="default",
        description="Memory per node in GB for the select chunk (default: the whole "
        "node's with +exclusive, else PBS's 10gb)",
    )

    variant(
        "reservation",
        default="none",
        description="PBS reservation ID (e.g. R6430662) to submit to instead of the "
        "routing queue; the node profile must match the reserved nodes",
    )

    variant(
        "exclusive",
        default=True,
        description="Request whole nodes (place=scatter:excl); required for "
        "performance baselines",
    )

    variant(
        "compiler",
        default="gcc",
        values=("gcc", "intel", "nvhpc", "cce"),
        description="Which compiler to use",
    )

    variant(
        "mpi",
        default="openmpi",
        values=("openmpi", "mpich", "cray-mpich", "intel-oneapi-mpi"),
        description="Which MPI implementation to use",
    )

    variant(
        "queue",
        default="none",
        values=("none", "htc", "largemem", "vis", "nvgpu", "amdgpu", "system"),
        multi=False,
        description="PBS queue to submit to (defaults to the node profile's queue); "
        "system: the maintenance queue (submitted to directly, not routed)",
    )

    def __init__(self, spec):
        super().__init__(spec)

        cluster = self.spec.variants["cluster"][0]
        node = self.spec.variants["node"][0]

        cluster_profiles = self.clusters.get(cluster)
        if cluster_profiles is None:
            raise ValueError(f"Unknown NCAR cluster: {cluster}")
        attrs = cluster_profiles.get(node)
        if attrs is None:
            raise ValueError(
                f"Node profile '{node}' is not defined for cluster '{cluster}'. "
                f"Available: {sorted(cluster_profiles.keys())}"
            )
        for k, v in attrs.items():
            setattr(self, k, v)
        if self.spec.variants["queue"][0] == "system":
            # Same limits as the profile's own queue (the base class looks the name up).
            q = self.queues[0]
            self.queues = [*self.queues, JobQueue("system", q.time_limit, q.max_nodes)]

        # Programming models depend on the selected node.
        ncarenv = self.spec.variants["ncarenv"][0]
        rocm_stack = "rocm" in self.library_modules[ncarenv]
        if rocm_stack != node.startswith("amdgpu-"):
            raise ValueError(
                f"ncarenv={ncarenv} doesn't fit node={node}: use ncarenv=25.10-rocm "
                "on AMD GPU nodes only"
            )
        pms = [OpenMPCPUOnlySystem()]
        if node.startswith("nvgpu-") or node.startswith("vis-"):
            pms.append(CudaSystem())
            # CudaSystem needs these; CUDA comes from the ncarenv module and
            # there is no Cray GTL on Casper.
            self.cuda_version = self.library_modules[ncarenv]["cuda"].split("/")[1]
            self.gtl_flag = False
        elif node.startswith("amdgpu-"):
            pms.append(ROCmSystem())
            self.rocm_version = self.library_modules[ncarenv]["rocm"].split("/")[1]
            self.gtl_flag = False
        self.programming_models = pms

        self.scheduler = "pbs"

    # Execution queues (htc, nvgpu, ...) are from_route_only; jobs must be
    # submitted to the cluster's routing queue, which places them by resources.
    routing_queues = {"casper": "casper"}
    # Per-job ngpus limits of the GPU execution queues (qstat -Qf, 2026-11-02):
    # vis takes exactly 1, amdgpu 4-8, nvgpu 1-32.
    min_ngpus = {"vis": 1, "nvgpu": 1, "amdgpu": 4}

    def compute_variables_section(self):
        section = super().compute_variables_section()
        variables = section["variables"]
        reservation = self.spec.variants["reservation"][0]
        if reservation != "none":
            # A reservation is its own queue (qsub -q R<id>), covering only its nodes.
            variables["queue"] = reservation
        elif self.spec.variants["queue"][0] == "system":
            # Not route-only; reaches nodes the routing queue can't during maintenance.
            variables["queue"] = "system"
        elif variables.get("queue") in (None, *(q.name for q in self.queues)):
            variables["queue"] = self.routing_queues[self.spec.variants["cluster"][0]]
        return section

    def system_specific_variables(self):
        variables = super().system_specific_variables()
        # The allocation modifier reads the project code from ``bank``.
        variables["bank"] = self.spec.variants["account"][0]
        chunk = [f"cpu_type={self.pbs_cpu_type}"]
        if getattr(self, "sys_gpus_per_node", 0):
            # Casper chunks take ngpus=<per node> + gpu_type, not gpus=<per rank>.
            chunk.append(f"gpu_type={self.gpu_type}")
            variables["pbs_emit_ngpus"] = 1
            # A job without ngpus is routed to a CPU queue, where no node has this
            # gpu_type; the amdgpu queue also takes whole nodes only (ngpus >= 4).
            variables["pbs_min_ngpus_per_node"] = self.min_ngpus.get(
                self.queues[0].name, 1
            )
        mem = self.spec.variants["mem"][0]
        # PBS caps a job at its requested mem even on an exclusive node.
        if mem == "default" and self.spec.satisfies("+exclusive"):
            mem = self.sys_mem_per_node_GB
        if mem != "default":
            chunk.append(f"mem={int(mem)}GB")
        variables["pbs_chunk_resources"] = ":".join(chunk)
        # One chunk per node: plain excl lets PBS pack chunks that fit onto one host,
        # so an n_nodes=2 job could run on a single node.
        variables["pbs_place"] = (
            "scatter:excl" if self.spec.satisfies("+exclusive") else "scatter"
        )
        # Keep PBS output with the experiment (not in the qsub cwd) and record
        # which job/hosts produced the result.
        variables["extra_batch_opts"] = "-j oe\n-o {experiment_run_dir}/pbs_job.out"
        variables["pre_exec_cmds"] = (
            'echo "$PBS_JOBID" > pbs_jobid; sort -u "$PBS_NODEFILE" > pbs_hosts'
        )
        return variables

    def module_stack(self):
        """Ordered modules to load for environment-modules experiments."""
        ncarenv = self.spec.variants["ncarenv"][0]
        compiler = self.spec.variants["compiler"][0]
        mpi = self.spec.variants["mpi"][0]
        stack = self.module_stacks[ncarenv].get((compiler, mpi))
        if stack is None:
            raise ValueError(
                f"No module stack for compiler={compiler} mpi={mpi} "
                f"on ncarenv/{ncarenv}. "
                f"Available: {sorted(self.module_stacks[ncarenv])}"
            )
        return [f"ncarenv/{ncarenv}"] + stack

    def library_modules_for(self, libraries):
        """Modules for the named libraries in the selected ncarenv."""
        ncarenv = self.spec.variants["ncarenv"][0]
        available = self.library_modules[ncarenv]
        missing = [lib for lib in libraries if lib not in available]
        if missing:
            raise ValueError(
                f"No ncarenv/{ncarenv} module for {missing}. "
                f"Available: {sorted(available)}"
            )
        return [available[lib] for lib in libraries]

    def nct_build_dir(self, app, version, build):
        """Install dir of an app built by NCT's build scripts (not a module).

        Must match NCT's build.py: $NCT_DATA/bin/<cluster>/<ncarenv>/
        <compiler>-<mpi>/<app>/<version>/<build>.
        """
        nct_data = os.environ.get(
            "NCT_DATA", os.path.expandvars("/glade/work/$USER/nct-data")
        )
        return os.path.join(
            nct_data,
            "bin",
            self.spec.variants["cluster"][0],
            self.spec.variants["ncarenv"][0],
            f"{self.spec.variants['compiler'][0]}-{self.spec.variants['mpi'][0]}",
            app,
            version,
            build,
        )

    def module_for(self, package):
        """Module providing ``package`` in the selected ncarenv."""
        ncarenv = self.spec.variants["ncarenv"][0]
        module = self.app_modules[ncarenv].get(package)
        if module is None:
            raise ValueError(
                f"No ncarenv/{ncarenv} module for '{package}'. "
                f"Available: {sorted(self.app_modules[ncarenv])}"
            )
        return module

    def compute_software_section(self):
        """Placeholder software section.

        NCAR clusters provide compilers, MPI, and math libraries via Lmod
        modules rather than fixed system paths, so concrete externals are
        left out and Spack is allowed to resolve them.
        """
        compiler = self.spec.variants["compiler"][0]
        mpi = self.spec.variants["mpi"][0]
        return {
            "software": {
                "packages": {
                    "default-compiler": {"pkg_spec": compiler},
                    "compiler-gcc": {"pkg_spec": "gcc"},
                    "compiler-intel": {"pkg_spec": "intel-oneapi-compilers"},
                    "compiler-nvhpc": {"pkg_spec": "nvhpc"},
                    "default-mpi": {"pkg_spec": mpi},
                    "blas": {"pkg_spec": "openblas"},
                    "lapack": {"pkg_spec": "openblas"},
                }
            }
        }
