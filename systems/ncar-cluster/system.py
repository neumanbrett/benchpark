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
    """NSF NCAR clusters: Casper and Derecho.

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
        # Derecho: one CPU and one GPU node type (pbsnodes / qstat -Qf @desched1,
        # 2026-11-24). Memory is the queue's default_chunk.mem. Walltimes are the
        # documented 12 h (main) / 6 h (develop); PBS itself sets no queue maximum.
        # develop is shared and capped per user at 256 cores / 8 GPUs: 2 nodes.
        "derecho": {
            "cpu-milan": {
                "cpu_arch": "zen3",
                "pbs_cpu_type": "milan",
                "sys_cores_per_node": 128,
                "sys_mem_per_node_GB": 235,
                "queues": [JobQueue("cpu", 720, 2468), JobQueue("develop", 360, 2)],
            },
            "gpu-a100": {
                "cpu_arch": "zen3",
                "pbs_cpu_type": "milan",
                "cuda_arch": "80",
                "sys_cores_per_node": 64,
                "sys_gpus_per_node": 4,
                "sys_mem_per_node_GB": 487,
                "gpu_type": "a100_40gb",
                "queues": [JobQueue("gpu", 720, 82), JobQueue("develop", 360, 2)],
            },
        },
    }

    variant(
        "cluster",
        default="casper",
        values=("casper", "derecho"),
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
            "cpu-milan",
            "gpu-a100",
        ),
        description="Which node profile within the selected cluster to target",
    )

    # Lmod load order (after ``ncarenv/<ver>``) used with
    # ``package_manager=environment-modules``, keyed by (cluster, ncarenv version)
    # then (compiler, mpi) variant values. Only combinations listed here are runnable.
    module_stacks = {
        ("casper", "25.10"): {
            ("gcc", "openmpi"): ["gcc/14.3.0", "openmpi/5.0.8"],
            ("intel", "openmpi"): ["intel/2025.2.1", "openmpi/5.0.8"],
            ("nvhpc", "openmpi"): ["nvhpc/25.9", "openmpi/5.0.8"],
        },
        # AMD GPU nodes only: ROCm-aware OpenMPI (no parallel-netcdf in this stack).
        ("casper", "25.10-rocm"): {
            ("gcc", "openmpi"): ["gcc/14.3.0", "openmpi/5.0.9"],
        },
        # A candidate stack (Phase 2 validation): bare names, so a job loads the
        # stack's defaults for this ncarenv and records the versions that resolved.
        # Needs module_init=<the stack's use_modules> to put its tree on the path.
        ("casper", "26.10"): {
            ("gcc", "openmpi"): ["gcc", "openmpi"],
            ("intel", "openmpi"): ["intel", "openmpi"],
            ("nvhpc", "openmpi"): ["nvhpc", "openmpi"],
        },
        # Derecho: cray-mpich only for now (8.1.32 is the stack default).
        ("derecho", "25.10"): {
            ("gcc", "cray-mpich"): ["gcc/14.3.0", "cray-mpich/8.1.32"],
            ("intel", "cray-mpich"): ["intel/2025.2.1", "cray-mpich/8.1.32"],
            ("nvhpc", "cray-mpich"): ["nvhpc/25.9", "cray-mpich/8.1.32"],
            ("cce", "cray-mpich"): ["cce/19.0.0", "cray-mpich/8.1.32"],
        },
    }

    # Library modules an experiment can request by name, loaded after the
    # compiler/MPI stack, per (cluster, ncarenv version).
    library_modules = {
        ("casper", "25.10"): {
            "cuda": "cuda/12.9.0",
            "netcdf": "netcdf/4.9.3",
            "parallel-netcdf": "parallel-netcdf/1.14.1",
        },
        ("casper", "25.10-rocm"): {
            "rocm": "rocm/6.4.3",
        },
        ("casper", "26.10"): {
            "cuda": "cuda",
            "netcdf": "netcdf",
            "parallel-netcdf": "parallel-netcdf",
        },
        ("derecho", "25.10"): {
            "cuda": "cuda/12.9.0",
            "netcdf": "netcdf/4.9.3",
            "parallel-netcdf": "parallel-netcdf/1.14.1",
        },
    }

    # Application package name -> module, per (cluster, ncarenv version).
    app_modules = {
        ("casper", "25.10"): {
            "osu-micro-benchmarks": "osu-micro-benchmarks/7.5.1",
        },
        ("casper", "25.10-rocm"): {
            "osu-micro-benchmarks": "osu-micro-benchmarks/7.5.1",
        },
        ("casper", "26.10"): {
            "osu-micro-benchmarks": "osu-micro-benchmarks",
        },
        # Every compiler's build is +cuda cuda_arch=80 and loads cuda/12.9.0.
        ("derecho", "25.10"): {
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
        values=("25.10", "25.10-rocm", "26.10"),
        description="ncarenv module version to load for environment-modules runs",
    )

    # Phase 2: a candidate stack whose module tree isn't on the default path.
    variant(
        "module_init",
        default="none",
        description="Script each job sources before loading modules (a candidate "
        "stack's use_modules); none: the system's module tree",
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
        values=(
            "none",
            "htc",
            "largemem",
            "vis",
            "nvgpu",
            "amdgpu",
            "system",
            "cpu",
            "gpu",
            "develop",
        ),
        multi=False,
        description="PBS queue to submit to (defaults to the node profile's queue); "
        "system: the maintenance queue (submitted to directly, not routed); "
        "develop: Derecho's shared development queue",
    )

    variant(
        "submit_from",
        default="auto",
        values=("auto", "casper", "derecho"),
        description="Cluster the jobs are submitted from; a job for the other "
        "cluster names its PBS server (queue@server). auto: $NCAR_HOST where "
        "benchpark runs",
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
        queue = self.spec.variants["queue"][0]
        if queue == "system":
            # Same limits as the profile's own queue (the base class looks the name up).
            q = self.queues[0]
            self.queues = [*self.queues, JobQueue("system", q.time_limit, q.max_nodes)]
        elif queue != "none" and queue not in (q.name for q in self.queues):
            raise ValueError(
                f"queue={queue} doesn't serve node={node} on {cluster}. "
                f"Available: {[q.name for q in self.queues]} (or system)"
            )

        # Programming models depend on the selected node.
        ncarenv = self.spec.variants["ncarenv"][0]
        if (cluster, ncarenv) not in self.module_stacks:
            raise ValueError(
                f"No ncarenv={ncarenv} for cluster={cluster}. Available: "
                f"{sorted(e for c, e in self.module_stacks if c == cluster)}"
            )
        libraries = self.library_modules[(cluster, ncarenv)]
        rocm_stack = "rocm" in libraries
        if rocm_stack != node.startswith("amdgpu-"):
            raise ValueError(
                f"ncarenv={ncarenv} doesn't fit node={node}: use ncarenv=25.10-rocm "
                "on AMD GPU nodes only"
            )
        pms = [OpenMPCPUOnlySystem()]
        if getattr(self, "cuda_arch", None):
            pms.append(CudaSystem())
            # CudaSystem needs these; CUDA comes from the ncarenv module. Cray GTL
            # (GPU-aware cray-mpich) exists on Derecho only.
            self.cuda_version = self.module_version(libraries["cuda"])
            self.gtl_flag = cluster == "derecho"
            if self.gtl_flag:
                # Used by +cuda experiments only. NCAR's documented launcher: per rank
                # it sets CUDA_VISIBLE_DEVICES from the local rank and $NGPUS, plus
                # MPICH_GPU_SUPPORT_ENABLED=1 and MPICH_OFI_NIC_POLICY=GPU. GPU support
                # makes cray-mpich abort without GTL and may slow host-buffer MPI
                # (job 8297269), so host-buffer runs don't use it.
                self.gpu_mpi_wrapper = "set_gpu_rank"
                # The 25.10 OSU modules aren't linked to the GTL: preload it into the
                # ranks only (not python/module). It needs libcudart, so +cuda only.
                self.gpu_mpi_cmd_opts = f"--env LD_PRELOAD={self.gtl_cuda_lib}"
        elif node.startswith("amdgpu-"):
            pms.append(ROCmSystem())
            self.rocm_version = self.module_version(libraries["rocm"])
            self.gtl_flag = False
        self.programming_models = pms

        self.scheduler = "pbs"

    # Execution queues (htc, nvgpu, ...) are from_route_only; jobs must be
    # submitted to the cluster's routing queue, which places them by resources.
    routing_queues = {"casper": "casper", "derecho": "main"}
    # PBS server of each cluster, for submitting from the other one (queue@server).
    pbs_servers = {"casper": "casper-pbs", "derecho": "desched1"}
    # Per-job ngpus limits of the GPU execution queues (qstat -Qf, 2026-11-02):
    # vis takes exactly 1, amdgpu 4-8, nvgpu 1-32; Derecho gpu/gpudev take >= 1.
    min_ngpus = {"vis": 1, "nvgpu": 1, "amdgpu": 4, "gpu": 1}
    # Clusters where cpu_type is a host-level PBS resource (qmgr flag h), so a select
    # chunk may name it. On Derecho it isn't: qsub rejects the select (2026-11-24).
    cpu_type_selectable = {"casper"}
    # cray-mpich's CUDA GTL on Derecho (cray-mpich/8.1.32).
    gtl_cuda_lib = "/opt/cray/pe/mpich/8.1.32/gtl/lib/libmpi_gtl_cuda.so"

    def compute_variables_section(self):
        section = super().compute_variables_section()
        variables = section["variables"]
        cluster = self.spec.variants["cluster"][0]
        reservation = self.spec.variants["reservation"][0]
        queue = self.spec.variants["queue"][0]
        if reservation != "none":
            # A reservation is its own queue (qsub -q R<id>), covering only its nodes.
            variables["queue"] = reservation
        elif queue in ("system", "develop"):
            # system: not route-only; reaches nodes the routing queue can't during
            # maintenance. develop: Derecho's own routing queue (to cpudev/gpudev).
            variables["queue"] = queue
        elif variables.get("queue") in (None, *(q.name for q in self.queues)):
            variables["queue"] = self.routing_queues[cluster]
        variables["queue"] += self.pbs_server_suffix()
        return section

    def pbs_server_suffix(self):
        """'@<server>' when jobs are submitted from the other cluster, else ''.

        auto reads $NCAR_HOST (set by ncarenv) where benchpark runs, so a workspace
        set up on one cluster must be submitted from that same cluster.
        """
        cluster = self.spec.variants["cluster"][0]
        submit_from = self.spec.variants["submit_from"][0]
        if submit_from == "auto":
            submit_from = os.environ.get("NCAR_HOST", cluster)
        if submit_from == cluster:
            return ""
        return f"@{self.pbs_servers[cluster]}"

    def system_specific_variables(self):
        variables = super().system_specific_variables()
        # The allocation modifier reads the project code from ``bank``.
        variables["bank"] = self.spec.variants["account"][0]
        chunk = []
        if self.spec.variants["cluster"][0] in self.cpu_type_selectable:
            chunk.append(f"cpu_type={self.pbs_cpu_type}")
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
        pre_exec = 'echo "$PBS_JOBID" > pbs_jobid; sort -u "$PBS_NODEFILE" > pbs_hosts'
        if self.spec.variants["cluster"][0] == "derecho":
            # Cray PALS mpiexec: -n plus --ppn (it has no documented -np, and
            # without --ppn ranks aren't spread one chunk's worth per node).
            variables["pbs_mpiexec_ppn"] = 1
            if getattr(self, "sys_gpus_per_node", 0):
                # PBS sets CUDA_VISIBLE_DEVICES to the head node's GPU UUIDs and
                # PALS copies it to every node, so remote ranks find no GPU. The
                # cgroup already limits each node to the job's GPUs (job 8297269).
                pre_exec += "\nunset CUDA_VISIBLE_DEVICES"
        module_init = self.spec.variants["module_init"][0]
        if module_init != "none":
            # A candidate stack: its module tree, before the job's module load line.
            pre_exec += f"\n. {module_init} > module_init.out 2>&1"
        variables["pre_exec_cmds"] = pre_exec
        # The module files the job actually loaded, so collect can check where
        # they came from (a candidate's must all be from its own tree).
        variables["post_exec_cmds"] = "echo \"$_LMFILES_\" | tr ':' '\\n' > lmod_files"
        return variables

    @staticmethod
    def module_version(module):
        """'cuda/12.9.0' -> '12.9.0'; a bare 'cuda' (a stack default) -> 'default'."""
        return module.split("/", 1)[1] if "/" in module else "default"

    def stack_key(self):
        """(cluster, ncarenv): the key of the module tables."""
        return (self.spec.variants["cluster"][0], self.spec.variants["ncarenv"][0])

    def module_stack(self):
        """Ordered modules to load for environment-modules experiments."""
        cluster, ncarenv = key = self.stack_key()
        compiler = self.spec.variants["compiler"][0]
        mpi = self.spec.variants["mpi"][0]
        stack = self.module_stacks[key].get((compiler, mpi))
        if stack is None:
            raise ValueError(
                f"No module stack for compiler={compiler} mpi={mpi} "
                f"on {cluster} ncarenv/{ncarenv}. "
                f"Available: {sorted(self.module_stacks[key])}"
            )
        return [f"ncarenv/{ncarenv}"] + stack

    def library_modules_for(self, libraries):
        """Modules for the named libraries in the selected cluster and ncarenv."""
        cluster, ncarenv = key = self.stack_key()
        available = self.library_modules[key]
        missing = [lib for lib in libraries if lib not in available]
        if missing:
            raise ValueError(
                f"No {cluster} ncarenv/{ncarenv} module for {missing}. "
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
        cluster, ncarenv = key = self.stack_key()
        module = self.app_modules[key].get(package)
        if module is None:
            raise ValueError(
                f"No {cluster} ncarenv/{ncarenv} module for '{package}'. "
                f"Available: {sorted(self.app_modules[key])}"
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
                    "compiler-cce": {"pkg_spec": "cce"},
                    "default-mpi": {"pkg_spec": mpi},
                    "blas": {"pkg_spec": "openblas"},
                    "lapack": {"pkg_spec": "openblas"},
                }
            }
        }
