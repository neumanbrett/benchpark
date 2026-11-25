# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import benchpark.spec


def ncar(spec, monkeypatch, host="casper"):
    monkeypatch.setenv("NCAR_HOST", host)
    return benchpark.spec.SystemSpec(f"ncar-cluster {spec}").concretize().system


def test_casper_unchanged(monkeypatch):
    system = ncar("cluster=casper node=nvgpu-a100", monkeypatch)
    v = system.compute_variables_section()["variables"]
    assert v["queue"] == "casper"
    assert v["pbs_chunk_resources"] == "cpu_type=milan:gpu_type=a100_80gb:mem=985GB"
    assert "pbs_mpiexec_ppn" not in v
    assert v["gtl_flag"] is False
    assert system.module_stack() == ["ncarenv/25.10", "gcc/14.3.0", "openmpi/5.0.8"]
    assert system.library_modules_for(["cuda"]) == ["cuda/12.9.0"]


def test_derecho_cpu(monkeypatch):
    system = ncar(
        "cluster=derecho node=cpu-milan compiler=cce mpi=cray-mpich",
        monkeypatch,
        "derecho",
    )
    v = system.compute_variables_section()["variables"]
    assert v["queue"] == "main"
    assert v["pbs_chunk_resources"] == "mem=235GB"
    assert v["pbs_place"] == "scatter:excl"
    assert v["pbs_mpiexec_ppn"] == 1
    assert "MPICH_GPU_SUPPORT_ENABLED" not in v["pre_exec_cmds"]
    assert system.module_stack() == ["ncarenv/25.10", "cce/19.0.0", "cray-mpich/8.1.32"]
    assert system.module_for("osu-micro-benchmarks") == "osu-micro-benchmarks/7.5.1"


def test_derecho_gpu_develop(monkeypatch):
    system = ncar(
        "cluster=derecho node=gpu-a100 mpi=cray-mpich queue=develop ~exclusive",
        monkeypatch,
        "derecho",
    )
    v = system.compute_variables_section()["variables"]
    assert v["queue"] == "develop"
    assert v["max_nodes"] == 2
    assert v["pbs_chunk_resources"] == "gpu_type=a100_40gb"
    assert v["pbs_place"] == "scatter"
    assert v["pbs_min_ngpus_per_node"] == 1
    assert v["gtl_flag"] is True
    assert v["cuda_version"] == "12.9.0"
    # GPU-aware MPI is per experiment (+cuda), not in every job on a GPU node.
    assert system.gpu_mpi_wrapper == "set_gpu_rank"
    assert system.gpu_mpi_cmd_opts == f"--env LD_PRELOAD={system.gtl_cuda_lib}"
    assert "extra_cmd_opts" not in v
    assert "unset CUDA_VISIBLE_DEVICES" in v["pre_exec_cmds"]


def test_casper_gpu_no_derecho_extras(monkeypatch):
    system = ncar("cluster=casper node=nvgpu-a100", monkeypatch)
    v = system.compute_variables_section()["variables"]
    assert not hasattr(system, "gpu_mpi_wrapper")
    assert not hasattr(system, "gpu_mpi_cmd_opts")
    assert "CUDA_VISIBLE_DEVICES" not in v["pre_exec_cmds"]
    assert "extra_cmd_opts" not in v


@pytest.mark.parametrize(
    "spec,host,queue",
    [
        ("cluster=derecho node=cpu-milan mpi=cray-mpich", "casper", "main@desched1"),
        (
            "cluster=derecho node=cpu-milan mpi=cray-mpich queue=develop",
            "casper",
            "develop@desched1",
        ),
        ("cluster=casper node=htc-genoa", "derecho", "casper@casper-pbs"),
        (
            "cluster=casper node=htc-genoa reservation=R123",
            "derecho",
            "R123@casper-pbs",
        ),
        # submit_from overrides $NCAR_HOST
        (
            "cluster=derecho node=cpu-milan mpi=cray-mpich submit_from=derecho",
            "casper",
            "main",
        ),
    ],
)
def test_cross_submission(monkeypatch, spec, host, queue):
    v = ncar(spec, monkeypatch, host).compute_variables_section()["variables"]
    assert v["queue"] == queue


@pytest.mark.parametrize(
    "spec,match",
    [
        ("cluster=derecho node=htc-genoa", "not defined for cluster 'derecho'"),
        ("cluster=derecho node=cpu-milan ncarenv=26.10", "No ncarenv=26.10"),
        ("cluster=casper node=htc-genoa queue=develop", "doesn't serve node"),
    ],
)
def test_bad_combinations(monkeypatch, spec, match):
    with pytest.raises(ValueError, match=match):
        ncar(spec, monkeypatch)


def test_derecho_openmpi_rejected(monkeypatch):
    system = ncar("cluster=derecho node=cpu-milan", monkeypatch, "derecho")
    with pytest.raises(
        ValueError, match="No module stack for compiler=gcc mpi=openmpi"
    ):
        system.module_stack()
