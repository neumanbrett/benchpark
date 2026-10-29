# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
import sys

from ramble.app.builtin.osu_micro_benchmarks import (
    OsuMicroBenchmarks as OsuMicroBenchmarksBase,
)
from ramble.appkit import *


class OsuMicroBenchmarks(OsuMicroBenchmarksBase):

    tags = ['synthetic',
            'large-scale','multi-node','single-node',
            'atomics','managed-memory',
            'mpi','openshmem','upc','upc++','nccl','rccl',
            'network-bandwidth-bound','network-bisection-bandwidth-bound',
            'network-collectives','network-latency-bound',
            'network-multi-threaded','network-nonblocking-collectives',
            'network-onesided','network-point-to-point',
            'c','java','python','openacc']

    def _prepare_analysis(self, workspace, app_inst=None):
        # NCAR fork: the base class registers per-experiment FOMs into the
        # class-level figures_of_merit dict, so FOMs from an experiment analyzed
        # earlier (e.g. osu_bw "Bandwidth") leak into later ones (osu_latency).
        # Start each analysis from the FOMs defined on the class itself.
        cls = type(self)
        if "_static_foms" not in cls.__dict__:
            cls._static_foms = deepcopy(cls.figures_of_merit)
        self.figures_of_merit = deepcopy(cls._static_foms)
        super()._prepare_analysis(workspace, app_inst)
