# Copyright 2023 Lawrence Livermore National Security, LLC and other
# Benchpark Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: Apache-2.0

"""NCAR fork: experiments for apps built by NCT's build scripts.

NCT (nct/build.py) builds these apps with the same ncarenv module stack the run
loads, and installs them under the system's ``nct_build_dir``. Runs need
``package_manager=environment-modules`` on a system with a module stack.
"""

import json
import os

from benchpark.error import BenchparkError


class NctBuiltApp:
    nct_built = True

    # Build name -> (compiler it needs, library modules, targets a GPU).
    # Must match NCT's apps/<app>/builds.yaml.
    nct_builds = {}

    # NCT app name of the build, when the experiment's name differs (nct-stream
    # runs the "stream" build).
    nct_app = None

    def nct_app_dir(self, build, app=None, version=None):
        """Install dir of an NCT build; checks the build matches the system.

        ``app``/``version`` name a second NCT app the experiment also uses (e.g. a
        Python env); its build must be listed in ``nct_builds`` too.
        """
        if self.spec.variants["package_manager"][0] != "environment-modules":
            raise BenchparkError(
                f"{self.name} is built by NCT; use package_manager=environment-modules"
            )
        system = self.system_spec.system
        compiler, libraries, gpu = self.nct_builds[build]
        if system.spec.variants["compiler"][0] != compiler:
            raise BenchparkError(
                f"{self.name} build '{build}' needs a system with compiler={compiler}"
            )
        # Keep load order; a second app adds its libraries after the first.
        self.module_libraries = list(
            dict.fromkeys([*self.module_libraries, *libraries])
        )

        build_name = build
        if gpu:
            cuda_arch = getattr(system, "cuda_arch", None)
            rocm_arch = getattr(system, "rocm_arch", None)
            if not (cuda_arch or rocm_arch):
                raise BenchparkError(
                    f"{self.name} build '{build}' needs a GPU node profile"
                )
            # Must match nct/build.py's install dir names (e.g. cuda-cc80, hip-gfx942).
            build_name = (
                f"{build}-cc{cuda_arch}" if cuda_arch else f"{build}-{rocm_arch}"
            )

        app = app or self.nct_app or self.name
        app_dir = system.nct_build_dir(app, version or self.nct_version, build_name)
        record = os.path.join(app_dir, "build.json")
        status = None
        if os.path.exists(record):
            with open(record) as f:
                status = json.load(f).get("status")
        if status != "pass":
            raise BenchparkError(
                f"No passing NCT build at {app_dir} (status: {status}). "
                f"Build it first: nct/build.py {app} {build}"
            )
        return app_dir

    # Filesystems the NCT I/O tests write to. The experiment name is the same on every
    # node profile, so the PBS job ID keeps concurrent runs apart.
    io_paths = {
        "scratch": "/glade/derecho/scratch/$USER/nct-io/{experiment_name}-$PBS_JOBID",
        "work": "/glade/work/$USER/nct-io/{experiment_name}-$PBS_JOBID",  # GPFS
        # Node-local, but an overlay on Casper compute nodes (no O_DIRECT; IOR there
        # would measure memory), so only mdtest uses it.
        "tmp": "/tmp/nct-io-{experiment_name}-$PBS_JOBID",
    }
