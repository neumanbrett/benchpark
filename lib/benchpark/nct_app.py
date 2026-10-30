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

    def nct_app_dir(self, build):
        """Install dir of an NCT build; checks the build matches the system."""
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
        self.module_libraries = libraries

        build_name = build
        if gpu:
            cuda_arch = getattr(system, "cuda_arch", None)
            if not cuda_arch:
                raise BenchparkError(
                    f"{self.name} build '{build}' needs a GPU node profile"
                )
            build_name = f"{build}-cc{cuda_arch}"

        app_dir = system.nct_build_dir(self.name, self.nct_version, build_name)
        record = os.path.join(app_dir, "build.json")
        status = None
        if os.path.exists(record):
            with open(record) as f:
                status = json.load(f).get("status")
        if status != "pass":
            raise BenchparkError(
                f"No passing NCT build at {app_dir} (status: {status}). "
                f"Build it first: nct/build.py {self.name} {build}"
            )
        return app_dir
