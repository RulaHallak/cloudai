# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import List, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy2
from cloudai.test_definitions.nccl import NCCLTestDefinition


class NcclTestSlurmCommandGenStrategy(SlurmCommandGenStrategy2):
    """Command generation strategy for NCCL tests on Slurm systems."""

    def tdef(self, tr: TestRun) -> NCCLTestDefinition:
        return cast(NCCLTestDefinition, tr.test.test_definition)

    def generate_srun_prefix(self, tr: TestRun) -> List[str]:
        srun_command_parts = ["srun", f"--mpi={self.system.mpi}"]
        tdef = self.tdef(tr)
        srun_command_parts.append(f"--container-image={tdef.docker_image.installed_path}")

        env_vars: dict[str, str] = {k: str(v) for k, v in self.system.global_env_vars.items()}
        env_vars.update(tr.test.extra_env_vars)

        container_mounts = ""
        if "NCCL_TOPO_FILE" in env_vars and "DOCKER_NCCL_TOPO_FILE" in env_vars:
            nccl_graph_path = Path(env_vars["NCCL_TOPO_FILE"]).resolve()
            nccl_graph_file = env_vars["DOCKER_NCCL_TOPO_FILE"]
            container_mounts = f"{nccl_graph_path}:{nccl_graph_file}"
        if container_mounts:
            srun_command_parts.append(f"--container-mounts={container_mounts}")

        if self.system.extra_srun_args:
            srun_command_parts.append(self.system.extra_srun_args)

        return srun_command_parts

    def generate_test_command(self, tr: TestRun) -> List[str]:
        cmd_args = self.tdef(tr).cmd_args
        srun_command_parts = [
            f"/usr/local/bin/{cmd_args.subtest_name}",
            f"--nthreads {cmd_args.nthreads}",
            f"--ngpus {cmd_args.ngpus}",
            f"--minbytes {cmd_args.minbytes}",
            f"--maxbytes {cmd_args.maxbytes}",
            f"--stepbytes {cmd_args.stepbytes}",
            f"--op {cmd_args.op}",
            f"--datatype {cmd_args.datatype}",
            f"--root {cmd_args.root}",
            f"--iters {cmd_args.iters}",
            f"--warmup_iters {cmd_args.warmup_iters}",
            f"--agg_iters {cmd_args.agg_iters}",
            f"--average {cmd_args.average}",
            f"--parallel_init {cmd_args.parallel_init}",
            f"--check {cmd_args.check}",
            f"--blocking {cmd_args.blocking}",
            f"--cudagraph {cmd_args.cudagraph}",
        ]

        if tr.test.extra_cmd_args:
            srun_command_parts.append(tr.test.extra_cmd_args)

        return srun_command_parts
