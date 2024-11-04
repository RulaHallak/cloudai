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

from typing import List, cast

from cloudai import TestRun
from cloudai.systems.slurm.strategy import SlurmCommandGenStrategy2
from cloudai.test_definitions.ucc import UCCTestDefinition


class UCCTestSlurmCommandGenStrategy(SlurmCommandGenStrategy2):
    """Command generation strategy for UCC tests on Slurm systems."""

    def tdef(self, tr: TestRun) -> UCCTestDefinition:
        return cast(UCCTestDefinition, tr.test.test_definition)

    def generate_srun_prefix(self, tr: TestRun) -> List[str]:
        srun_command_parts = ["srun", f"--mpi={self.system.mpi}"]
        tdef = self.tdef(tr)
        srun_command_parts.append(f"--container-image={tdef.docker_image.installed_path}")
        if self.system.extra_srun_args:
            srun_command_parts.append(self.system.extra_srun_args)

        return srun_command_parts

    def generate_test_command(self, tr: TestRun) -> List[str]:
        srun_command_parts = ["/opt/hpcx/ucc/bin/ucc_perftest"]

        tdef = self.tdef(tr)
        srun_command_parts.append(f"-c {tdef.cmd_args.collective}")
        srun_command_parts.append(f"-b {tdef.cmd_args.b}")
        srun_command_parts.append(f"-e {tdef.cmd_args.e}")

        # Append fixed string options for memory type and additional flags
        srun_command_parts.append("-m cuda")
        srun_command_parts.append("-F")

        if tr.test.extra_cmd_args:
            srun_command_parts.append(tr.test.extra_cmd_args)

        return srun_command_parts

    def _write_sbatch_script2(self, srun_command: str, tr: TestRun) -> str:
        batch_script_content = self.create_sbatch_directives(tr)

        env_vars: dict[str, str] = {k: str(v) for k, v in self.system.global_env_vars.items()}
        env_vars.update(tr.test.extra_env_vars)
        for key, value in env_vars.items():
            batch_script_content.append(f"export {key}={value}")
        batch_script_content.extend(["", srun_command])

        batch_script_path = tr.output_path / "cloudai_sbatch_script.sh"
        with batch_script_path.open("w") as batch_file:
            batch_file.write("\n".join(batch_script_content))

        return f"sbatch {batch_script_path}"
