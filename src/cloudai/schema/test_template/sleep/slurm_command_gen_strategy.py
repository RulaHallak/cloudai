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
from cloudai.test_definitions.sleep import SleepTestDefinition


class SleepSlurmCommandGenStrategy(SlurmCommandGenStrategy2):
    """Command generation strategy for Sleep on Slurm systems."""

    def tdef(self, tr: TestRun) -> SleepTestDefinition:
        return cast(SleepTestDefinition, tr.test.test_definition)

    def generate_srun_prefix(self, tr: TestRun) -> List[str]:
        srun_command_parts = ["srun", f"--mpi={self.system.mpi}"]
        if self.system.extra_srun_args:
            srun_command_parts.append(self.system.extra_srun_args)

        return srun_command_parts

    def generate_test_command(self, tr: TestRun) -> List[str]:
        srun_command_parts = ["sleep", str(self.tdef(tr).cmd_args.seconds)]

        if tr.test.extra_cmd_args:
            srun_command_parts.append(tr.test.extra_cmd_args)

        return srun_command_parts
