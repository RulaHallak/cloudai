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

from typing import List

import pytest

from cloudai._core.test_scenario import TestRun
from cloudai.schema.test_template.ucc_test.slurm_command_gen_strategy import UCCTestSlurmCommandGenStrategy
from cloudai.systems import SlurmSystem
from cloudai.test_definitions.ucc import UCCCmdArgs
from tests.conftest import create_autospec_dataclass


class TestUCCTestSlurmCommandGenStrategy:
    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> UCCTestSlurmCommandGenStrategy:
        return UCCTestSlurmCommandGenStrategy(slurm_system, {})

    @pytest.mark.parametrize(
        "cmd_args, extra_cmd_args, expected_command",
        [
            (
                UCCCmdArgs(collective="allgather", b=8, e="256M"),
                "--max-steps 100",
                [
                    "/opt/hpcx/ucc/bin/ucc_perftest",
                    "-c allgather",
                    "-b 8",
                    "-e 256M",
                    "-m cuda",
                    "-F",
                    "--max-steps 100",
                ],
            ),
            (
                UCCCmdArgs(collective="allreduce", b=4),
                "",
                [
                    "/opt/hpcx/ucc/bin/ucc_perftest",
                    "-c allreduce",
                    "-b 4",
                    "-e 8M",
                    "-m cuda",
                    "-F",
                ],
            ),
        ],
    )
    def test_generate_test_command(
        self,
        cmd_args: UCCCmdArgs,
        extra_cmd_args: str,
        expected_command: List[str],
        cmd_gen_strategy: UCCTestSlurmCommandGenStrategy,
    ) -> None:
        tr = create_autospec_dataclass(TestRun)
        tr.test.test_definition.cmd_args = cmd_args
        tr.test.extra_cmd_args = extra_cmd_args
        command = cmd_gen_strategy.generate_test_command(tr)
        assert command == expected_command
