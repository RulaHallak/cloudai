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
from unittest.mock import Mock

import pytest

from cloudai._core.test import Test
from cloudai._core.test_scenario import TestRun
from cloudai.schema.test_template.nccl_test.slurm_command_gen_strategy import NcclTestSlurmCommandGenStrategy
from cloudai.systems import SlurmSystem
from cloudai.test_definitions.nccl import NCCLCmdArgs, NCCLTestDefinition


class TestNcclTestSlurmCommandGenStrategy:
    @pytest.fixture
    def cmd_gen_strategy(self, slurm_system: SlurmSystem) -> NcclTestSlurmCommandGenStrategy:
        return NcclTestSlurmCommandGenStrategy(slurm_system, {})

    @pytest.mark.parametrize(
        "cmd_args, extra_cmd_args, expected_args",
        [
            (
                NCCLCmdArgs(subtest_name="all_reduce_perf", nthreads=4, ngpus=2),
                {"--max-steps": "100"},
                [
                    "/usr/local/bin/all_reduce_perf",
                    "--nthreads 4",
                    "--ngpus 2",
                    "--max-steps 100",
                ],
            ),
            (
                NCCLCmdArgs(subtest_name="all_reduce_perf", op="sum", datatype="float"),
                {},
                [
                    "/usr/local/bin/all_reduce_perf",
                    "--op sum",
                    "--datatype float",
                ],
            ),
        ],
    )
    def test_generate_test_command(
        self,
        cmd_args: NCCLCmdArgs,
        extra_cmd_args: dict[str, str],
        expected_args: List[str],
        cmd_gen_strategy: NcclTestSlurmCommandGenStrategy,
    ) -> None:
        tdef = NCCLTestDefinition(
            name="nccl",
            description="desc",
            test_template_name="nccl",
            cmd_args=cmd_args,
            extra_cmd_args=extra_cmd_args,
        )
        tr = TestRun(name="name", test=Test(test_definition=tdef, test_template=Mock()), num_nodes=1, nodes=[])
        command = cmd_gen_strategy.generate_test_command(tr)
        for arg in expected_args:
            assert arg in command
