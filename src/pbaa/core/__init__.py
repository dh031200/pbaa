# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
from check import init, check
from dependencies import dependencies

from grounded_sam import model_init, run

init()
check()

__all__ = "model_init", "run"
