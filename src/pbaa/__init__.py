# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
from loguru import logger

from pbaa.core import PBAA, app


def test_success():
    logger.info("Test Successed")


__all__ = "PBAA", "app", "test_success"
