# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
from loguru import logger

from pbaa.core import app, inference, model_init


def test_success():
    logger.info("Test Successed")


__all__ = "model_init", "inference", "app", "test_success"
