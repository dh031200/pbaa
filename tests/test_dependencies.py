# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
from loguru import logger

try:
    import pbaa
    pbaa.test_success()
except Exception as E:
    logger.error(E)
    raise ImportError from E
