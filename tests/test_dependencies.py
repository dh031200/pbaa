# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
from loguru import logger

try:
    import pbaa
except Exception:
    logger.error(pbaa)
    raise ImportError
