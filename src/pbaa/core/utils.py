# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
from json import dump

import cv2


def save_images(name, det, seg, mask):
    if det is not None:
        cv2.imwrite(f"{name}_det.jpg", det)
    if seg is not None:
        cv2.imwrite(f"{name}_seg.jpg", seg)
    if mask is not None:
        cv2.imwrite(f"{name}_mask.jpg", mask)


def save_json(name, json_data):
    with open(f"{name}.json", "w") as f:
        dump(json_data, f, indent=4, ensure_ascii=False)
