# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import sys

import click

from pbaa import PBAA, app
from pbaa.__about__ import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="pbaa")
@click.option("--src", "-s", type=str, help="Source image or directory path")
@click.option(
    "--prompt",
    "-p",
    type=(str, str),
    multiple=True,
    help="Space-separated a pair of prompt and target classe. (Multi)",
)
@click.option("--box_threshold", "-b", type=float, default=0.25, help="Threshold for Object Detection (default: 0.25)")
@click.option("--nms_threshold", "-n", type=float, default=0.8, help="Threshold for NMS (default: 0.8)")
@click.option("--output_dir", "-o", type=str, default="outputs", help="Path to result data (default: 'outputs')")
@click.option("--gradio", "-g", type=bool, default=False, is_flag=True, help="Launch gradio app")
def pbaa(src, prompt, box_threshold, nms_threshold, output_dir, gradio):
    annotator = PBAA()
    if gradio:
        click.echo("Launch gradio app")
        app(annotator.inference)
    else:
        is_failed = False
        if not src:
            click.echo("Error: Missing option '--src' / '-s'.")
            is_failed = True
        if not prompt:
            click.echo("Error: Missing option '--prompt' / '-p'.")
            is_failed = True
        if is_failed:
            sys.exit(-1)
        _prompt = {i.lower(): v for i, v in prompt}
        annotator.inference(
            src=src,
            _prompt=_prompt,
            annot_format=None,
            box_threshold=box_threshold,
            nms_threshold=nms_threshold,
            save=True,
            output_dir=output_dir,
        )
