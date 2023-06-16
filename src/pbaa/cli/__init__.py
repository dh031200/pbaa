# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import click

from pbaa import inference, model_init
from pbaa.__about__ import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="pbaa")
@click.option("--src", "-s", help="Path to source image or directory")
@click.option(
    "--prompt", "-p", type=(str, str), multiple=True, help="Space-separated a pair of prompt and target classe. (Multi)"
)
def pbaa(src, prompt):
    model_init()
    inference(src, dict(prompt))
