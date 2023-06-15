# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import click

from pbaa import run
from pbaa.__about__ import __version__


@click.group(context_settings={"help_option_names": ["-h", "--help"]}, invoke_without_command=True)
@click.version_option(version=__version__, prog_name="pbaa")
@click.option("--hq", is_flag=True, help="whether segmentation with sam-hq")
def pbaa(hq):
    click.echo("Hello world!")
    run()
