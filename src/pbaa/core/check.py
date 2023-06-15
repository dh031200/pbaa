# SPDX-FileCopyrightText: 2023-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: Apache-2.0
import importlib
import os
import platform
import subprocess
import sys
from collections import defaultdict

from loguru import logger

from pbaa.__about__ import PYTHON_VERSION_MAJOR, PYTHON_VERSION_MINOR


def init():
    sys.path.append(f"{os.getcwd()}/GroundingDINO")
    sys.path.append(f"{os.getcwd()}/segment-anything")


def check_cuda():
    import torch

    return torch.cuda.is_available()


def get_dependencies():
    if __name__ == "__main__":
        from dependencies import dependencies
    else:
        from pbaa.core.dependencies import dependencies

    return dependencies.split()


def python_version_check():
    major, minor, patch = map(int, platform.python_version().split("."))
    logger.info(f"Python Version : {major}.{minor}")
    return all([major == PYTHON_VERSION_MAJOR, minor == PYTHON_VERSION_MINOR])


def get_installed_packages():
    packages = defaultdict(str)
    for p in subprocess.check_output(["pip", "list"]).decode("utf-8").splitlines()[2:]:
        n, v = p.split()[:2]
        packages[n] = v
    return packages


def package_rename(_name):
    if _name == "opencv":
        _name = "opencv-python"
    elif _name == "segment_anything":
        _name = "segment-anything"

    return _name


def dependency_check(_dependency):
    checked = False
    try:
        importlib.import_module(_dependency)
        checked = True
    except ImportError:
        logger.warning(f"{_dependency} is not installed.")

    return checked


def install_dependency(_dependency, _version):
    s = f"=={_version}" if _version else ""
    logger.info(subprocess.check_output(f"pip install --no-cache {_dependency}{s}", shell=True).decode())


def install_torch_arm():
    logger.info(subprocess.check_output("sudo apt-get -y update", shell=True).decode())
    logger.info(
        subprocess.check_output(
            "sudo apt-get -y install "
            "autoconf bc build-essential "
            "g++-8 gcc-8 clang-8 lld-8 gettext-base gfortran-8 "
            "iputils-ping libbz2-dev libc++-dev libcgal-dev libffi-dev "
            "libfreetype6-dev libhdf5-dev libjpeg-dev liblzma-dev "
            "libncurses5-dev libncursesw5-dev libpng-dev libreadline-dev "
            "libssl-dev libsqlite3-dev libxml2-dev libxslt-dev locales "
            "moreutils openssl python-openssl rsync scons python3-pip libopenblas-dev",
            shell=True,
        ).decode()
    )
    logger.info(
        subprocess.check_output('export "LD_LIBRARY_PATH=/usr/lib/llvm-8/lib:$LD_LIBRARY_PATH"', shell=True).decode()
    )
    logger.info(subprocess.check_output("pip install --upgrade protobuf", shell=True).decode())
    logger.info(
        subprocess.check_output(
            "pip install --no-cache https://developer.download.nvidia.com/"
            "compute/redist/jp/v51/pytorch/torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl",
            shell=True,
        ).decode()
    )


def install_torchvision_arm():
    logger.info(subprocess.check_output("sudo apt-get -y update", shell=True).decode())
    logger.info(
        subprocess.check_output(
            "sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev",
            shell=True,
        ).decode()
    )
    logger.info(
        subprocess.check_output(
            "git clone --branch release/0.15 https://github.com/pytorch/vision torchvision", shell=True
        ).decode()
    )
    logger.info(
        subprocess.check_output(
            "cd torchvision && export BUILD_VERSION=0.15.0 && python setup.py install --user", shell=True
        ).decode()
    )


def install_torchvision_linux_cuda():
    logger.info(
        subprocess.check_output(
            "pip install --no-cache torchvision",
            shell=True,
        ).decode()
    )


def install_torchvision_linux_cpu():
    logger.info(subprocess.check_output("pip uninstall -y torch", shell=True).decode())
    logger.info(
        subprocess.check_output(
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
            shell=True,
        ).decode()
    )


def install_from_git(name):
    if name == "groundingdino":
        logger.info(
            subprocess.check_output(
                "pip install git+https://github.com/IDEA-Research/GroundingDINO", shell=True
            ).decode()
        )
    elif name == "segment-anything":
        logger.info(
            subprocess.check_output(
                "pip install git+https://github.com/facebookresearch/segment-anything", shell=True
            ).decode()
        )
    elif name == "pycocotools":
        logger.info(
            subprocess.check_output(
                'pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"',
                shell=True,
            ).decode()
        )
    else:
        raise ModuleNotFoundError


def check():
    is_mac_os, is_linux, is_windows = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])
    is_arm = platform.processor() == "aarch64"
    _os = None
    if is_windows:
        _os = "Windows"
    elif is_linux:
        _os = "Linux"
    elif is_mac_os:
        _os = "MacOS"
    else:
        raise OSError

    logger.info(f"Running on {_os}")

    installed_packages = get_installed_packages()
    dependencies = get_dependencies()

    for dependency in dependencies:
        parsed = [i for i in dependency.split("=") if i]
        if len(parsed) == 1:
            name, version = parsed[0], None
        else:
            name, version = parsed

        name = package_rename(name)
        if name == "python":
            python_version_check()
        elif name == "pip":
            pass
        else:
            if not installed_packages[name]:
                if name == "torch" and is_arm:
                    install_torch_arm()
                elif name == "torchvision":
                    if is_arm:
                        install_torchvision_arm()
                    elif is_linux:
                        if check_cuda():
                            install_torchvision_linux_cuda()
                        else:
                            install_torchvision_linux_cpu()
                    elif is_windows:
                        install_dependency(name, version)
                elif name in ["groundingdino", "segment-anything"]:
                    install_from_git(name)
                # elif name == 'pycocotools' and is_windows:
                #     install_from_git(name)
                else:
                    install_dependency(name, version)
                installed_packages = get_installed_packages()
            else:
                logger.info(f"{name} : {installed_packages[name]}")

            if not installed_packages[name]:
                raise ModuleNotFoundError


if __name__ == "__main__":
    check()
