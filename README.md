# pbaa : Prompt-Based Automatic Annotation

[![PyPI - Version](https://img.shields.io/pypi/v/pbaa.svg)](https://pypi.org/project/pbaa)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pbaa.svg)](https://pypi.org/project/pbaa)

Easy inference implementation of [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) for
**Prompt-based automatic annotation**

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#Usage)
- [Demo](#Demo)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

### Docker (Recommend)

```console
git clone https://github.com/dh031200/pbaa.git
docker build docker -t pbaa:latest
docker run --gpus all -it --ipc=host -v `pwd`:/workspace pbaa:latest
```

### Without docker

The code requires `python>=3.8`, `CUDA==11.7`.

```console
pip install pbaa
```

## Usage

### Options

```console
Usage: pbaa [OPTIONS]

Options:
  --version                    Show the version and exit.
  -s, --src TEXT               Source image or directory path  [required]
  -p, --prompt <TEXT TEXT>...  Space-separated a pair of prompt and target
                               classe. (Multi)  [required]
  -b, --box_threshold FLOAT    Threshold for Object Detection (default: 0.25)
  -n, --nms_threshold FLOAT    Threshold for NMS (default: 0.8)
  -o, --output_dir TEXT        Path to result data (default: '.')
  -h, --help                   Show this message and exit.
```

### CLI

```console
# pbaa -s <Source> -p <prompt> <class> -p <prompt> <class> ...

pbaa -s source_image.jpg -p "black dog" dog
pbaa -s source_image.jpg -p "black dog" dog -p "white cat" cat
```

### Python

```python
from pbaa import model_init, inference

model_init()
# inference(<Source path>, <prompt:class>, box_threshold=0.25, nms_threshold=0.8, output_dir=".")
inference("path/to/source_image.jpg", {"black dog": "dog", "white cat": "cat"})
```

## Demo

```console
# Source : assets/demo9.jpg
# prompts : {"plant" : "plant", "picture" : "picture", "dog": "dog", "lamp" : "lamp", "carpet" : "carpet", "sofa" : "sofa"}

pbaa -s assets/demo9.jpg -p plant plant -p picture picture -p dog dog -p lamp lamp -p carpet carpet -p sofa sofa
```

|                                     Origin                                      |                                       Detection                                        |                                       Segmentation                                        |
|:-------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------:|
| ![Before](https://github.com/dh031200/pbaa/blob/main/assets/demo9.jpg?raw=true) | ![detection](https://github.com/dh031200/pbaa/blob/main/assets/demo9_det.jpg?raw=true) | ![segmentation](https://github.com/dh031200/pbaa/blob/main/assets/demo9_seg.jpg?raw=true) |

### Result data

[demo9.json](https://github.com/dh031200/pbaa/blob/main/assets/demo9.json)<br>

```console
json structure

index
  ├ cls : class name
  ├ conf : confidence score
  ├ box : bounding box coordinates
  └ poly : polygon coordinates
```

## License

`pbaa` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.

## Acknowledgements

Grounded-Segment-Anything : [https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)<br>
Grounding DINO : [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)<br>
Segment-anything : [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)<br>
