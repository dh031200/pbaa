# pbaa : Prompt-based automatic annotation

[![PyPI - Version](https://img.shields.io/pypi/v/pbaa.svg)](https://pypi.org/project/pbaa)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pbaa.svg)](https://pypi.org/project/pbaa)

Easy inference implementation of [Grounded-Sam](https://github.com/IDEA-Research/Grounded-Segment-Anything) for
**Prompt-based automatic annotation**

-----

**Table of Contents**

- [Installation](#installation)
- [Usage](#Usage)
- [License](#license)

## Grounded-SAM

Most of code based on [Grounded-Sam](https://github.com/IDEA-Research/Grounded-Segment-Anything)

## Installation

```console
pip install pbaa
```

## Usage

### Option

* `--source`, `-s` : Source image or ~~directory~~ path. (processing)
    * --source \<Source image>
    * --source source_image.jpg
* `--prompt`, `-p` : Space-separated a pair of prompt and target classe. (Multi)
  * --prompt \"\<Prompt>" \<Class>
  * --prompt "black dog" dog

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
inference("path/to/source_image.jpg", {"black dog": "dog", "white cat": "cat"})
```

## License

`pbaa` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.

## Acknowledgements
Grounded-Segment-Anything : [https://github.com/IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
Grounding DINO : [https://github.com/IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
Segment-anything : [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
