# Fraunhofer Neural Network Encoder/Decoder (NNCodec)

The Fraunhofer Neural Network Encoder/Decoder Software (NNCodec) is an efficient implementation of NNC (Neural Network Coding / ISO/IEC 15938-17 or MPEG-7 part 17), which is the first international standard on compression of neural networks.
NNCodec provides an encoder and decoder with the following main features:
- Standard compliant implementation of the core compression technologies including, e.g., DeepCABAC
- Easy-to-use user interface
- Built-in support for common frameworks (e.g. TensorFlow and PyTorch)
- Built-in ImageNet-support for data driven compression tools
- Extensibility of data driven compression tools for arbitrary datasets

## Highlights
### ICML 2023 Neural Compression Workshop
Our paper titled **"NNCodec: An Open Source Software Implementation of the Neural Network Coding 
ISO/IEC Standard"** was awarded a Spotlight Paper at the ICML 2023 Neural Compression Workshop.

**TL;DR** -  The paper presents NNCodec, analyses its coding tools with respect to the principles of information theory and gives comparative results for a broad range of neural network architectures. 

The code for reproducing the experimental results of the paper and a software demo will be available 
from July 29.

[![Conference](https://img.shields.io/badge/ICML-Paper-blue)](https://openreview.net/forum?id=5VgMDKUgX0)
[![Conference](https://img.shields.io/badge/ICML-Code-red)]()

## Information

Refer to the [Wiki-Page](https://github.com/fraunhoferhhi/nncodec/wiki) for more information:
- [Installation-Guide](https://github.com/fraunhoferhhi/nncodec/wiki/Installation-Guide)
- [Usage](https://github.com/fraunhoferhhi/nncodec/wiki/Usage)
- [Compression-Performance](https://github.com/fraunhoferhhi/nncodec/wiki/Compression-Performance)
- [License](https://github.com/fraunhoferhhi/nncodec/wiki/License)
- [References](https://github.com/fraunhoferhhi/nncodec/wiki/References)
- [Version history](https://github.com/fraunhoferhhi/nncodec/wiki/Changelog)

## Installation

The software provides python packages which can be installed using pip. However, core technologies are implemented using C++, which requires a C++ compiler for the installation process.

The software has been tested on different target platforms (Windows, Linux and MacOS).

### Requirements

- python >= 3.6 (recommended versions 3.6, 3.7 and 3.8) with working pip
- **Windows:** Microsoft Visual Studio 2015 Update 3 or later

### Package installation

From the root of the cloned repository, issue

```
pip install wheel
pip install -r requirements.txt
pip install .
```

and for CUDA11 support

```
pip install wheel
pip install -r requirements_cu11.txt
pip install .
```

for installation.

**Information**: On Linux/Mac the scripts `create_env.sh` and `create_env_cu11.sh` (for Cuda 11 support) set up a virtual python environment "env" and install all required packages and the software itself, automatically. For activating this environment, issue:

```
source env/bin/activate
```

**Note**: For further information on how to set up a virtual python environment (also on **Windows**) refer to https://docs.python.org/3/library/venv.html .

When successfully installed, the software outputs the line : "Successfully installed NNC-0.2.2"

### Importing the main module

After installation the software can be used by importing the main python module 'nnc':
```python
import nnc
```

## References and Publications
If you use NNCodec in your work, please cite:
```
@inproceedings{becking2023nncodec,
title={{NNC}odec: An Open Source Software Implementation of the Neural Network Coding {ISO}/{IEC} Standard},
author={Daniel Becking and Paul Haase and Heiner Kirchhoffer and Karsten M{\"u}ller and Wojciech Samek},
booktitle={ICML 2023 Workshop Neural Compression: From Information Theory to Applications},
year={2023},
url={https://openreview.net/forum?id=5VgMDKUgX0}
}
```
### Publications
- H. Kirchhoffer, et al. **"Overview of the Neural Network Compression and Representation (NNR) Standard"**, IEEE Transactions on Circuits and Systems for Video Technology, pp. 1-14, July 2021, doi: 10.1109/TCSVT.2021.3095970, Open Access
- P. Haase et al., **"Encoder Optimizations For The NNR Standard On Neural Network Compression"**, 2021 IEEE International Conference on Image Processing (ICIP), 2021, pp. 3522-3526, doi: 10.1109/ICIP42928.2021.9506655.
- K. Müller et al., **"Ein internationaler KI-Standard zur Kompression Neuronaler Netze"**, FKT- Fachzeitschrift für Fernsehen, Film und Elektronische Medien, pp. 33-36, September 2021
- S. Wiedemann et al., **"DeepCABAC: A universal compression algorithm for deep neural networks"**, in IEEE Journal of Selected Topics in Signal Processing, doi: 10.1109/JSTSP.2020.2969554.

## Contributing

Everyone is invited to contribute. To do so:
- Fork the current-most state of the master branch
- Implement features or changes
- Add your name to [AUTHORS.md](./AUTHORS.md)
- Create a pull-request to the upstream repository

## License

Please see [LICENSE.txt](./LICENSE.txt) file for the terms of the use of the contents of this repository.

For more information and bug reports, please contact: nncodec@hhi.fraunhofer.de

**Copyright (c) 2019-2023, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.**

**All rights reserved.**
