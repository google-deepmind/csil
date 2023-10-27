# Coherent Soft Imitation Learning

This repository contains an implementation of Coherent Soft Imitation Learning,
published at NeurIPS 2023. It also contains implementations of two other soft
imitation learning algorithms: Inverse Soft Q-Learning (IQ-Learn) and Proximal
Point Imitation Learning (P2IL).

## Content

```
.
├── scripts
|   ├── helpers.py                   - Helper methods: dataset iterators and environment creation.
|   ├── run_csil.py                  - Example of running CSIL on continuous control tasks.
|   ├── run_iqlearn.py               - Example of running IQ-Learn on continuous control tasks.
|   ├── run_ppil.py                  - Example of running P2IL on continuous control tasks.
|   └── soft_policy_iteration.ipynb  - Evaluation of SIL algorithms in a discrete tabular setting.
|
├── sil
|   ├── builder.py                   - Creates learner, actor, and policy.
|   ├── config.py                    - Algorithm-specific configurations.
|   ├── evaluator.py                 - Creates evaluators and video recorders.
|   ├── learning.py                  - Implements the SIL learner.
|   ├── networks.py                  - Defines networks used by SIL algorithms.
|   └── pretraining.py               - Implements pre-training for policy and critic.
|
├── README.md
└── requirements.txt                 - Dependencies.
```

## Installation

First clone this code repository into a local directory:
```bash
git clone https://github.com/google-deepmind/csil.git
cd csil
```

We recommend installing required dependencies inside a
[conda environment](https://www.anaconda.com/). To do this, first install
[Anaconda](https://www.anaconda.com/download#downloads) and then create and
activate the conda environment:
```bash
conda create --name csil python=3.9
conda activate csil
```

Then install `pip` and use it to install all the dependencies:
```bash
conda install pip
pip install -r requirements.txt
```

MuJoCo must also be installed, in order to load the environments. Please follow
the instructions [here](https://github.com/openai/mujoco-py#install-mujoco) to
install the MuJoCo binary and place it in a directory where `mujoco-py` can find
it.

## Troubleshooting

If you get the error
```
Command conda not found
```
then you need to add the folder where Anaconda is installed to your `PATH`
variable:
```bash
export PATH=/path/to/anaconda/bin:$PATH
```

If you get the error
```
ImportError: libpython3.9.so.1.0: cannot open shared object file: No such file or directory
```
first activate the conda environment and then add it to the `LD_LIBRARY_PATH`:
```bash
conda activate csil
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$CONDA_PREFIX/lib"
```

If you get the error
```
cannot find -lGL: No such file or directory
```
then install libGL with:
```
sudo apt install libgl-dev
```


If you get the error
```
fatal error: GL/glew.h: No such file or directory
```
then you need to install the following in your conda environment and update the
`CPATH`:
```bash
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
export CPATH=$CONDA_PREFIX/include
```

If you get the error
```
ImportError: libgmpxx.so.4: cannot open shared object file: No such file or directory
```
then you need to install the following in your conda environment and update the
`CPATH`:
```bash
conda install -c conda-forge gmp
export CPATH=$CONDA_PREFIX/include
```

## Usage

Before running any code, first activate the conda environment and set the
`PYTHONPATH`:
```bash
conda activate csil
export PYTHONPATH=$(pwd)/..
```

To run CSIL with default settings:
```bash
python scripts/run_csil.py
```
This runs the online version of CSIL on HalfCheetah-v2.

The experiment configurations for each algorithm (CSIL, IQ-Learn, and PPIL), can
be adjusted via the flags defined at the start of `run_*.py`.

The available tasks (specified with the `--env_name` flag) are:
```
HalfCheetah-v2
Ant-v2
Walker2d-v2
Hopper-v2
Humanoid-v2
door-v0         # Adroit hand
hammer-v0       # Adroit hand
pen-v0          # Adroit hand
```

The default setting is online soft imitation learning. To run the offline
version on the Adroit door task, for example:
```bash
python scripts/run_{algo_name}.py --offline=True --env_name=door-v0
```
replacing `{algo_name}` with either csil, iqlearn, or ppil.

We have also included a Colab [here](https://colab.research.google.com/github/deepmind/csil/blob/master/scripts/soft_policy_iteration.ipynb) that reproduces
the discrete grid world experiments shown in the paper, for a range of soft
imitation learning and IRL algorithms.


## Citing this work

```bibtex
@inproceedings{watson2023csil,
  author       = {Joe Watson and
                  Sandy H. Huang and
                  Nicolas Heess},
  title        = {Coherent Soft Imitation Learning},
  booktitle    = {Advances in Neural Information Processing Systems},
  year         = {2023}
}
```

## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
