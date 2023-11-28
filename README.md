# Coherent Soft Imitation Learning

[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2305.16498-B31B1B.svg)](https://arxiv.org/abs/2305.16498)
[![Python 3.7+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-376/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align=center>
<img src="/assets/csil.gif?raw=true" width="750px">
</p>


This repository contains an implementation of [coherent soft imitation learning (CSIL)](https://arxiv.org/abs/2305.16498),
published at [NeurIPS 2023](https://openreview.net/forum?id=kCCD8d2aEu).

We also provide implementations of other 'soft'
imitation learning (SIL) algorithms: [Inverse soft Q-learning (IQ-Learn)](https://arxiv.org/abs/2106.12142) and [proximal
point imitation learning (PPIL)](https://arxiv.org/abs/2209.10968).

## Content
The implementation is built on top of [Acme](https://github.com/google-deepmind/acme) and follows their agent structure.
```
.
├── run_csil.py                      - Example of running CSIL on continuous control tasks.
├── run_iqlearn.py                   - Example of running IQ-Learn on continuous control tasks.
├── run_ppil.py                      - Example of running PPIL on continuous control tasks.
├── soft_policy_iteration.ipynb      - Evaluation of SIL algorithms in a discrete tabular setting.
├── helpers.py                       - Utilities such as dataset iterators and environment creation.
├── experiment_logger.py             - Implements a Weights & Biases logger within the Acme framework.
|
├── sil
|   ├── config.py                    - Algorithm-specific configurations for soft imitation learning (SIL).
|   ├── builder.py                   - Creates the learner, actor, and policy.
|   ├── evaluator.py                 - Creates the evaluators and video recorders.
|   ├── learning.py                  - Implements the imitation learners.
|   ├── networks.py                  - Defines the policy, reward and critic networks.
|   └── pretraining.py               - Implements pre-training for policy and critic.
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

We have also included a Colab [here](https://colab.research.google.com/github/google-deepmind/csil/blob/main/soft_policy_iteration.ipynb) that reproduces
the discrete grid world experiments shown in the paper, for a range of imitation learning algorithms.

We highly encourage the use of accelerators (i.e. GPUs, TPUs) for CSIL. As CSIL requires a larger policy architecture, it has a slow wallclock time if run only on CPUs.

For a reproduction of the paper's experiment, [see this Weights & Biases project](https://wandb.ai/jmw125/csil/workspace).

The additional imitiation learning baselines shown in the paper [are available in Acme](https://github.com/google-deepmind/acme/tree/master/examples/baselines/imitation). 

### Open issues

[Distribued Acme experiments currently do not finish cleanly, so they appear as 'Crashed' on W&B when they finish successfully.](https://github.com/google-deepmind/acme/issues/312#issue-1990249288)

The robomimic experiments are currently not open-sourced. 

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
CSIL is written in JAX, so first install the correct version of JAX for your system by [following the installation instructions](https://jax.readthedocs.io/en/latest/installation.html).
Acme requires `jax 0.4.3` and will install that version. This may need to be uninstalled for a CUDA-based JAX installation, e.g.
```bash
pip install jax==0.4.7 https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.7+cuda12.cudnn88-cp39-cp39-manylinux2014_x86_64.whl
```

MuJoCo must also be installed, in order to load the environments. Please follow
the instructions [here](https://github.com/openai/mujoco-py#install-mujoco) to
install the MuJoCo binary and place it in a directory where `mujoco-py` can find
it.
This installation uses `mujoco200`, `gym < 0.24.0` and `mujoco-py 2.0.2.5` for compatibility reasons.

Then install `pip` and use it to install all the dependencies:
```bash
pip install -r requirements.txt
```
To verify the installation, run
```bash
python -c "import jax.numpy as jnp; print(jnp.ones((1,)).device); import acme; import mujoco_py; import gym; print(gym.make('HalfCheetah-v2').reset())"
```
If this fails, follow the guidance below. 

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
If you get the error
```commandline
ImportError: ../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1)
```
try
```commandline
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```
according to [this advice](https://stackoverflow.com/a/73708979).

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
