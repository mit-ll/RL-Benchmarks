<p align="center">
  <a href="https://github.com/destin-v/">
    <img src="https://raw.githubusercontent.com/destin-v/destin-v/main/docs/pics/logo.gif" alt="drawing" width="500"/>
  </a>
</p>

# 📒 Description
<p align="center">
  <img src="docs/pics/program_logo.png" alt="drawing" width="250"/>
</p>

<p align="center">
  <a href="https://devguide.python.org/versions/">              <img alt="" src="https://img.shields.io/badge/python-3.10|3.11-blue?logo=python&logoColor=white"></a>
  <a href="https://docs.github.com/en/actions/quickstart">      <img alt="" src="https://img.shields.io/badge/CI-github-blue?logo=github&logoColor=white"></a>
  <a href="https://black.readthedocs.io/en/stable/index.html">  <img alt="" src="https://img.shields.io/badge/code%20style-black-blue"></a>
</p>

<p align="center">
  <a href="https://github.com/mit-ll/RL-Benchmarks/actions/workflows/pre-commit.yml">  <img alt="pre-commit" src="https://github.com/mit-ll/RL-Benchmarks/actions/workflows/pre-commit.yml/badge.svg"></a>
  <a href="https://github.com/mit-ll/RL-Benchmarks/actions/workflows/pytest.yml">      <img alt="pytest"     src="https://github.com/mit-ll/RL-Benchmarks/actions/workflows/pytest.yml/badge.svg"></a>

</p>

This repo is designed to benchmark neural networks against various classic control and Mujoco environments. A few default networks have been included:  Multilayer Perception (MLP), Long Short Term Memory (LSTM), and Neural Circuit Policy (NCP) networks.

# 🐳 Build Container

## Definition File
This repo includes an Apptainer definition file located at **containers/rl_benchmarks.def**.  You will need to log into a computer with `root` access to build this file.  Execute the following:

```console
sudo su -

cd containers
singularity build miniconda.sif miniconda.def
singularity build rl_benchmarks.sif rl_benchmarks.def
```

From there you can shell into the container to test the code:

```console
singularity shell --tmp-sandbox --writable --nv rl_benchmarks.sif
```


## Syndeo
The container should be used with [**Syndeo**](https://github.com/mit-ll/Syndeo) to launch parallel jobs on SLURM.  A template for launching the parallel job using Syndeo can be found at **containers/syndeo.sh**.  Use that as a launching pad to parallelize jobs on SLURM.


# 🛠️ Installing (Bare Metal)
```console
conda create -n rl_benchmarks python=3.10
pip install poetry
poetry install --with=dev
```

You will need a custom installation of `Mujoco 2.1.0` found [**here**](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco).

Mujoco requires some environmental variables to be set which can be added to your `.bashrc`:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/domi/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

If you need a specific version of CUDA or CUDNN perform the proper installation from the PyTorch [**documentation**](https://pytorch.org/get-started/previous-versions/).

# 🧪 Testing
To evaluate whether you have setup the install properly run all the tests after installation.

    pytest

This will tell you exactly where the failures are.

# 🏝️ Environments


|                          **CartPole**                          |                                 **Pendulum**                                  |                              **Acrobot**                               |
| :------------------------------------------------------------: | :---------------------------------------------------------------------------: | :--------------------------------------------------------------------: |
| <img src="https://gymnasium.farama.org/_images/cart_pole.gif"> |         <img src="https://gymnasium.farama.org/_images/pendulum.gif">         |      <img src="https://gymnasium.farama.org/_images/acrobot.gif">      |
|                            **Ant**                             |                               **Half Cheetah**                                |                          **Humanoid Standup**                          |
|    <img src="https://gymnasium.farama.org/_images/ant.gif">    |       <img src="https://gymnasium.farama.org/_images/half_cheetah.gif">       | <img src="https://gymnasium.farama.org/_images/humanoid_standup.gif">  |
|                          **Humanoid**                          |                         **Inverted Double Pendulum**                          |                         **Inverted Pendulum**                          |
| <img src="https://gymnasium.farama.org/_images/humanoid.gif">  | <img src="https://gymnasium.farama.org/_images/inverted_double_pendulum.gif"> | <img src="https://gymnasium.farama.org/_images/inverted_pendulum.gif"> |
|                           **Pusher**                           |                                  **Reacher**                                  |                              **Swimmer**                               |
|  <img src="https://gymnasium.farama.org/_images/pusher.gif">   |         <img src="https://gymnasium.farama.org/_images/reacher.gif">          |      <img src="https://gymnasium.farama.org/_images/swimmer.gif">      |

Classic Control Environments
* CartPole-v1
* Pendulum-v1
* Acrobot-v1

Mujoco Environments
* Ant-v4
* HalfCheetah-v4
* HumanoidStandup-v4
* Humanoid-v4
* InvertedDoublePendulum-v4
* InvertedPendulum-v4
* Reacher-v4
* Swimmer-v4
* Pusher-v4

> [!NOTE]
> “All of these environments are **stochastic in terms of their initial state, with a Gaussian noise added** to a fixed initial state in order to add stochasticity.”

# 🚊 Running Training

To run the training pipeline first select the type of network you want to evaluate.  Three different network types have been provided for you:

    from src.models import CustomTorchModelCfc
    from src.models import CustomTorchModelMlp
    from src.models import CustomTorchModelLtc

To register the network you want to test set the following within `main.py`:

    ModelCatalog.register_custom_model("my_torch_model", CustomTorchModelCfc)

To run the experiment execute:

    python main.py

Outputs will be saved to the `save/` folder.  You can view them in Tensorboard using:

    tensorboard --logdir=./save


If you are running on SLURM there is a script provided called `grid_batch.sh`.
    sbatch grid_batch.sh

# ⬛️ Tmux
If using tmux to do training here are some useful commands:

    tmux new -s <session> # create a new session
    tmux ls # list all sessions
    tmux attach -t <session> # attach to session

Within the tmux window here are some useful commands:

    control-b d # break from session
    control-b [ # enter scrolling mode (q to quit)


# 🔧 Troubleshooting
>[!CAUTION]
The `gymnasium` package is very much in beta.  There are bugs and problems with setting up the Mujoco environments.  The current `gymnasium==0.29.1` version doesn't support XML files even though it is documented on the [**website**](https://gymnasium.farama.org/environments/mujoco/swimmer/).  For the `Swimmer` environment it states:


> v3 and v4 take gymnasium.make kwargs such as **xml_file**, ctrl_cost_weight, reset_noise_scale, etc.

The `xml_file` argument is not accepted and will throw an error.  This means that in order to insert an XML file you need to install it into the `site-packages` of the virtual environment containing `mujoco`.

This repo has code that will replace the existing XML files for Mujoco, but the user should be aware.  The developers are aware of the [**issue**](https://github.com/Farama-Foundation/Gymnasium/pull/746), but they don't plan to fix it until `v1.0.0` release.


# 📔 Citations
The author acknowledges the MIT Lincoln Laboratory Supercomputing Center for providing (HPC, database, consultation) resources that have contributed to the research results reported within this paper/report.

@misc{towers_gymnasium_2023,
        title = {Gymnasium},
        url = {https://zenodo.org/record/8127025},
        abstract = {An API standard for single-agent reinforcement learning environments, with popular reference environments and related utilities (formerly Gym)},
        urldate = {2023-07-08},
        publisher = {Zenodo},
        author = {Towers, Mark and Terry, Jordan K. and Kwiatkowski, Ariel and Balis, John U. and Cola, Gianluca de and Deleu, Tristan and Goulão, Manuel and Kallinteris, Andreas and KG, Arjun and Krimmel, Markus and Perez-Vicente, Rodrigo and Pierré, Andrea and Schulhoff, Sander and Tai, Jun Jet and Shen, Andrew Tan Jin and Younis, Omar G.},
        month = mar,
        year = {2023},
        doi = {10.5281/zenodo.8127026},
}

@inproceedings{reuther2018interactive,
title={Interactive supercomputing on 40,000 cores for machine learning and data analysis},
author={Reuther, Albert and Kepner, Jeremy and Byun, Chansup and Samsi, Siddharth and Arcand, William and Bestor, David and Bergeron, Bill and Gadepally, Vijay and Houle, Michael and Hubbell, Matthew and Jones, Michael and Klein, Anna and Milechin, Lauren and Mullen, Julia and Prout, Andrew and Rosa, Antonio and Yee, Charles and Michaleas, Peter},
booktitle={2018 IEEE High Performance extreme Computing Conference (HPEC)},
pages={1--6},
year={2018},
organization={IEEE}
}

# ♖ Distribution
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.

© 2024 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
