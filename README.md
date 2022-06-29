# Udacity Deep Reinforcement Learning course - Project 1: Continuous control

![my_reacher_demo](my_reacher_demo.gif)

This repository contains my solution for the second project of Udacity's course on Reinforcement Learning. The scenario's goal is to teach a creature with four legs to walk forward without falling.
## Contents
This repo contains:
* `setup.sh` - Bash script to setup the environment
* `train.py` - Script to train the agent. I used jupytext to run it as jupyter notebook
* `agent.py` - Implementation of the learning algorithm
* `agent_utils` - Folder containing several code files that are used by the agent class(e.g. replaybuffer)
* `report.ipynb` - Final report of the project

## Getting Started

This project was developed and tested on an Apple Macbook with an intel i5. No guarantees are made on performance in other systems. 

### Mac

To setup your coding evironment you need to perform 3 steps after cloning this repository:

1. Make `setup.sh` executable. There are many ways to do this. One way is through the terminal run this command:

```bash
chmod +x setup.py
```

2. Then you simply run `setup.py`.

3. Finally you activate the conda environment in your terminal or on your notebook change the kernel to `drl_navigation`
### Others

If you are running this on other operating systems. There's a strong possibility that you can just follow the instructions for mac. Otherwise you will need to follow the steps in the [readme](Value-based-methods/README.md) in the `Value-based-methods` repo.

#### To download the environment for other Operating Systems
If you are using another OS you'll need to manually download the environment from one of the links below.  You need only select the environment that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Crawler/Crawler_Windows_x86_64.zip)

## The Environment

TODO

## Training and report

The training code can be found in `train.py`. This is a jupyter notebook created with [jupytext](https://github.com/mwouts/jupytext#:~:text=Jupytext%20is%20a%20plugin%20for,Scripts%20in%20many%20languages.) so it can be opened either as a notebook or used as a script. 

The final report can be found in `report.ipynb` and is a regular notebook as required for the delivery of the project.

## Author

Diogo Oliveira