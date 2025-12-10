# TensorImgPipeline

![LOGO](assets/logos/tipi_logo_text.v2.png)

[![Release](https://img.shields.io/github/v/release/tensorimgpipeline/TensorImgPipeline)](https://img.shields.io/github/v/release/tensorimgpipeline/TensorImgPipeline)
[![Build status](https://img.shields.io/github/actions/workflow/status/tensorimgpipeline/TensorImgPipeline/main.yml?branch=main)](https://github.com/tensorimgpipeline/TensorImgPipeline/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/tensorimgpipeline/TensorImgPipeline)](https://img.shields.io/github/commit-activity/m/tensorimgpipeline/TensorImgPipeline)
[![License](https://img.shields.io/github/license/tensorimgpipeline/TensorImgPipeline)](https://img.shields.io/github/license/tensorimgpipeline/TensorImgPipeline)

This is a repository for creating and running Tensor Image Pipelines, short tipis.

## Overview

The TensorImgPipeline is a python package, which allows the developer to create new Deep Learning approaches in as a generalized Pipeline.

The most approaches to new experiments are script or notebook based.
While this approach helps to create fast results and showcase the Experiment, it is difficult to create a robust and repeatable structure.

The script approach often leads to not maintainable code structure.
Often multiple things are done in one place, which logical are only weakly connected.
As example creating visualization, providing the visualization to the experiment server and training are often applied in the same script.
Combined with configuration of those elements in the same script leads to a chaotic structure.

We might need the structure, because they were a direct execution of our thoughts, which are also just came up while creating the first prototype of the experiment.
But melting this chaos into a more structured and organized environment, often seems a huge effort.
An effort where we might think it isn't worth it.

At exactly this point TensorImgPipeline wants to shine.
The Project provides the base structure to build structured, repeatable Experiments.

To start the new experiment journey the developer/scientist/engineer just needs to categorize parts of the Experiments as Permanence or Process.
After that, the [Getting Started](getting_started.md) provide the information needed to build the structured Pipeline.
