# Getting Started

> If you want to study how to execute already implemented pipelines, please refer [Usage](usage.md)

This document will teach how to create a new Pipeline, with all necessary parts.

The document is written under the Assumption a developer has already created strong ideas, that should flow into a pipeline.
This could be done on paper, script or notebook, but the final product should already kind of manifested.
Better in text, but in mind is also no issue.

As described in the [Overview](index.md/#overview), the first step is to decide if a part of the pipeline is a [`Permanence`][{{ permanence }}] or a [`PipelineProcess`][{{ process }}].

## Internal logic

In detail the following image provides some information how the Pipeline is structured.
If you do not want to read the internal logic jump straight to [Create Config](#create-config)

![UML](assets/TensorImgPipeline.svg)
///caption
UML Diagram sketch of `TensorImgPipeline`
///

In the diagram are the three main components of the TensorImgPipeline depicted.
The [`PipelineController`][{{ controller }}] class is central part of the pipeline, storing and providing all the necessary components of the pipeline.
The [`Permanence`][{{ permanence }}] abstract class is used to create new implementations for objects which hold permanent information about the pipeline.
As example a `Dataset` is a kind of information, which a process needs to access, but could be created without the flow of the pipeline.
Compared to a script a `Dataset` is mostly an instance which is in the global scope.
The [`PipelineProcess`][{{ process }}] abstract class is the opposite to a [`Permanence`][{{ permanence }}].
As example a `Visualization` creates a figure, which displays given batch of images as subplots with certain config.
Of course, we could extract further parts of the `Visualization` as new [`Permanence`][{{ permanence }}] implementations, as example the parameters of the figure.

The [`PipelineController`][{{ controller }}] is instantiated with a dictionary of zero [`Permanence`][{{ permanence }}] or more.
This step is displayed as the aggregation[^1].
Afterwards one [`PipelineProcesse`][{{ process }}] or more are added to the [`PipelineController`][{{ controller }}].
This step is displayed as the composition[^1].
The run method of the [`PipelineController`][{{ controller }}] class iterates over each [`PipelineProcess`][{{ process }}] and calls it [`execute`][{{ process_execute }}] method, which uses (displayed as association[^1]) the controller itself to access the [`Permanence`][{{ permanence }}] if needed.

[^1]: A short explanation between an [aggregation vs. composition vs. association](https://www.visual-paradigm.com/guide/uml-unified-modeling-language/uml-aggregation-vs-composition/).

Since doing this by hand is cumbersome and error-prone, a [`PipelineBuilder`][{{ builder }}] class is provided, which executes this behavior.

## Create Config

After the decision which Parts fall into either the [`Permanence`][{{ permanence }}] or the [`PipelineProcess`][{{ process }}] category, the creation of the config files begins.

> It is not necessary to complete the configs at this point. The goal is to create a structure to begin with and complete later.

Following steps are necessary to create the config structure:

1. Create new folder under configs with the name of the Pipeline.
2. Create `execute_pipeline.toml` file inside this folder.
3. Create `initial_pipeline.toml` file inside this folder.

### The `initial_pipeline.toml`

Since not all components of a Pipeline should be part of the repository, it is necessary to provide such information.

Following components of a Pipeline needed to be defined inside this config:

- Dependencies
- Dataset locations
- Model locations

An Example config could look like the following:

```toml
[dependencies.submodules.sam]
name = "segment_anything"
url = "https://github.com/facebookresearch/segment-anything.git"
commit = "dca509fe"

[data.datasets.pascal]
location = "/mnt/data1/datasets/image/voc/VOCdevkit/VOC2012/"
format = "pascalvoc"

[data.models.sam]
location = "/mnt/data1/models/sam/sam_vit_h_4b8939.pth"
format = "pth"
```

The dependency is in this case a git submodule, but could be also a pip package.
The dependencies will be installed via `uv pip install`, so it will not interfere with `pyproject.toml`.

A dataset and a model are always provided with a key (`pascal` or `sam` in this case), a location and a format.
The key will be used to create the symlink below `data/dataset` or `data/models` based on location.
The format helps to reduce Boilerplate for dataset creation.

### The `execute_pipeline.toml`

This config file is used to provide the different [`Permanence`][{{ permanence }}] and [`PipelineProcess`][{{ process }}] object configurations.

An example config could look like the following:

```toml
[permanences.data]
type = "Datasets"
params = { root = "localhost", format = "pascalvoc" }

[processes.viz]
type = "Visualization"
```

Here are one [`Permanence`][{{ permanence }}] and one [`PipelineProcess`][{{ process }}] object provided.
The `params` field is always Optional.
The `type` field provides the classes of the Pipeline, which we need to create inside our [Pipeline Library](#create-pipeline-library).
The `params` need to match accordingly to the `__init__` method of the Implementation.

Additional configuration for provided [`Permanence`][{{ permanence }}] or [`PipelineProcess`][{{ process }}] classes could here be provided.
As example from the core package the `Visualization` process is defined in the config.
The [`Builder`][{{ builder }}] will first register the `Visualization` class from core package.
It is also possible to override the `Visualization` class inside the pipeline library.

## Create Pipeline Library

With the initiation of the config files we can begin creating the actual pipeline library.
It is necessary to create for every entry in the config its implementation, but we have the freedom to provide additional pipeline internal structure.
The structure of a new pipeline subpackage looks the following:

```tree
📦tipi
 ┗ 📦pipelines
    ┗ 📦node_modules
       ┣ 📜__init__.py
       ┣ 📜permanences.py
       ┗ 📜processes.py
```

Following the structure the implementations should be provided dependent on the config either inside the `permanences.py` or `processes.py` module.
This is just a recommendation to follow the pattern of the package, but a developer can decide against the pattern.
The only necessary part is the announcement of the [`Permanence`][{{ permanence }}] and the [`PipelineProcess`][{{ process }}] inside the subpackage `__init__.py` module.

For the previous example the `__init__.py` module looks the following:

```python
from permaneces import Datasets
from processes import DummyProcess

permanences_to_register = {"Datasets": Datasets}
processes_to_register = {"Visualization": Visualization}
```

## Execute or testing

From that point there are initial two possible ways to proceed.

1. Execute the Pipeline directly

This could be a good start point to get familiar how the pipeline works and looks as the final product.
It is good practice to follow this approach if the developer works first time with this project.
Nevertheless, the second way is an absolute recommendation, to create the final pipeline.

2. Creating test cases

The creation of test cases is a recommended approach, since at the end multiple small components of the pipeline are provided.
Instead of running all the time the complete pipeline, checking each bit with artificial and abstract test cases could provide spare time.
Not only this, after iterating through changes in the pipeline library, we are able to verify the correctness of the step in the Pipeline.
In a certain situation it might be necessary to extend the test case to fulfil new challenges, which were not thought about at the beginning of the Pipeline development.

## Documentation

As the last step or in parallel while developing a new pipeline, documenting the pipeline is a key feature of this package.

- Create a new markdown file with the name of the pipeline under `docs/pipelines`.
- Fill the markdown with the description, goal and results of the pipeline.
  - As example the `Visualization` could directly output images for documentation to the `docs/assets` location.
- Add the markdown file to navigation in `mkdocs.yml`:

```diff
@@ -12,7 +12,8 @@ nav:
   - Installation: installation.md
   - Usage: usage.md
   - Getting Started: getting_started.md
   - Pipelines:
+      - new_pipeline: pipelines/new_pipeline.md
   - Modules:
       - builder.py: modules/builder.md
       - controller.py: modules/controller.md
```

Additional also extending the module section is possible.
