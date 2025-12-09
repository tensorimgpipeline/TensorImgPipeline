# Ideas

This Chapter describes Ideas to enhance the project.
It can be interpreted as lists of TODOS.

## Nested Process Bars

Since a pipeline does have various steps we want an informative way to show the status of the current running pipeline.

Here are two approaches for nested progress bars:

=== "TQDM"

    [source](https://stackoverflow.com/a/38489852/10985257)

    ```python
    from tqdm import tqdm
    # from tqdm.auto import tqdm  # notebook compatible
    import time
    for i1 in tqdm(range(5)):
        for i2 in tqdm(range(300), leave=False):
            # do something, e.g. sleep
            time.sleep(0.01)
    ```

=== "RICH"

    [source](https://stackoverflow.com/a/70611472/10985257)

    ```python
    import time

    from rich.progress import Progress

    with Progress() as progress:

        task1 = progress.add_task("[red]Downloading...", total=1000)
        task2 = progress.add_task("[green]Processing...", total=1000)
        task3 = progress.add_task("[cyan]Cooking...", total=1000)

        while not progress.finished:
            progress.update(task1, advance=0.5)
            progress.update(task2, advance=0.3)
            progress.update(task3, advance=0.9)
            time.sleep(0.02)
    ```

### Ansolved Questions:

- How to handle monitoring of internal loops?

## Tensorboard Process

A core Tensorboard process which based on config creates different visualizations.

## Wandb Process

A core Wandb process which based on config creates different visualizations and perform hyperparam trainings.

## PreRun Validation

Since we create mostly everything dynamic from the config and only checking that the general structure of the config is correct, also validating the actual run before executing, would be beneficial.
Since most Processes will access Permanences of the PipelineController, we could before the final execution check that every call of controller.used_permanence of every process doesn't result in a `NameError` (if `Permanence` is not defined), `AttributeError` (if `Permanence` does not have given Attribute) or `TypeError` (if `Permanece` method got not the correct Attributes).

The Pipeline can still crash caused by internal other errors, but we ensure, that the Process will not fail caused by issue in the config.

## Init Script / cli command for new Pipelines

A init script or cli command could speed up start times.
