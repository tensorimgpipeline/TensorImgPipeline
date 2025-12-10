# TensorImgPipeline - Proposed Architecture

**Philosophy**: Progressive Enhancement - Start Simple, Scale When Needed

This architecture supports researchers from initial script to production pipeline with minimal friction at each step.

## Overview Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                                  CLI LAYER                                 │
│                               (cli.py)                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • Parse command line arguments with Typer                          │   │
│  │  • Display formatted errors with Rich                               │   │
│  │  • Delegate to PipelineRunner                                       │   │
│  │                                                                     │   │
│  │  @app.command()                                                     │   │
│  │  def build_pipeline(pipeline_name: str):                            │   │
│  │      runner = PipelineRunner(pipeline_name)                         │   │
│  │      runner.run()                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────┬────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                              │
│                            (runner.py)                                  │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │  class PipelineRunner:                                              │ │
│ │      • High-level orchestration                                     │ │
│ │      • Coordinates Builder → Controller → Executor                  │ │
│ │      • Handles WandB sweep integration                              │ │
│ │      • Provides programmatic API entry point                        │ │
│ │                                                                     │ │
│ │      def build() -> (controller, error)                             │ │
│ │      def run() -> None                                              │ │
│ │      def _run_once(controller) -> None                              │ │
│ │      def _run_with_sweep(controller, wandb) -> None                 │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────┬─────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
┌────────────────────────────────────────┐   ┌────────────────────────────────┐
│         BUILDER LAYER                  │   │      CONTROLLER LAYER          │
│         (builder.py)                   │   │      (controller.py)           │
│  ┌─────────────────────────────────┐   │   │  ┌───────────────────────────┐ │
│  │  class PipelineBuilder:         │   │   │  │  class PipelineController │ │
│  │    • Load & validate config     │   │   │  │    • Manage permanence    │ │
│  │    • Register classes           │   │   │  │      lifecycle            │ │
│  │    • Build permanences          │   │   │  │    • Provide permanence   │ │
│  │    • Build process specs        │   │   │  │      access               │ │
│  │                                 │   │   │  │    • Instantiate          │ │
│  │    def build() ->               │   │   │  │      processes            │ │
│  │      (permanences,              │───┼───┼▶│    • Coordinate cleanup   │ │
│  │       process_specs,            │   │   │  │                           │ │
│  │       error)                    │   │   │  │    Methods:               │ │
│  └─────────────────────────────────┘   │   │  │    • initialize_          │ │
└────────────────────────────────────────┘   │  │      permanences()        │ │
                                             │  │    • validate_            │ │
                                             │  │      permanences()        │ │
                                             │  │    • checkpoint_          │ │
                                             │  │      permanences()        │ │
                                             │  │    • get_permanence()     │ │
                                             │  │    • iterate_processes()  │ │
                                             │  │    • cleanup()            │ │
                                             │  └───────────────────────────┘ │
                                             └────────────┬───────────────────┘
                                                            │
                                                            ▼
                                               ┌───────────────────────────────┐
                                               │      EXECUTOR LAYER           │
                                               │      (executor.py)            │
                                               │  ┌────────────────────────┐   │
                                               │  │  class PipelineExecutor│   │
                                               │  │    • Execute processes │   │
                                               │  │    • Apply progress    │   │
                                               │  │      decoration        │   │
                                               │  │    • Handle nested     │   │
                                               │  │      progress bars     │   │
                                               │  │    • Integrate WandB   │   │
                                               │  │    • Error handling    │   │
                                               │  │                        │   │
                                               │  │    Methods:            │   │
                                               │  │    • run()             │   │
                                               │  │    • _run_processes()  │   │
                                               │  │    • _run_cleanup()    │   │
                                               │  │    • _handle_error()   │   │
                                               │  └────────────────────────┘   │
                                               └────────────┬──────────────────┘
                                                            │
                    ┌───────────────────────────────────────┴─────────────┐
                    ▼                                                     ▼
┌─────────────────────────────────────────┐              ┌────────────────────────────────┐
│         PERMANENCES                     │              │         PROCESSES              │
│         (abstractions.py)               │              │         (abstractions.py)      │
│  ┌──────────────────────────────────┐   │              │  ┌──────────────────────────┐  │
│  │  class Permanence(ABC):          │   │              │  │  class PipelineProcess:  │  │
│  │    • cleanup()                   │   │              │  │    • execute()           │  │
│  │    • initialize() [optional]     │   │              │  │    • skip()              │  │
│  │    • validate() [optional]       │   │              │  │    • Access controller   │  │
│  │    • checkpoint() [optional]     │   │              │  └──────────────────────────┘  │
│  │    • get_state() [optional]      │   │              │                                │
│  └──────────────────────────────────┘   │              │  Implementations:              │
│                                         │              │  • ResultProcess               │
│  Implementations:                       │              │  • TrainingProcess             │
│  ┌──────────────────────────────────┐   │              │  • ValidationProcess           │
│  │  Device                          │   │              │  • TestProcess                 │
│  │    • Manages CUDA device         │   │              │  • DataLoadProcess             │
│  │    • Selects best GPU            │   │              │  • ...custom processes         │
│  ├──────────────────────────────────┤   │              └────────────────────────────────┘
│  │  ProgressManager                 │   │
│  │    • Rich progress bars          │   │
│  │    • Nested progress tracking    │   │
│  ├──────────────────────────────────┤   │
│  │  WandBManager                    │   │
│  │    • Experiment logging          │   │
│  │    • Sweep management            │   │
│  ├──────────────────────────────────┤   │
│  │  Network                         │   │
│  │    • Model instance              │   │
│  │    • Model state                 │   │
│  ├──────────────────────────────────┤   │
│  │  Data                            │   │
│  │    • Datasets                    │   │
│  │    • Data loaders                │   │
│  ├──────────────────────────────────┤   │
│  │  Hyperparameters                 │   │
│  │    • Configuration               │   │
│  │    • Sweep config                │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

## Permanence Lifecycle Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PERMANENCE LIFECYCLE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: CONSTRUCTION
┌─────────────────────────────────────────────────────────────┐
│  Builder instantiates permanence with config params         │
│  permanence = Device()                                      │
│  permanence = ProgressManager(console=console)              │
│  permanence = WandBManager(project="...", entity="...")     │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
Phase 2: INITIALIZATION (Controller)
┌─────────────────────────────────────────────────────────────┐
│  controller.initialize_permanences()                        │
│    → permanence.initialize()                                │
│       • Validate dependencies exist                         │
│       • Allocate resources (memory, GPU)                    │
│       • Setup connections (WandB, databases)                │
│       • Verify configuration validity                       │
│                                                             │
│  Example: Device.initialize()                               │
│    - Checks CUDA availability                               │
│    - Validates VRAM thresholds                              │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
Phase 3: VALIDATION (Executor - Optional)
┌─────────────────────────────────────────────────────────────┐
│  controller.validate_permanences()                          │
│    → permanence.validate()                                  │
│       • Health checks                                       │
│       • State verification                                  │
│       • Consistency checks                                  │
│                                                             │
│  Example: Device.validate()                                 │
│    - Checks VRAM usage < 95%                                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
Phase 4: EXECUTION (Processes access permanences)
┌─────────────────────────────────────────────────────────────┐
│  Process execution:                                         │
│    device = controller.get_permanence("device")             │
│    progress = controller.get_permanence("progress_manager") │
│    wandb = controller.get_permanence("wandb_logger")        │
│                                                             │
│  Processes use permanences throughout execution:            │
│    • Access device for GPU operations                       │
│    • Update progress bars                                   │
│    • Log metrics to WandB                                   │
│    • Load data from datasets                                │
│                                                             │
│  Nested progress bars created by processes:                 │
│    @progress_manager.progress_task("train")                 │
│    def train_epoch(task_id, total, progress):               │
│        for batch in dataloader:                             │
│            progress.advance(task_id)                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
Phase 5: CHECKPOINTING (Executor - Optional)
┌─────────────────────────────────────────────────────────────┐
│  controller.checkpoint_permanences()                        │
│    → permanence.checkpoint()                                │
│       • Save intermediate state                             │
│       • Create backups                                      │
│       • Log checkpoint metrics                              │
│                                                             │
│  Example: Network.checkpoint()                              │
│    - Saves model weights                                    │
│    - Saves optimizer state                                  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
Phase 6: CLEANUP (Always executed)
┌─────────────────────────────────────────────────────────────┐
│  controller.cleanup()                                       │
│    → permanence.cleanup()                                   │
│       • Release memory (RAM/VRAM)                           │
│       • Close file handles                                  │
│       • Close network connections                           │
│       • Finalize logging                                    │
│       • Save final state                                    │
│                                                             │
│  Example: Device.cleanup()                                  │
│    - Clears CUDA cache                                      │
│  Example: WandBManager.cleanup()                            │
│    - Finalizes WandB run                                    │
│  Example: ProgressManager.cleanup()                         │
│    - Closes progress bars                                   │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW                                        │
└─────────────────────────────────────────────────────────────────────────────┘

1. Configuration Loading
   ┌────────────────┐
   │ TOML Config    │
   │ Files          │
   └────────┬───────┘
            │
            ▼
   ┌────────────────┐       ┌──────────────────┐
   │ PipelineBuilder│─────▶│ Class Registry   │
   │                │       │ (Permanences +   │
   │ • Load TOML    │       │  Processes)      │
   │ • Validate     │       └──────────────────┘
   │ • Parse        │
   └────────┬───────┘
            │
            ▼
   ┌────────────────────────────────┐
   │ Permanence Instances           │
   │ + Process Specifications       │
   └────────┬───────────────────────┘
            │
            ▼

2. Pipeline Execution Flow
   ┌─────────────────────────────────────────────────────────┐
   │ PipelineRunner                                          │
   │   runner.run()                                          │
   └───────────────────┬─────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
   ┌─────────┐   ┌─────────┐   ┌─────────┐
   │ Builder │   │Controller│  │Executor │
   └────┬────┘   └────┬────┘   └────┬────┘
        │             │              │
        │ build()     │              │
        ├───────────▶│              │
        │             │              │
        │         initialize_        │
        │         permanences()      │
        │             ├────────────▶│
        │             │              │
        │             │         validate_
        │             │         permanences()
        │             │              │
        │             │              │
        │             │         run()│
        │             │              ├─────┐
        │             │              │     │
        │             │              │  Execute Processes
        │             │              │     │
        │             │              │     ▼
        │             │              │  ┌──────────────┐
        │             │              │  │ Process 1    │
        │             │◀────────────┼──│ get_perm()   │
        │             │              │  └──────────────┘
        │             │              │     │
        │             │              │     ▼
        │             │              │  ┌──────────────┐
        │             │              │  │ Process 2    │
        │             │◀────────────┼──│ get_perm()   │
        │             │              │  └──────────────┘
        │             │              │     │
        │             │              │     ▼
        │             │              │  ┌──────────────┐
        │             │              │  │ Process N    │
        │             │◀────────────┼──│ get_perm()   │
        │             │              │  └──────────────┘
        │             │              │      │
        │             │              │  checkpoint_
        │             │              │  permanences()
        │             │              ◀─────┘
        │             │              │
        │             │         cleanup()
        │             ◀─────────────│
        │             │              │
        │         cleanup()          │
        │             │              │
        ▼             ▼              ▼
```

## Process Access Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROCESS PERMANENCE ACCESS                                │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────┐
  │  class TrainingProcess(PipelineProcess):                      │
  │                                                               │
  │      def __init__(self, controller, force, epochs):           │
  │          super().__init__(controller, force)                  │
  │                                                               │
  │          # Access permanences through controller              │
  │          self.device = controller.get_permanence("device")    │
  │          self.network = controller.get_permanence("network")  │
  │          self.progress = controller.get_permanence(           │
  │              "progress_manager"                               │
  │          )                                                    │
  │          self.wandb = controller.get_permanence(              │
  │              "wandb_logger"                                   │
  │          )                                                    │
  │          self.data = controller.get_permanence("data")        │
  │                                                               │
  │      def execute(self) -> Optional[Exception]:                │
  │          # Use permanences in execution                       │
  │          model = self.network.model_instance                  │
  │          model.to(self.device.device)                         │
  │                                                               │
  │          # Create nested progress bar                         │
  │          @self.progress.progress_task("train")                │
  │          def train_loop(task_id, total, progress):            │
  │              for epoch in range(total):                       │
  │                  loss = self._train_epoch(model)              │
  │                  self.wandb.log_metrics({"loss": loss})       │
  │                  progress.advance(task_id)                    │
  │                                                               │
  │          train_loop(self.epochs)                              │
  │          return None                                          │
  └───────────────────────────────────────────────────────────────┘
                                    │
                                    │ Access Pattern
                                    ▼
  ┌───────────────────────────────────────────────────────────────┐
  │  PipelineController                                           │
  │                                                               │
  │  _permanences = {                                             │
  │      "device": <Device instance>,                             │
  │      "network": <Network instance>,                           │
  │      "progress_manager": <ProgressManager instance>,          │
  │      "wandb_logger": <WandBManager instance>,                 │
  │      "data": <Data instance>                                  │
  │  }                                                            │
  │                                                               │
  │  def get_permanence(self, name: str) -> Any:                  │
  │      if name not in self._permanences:                        │
  │          raise PermanenceKeyError(...)                        │
  │      return self._permanences[name]                           │
  └───────────────────────────────────────────────────────────────┘
```

## WandB Integration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WandB INTEGRATION                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Standard Run (No Sweep)
┌─────────────────────────────────────────────────────────┐
│  PipelineRunner.run()                                   │
│    │                                                    │
│    ├─ Check for WandB sweep config                      │
│    │    wandb = controller.get_permanence("wandb")      │
│    │    if not wandb.sweep_id:                          │
│    │                                                    │
│    └─▶ runner._run_once(controller)                    │
│          │                                              │
│          └─▶ PipelineExecutor.run()                    │
│                │                                        │
│                ├─ wandb.init_wandb()  # Init single run │
│                │                                        │
│                ├─ executor._run_processes()             │
│                │    └─▶ Processes log to WandB         │
│                │                                        │
│                └─ executor._run_cleanup()               │
│                     └─▶ wandb.cleanup() # Finalize run │
└─────────────────────────────────────────────────────────┘

Sweep Run (Multiple Runs)
┌────────────────────────────────────────────────────────┐
│  PipelineRunner.run()                                  │
│    │                                                   │
│    ├─ Check for WandB sweep config                     │
│    │    wandb = controller.get_permanence("wandb")     │
│    │    hyperparams = controller.get_permanence(       │
│    │        "hyperparams"                              │
│    │    )                                              │
│    │    if wandb.sweep_id:                             │
│    │                                                   │
│    └─▶ runner._run_with_sweep(controller, wandb)      │
│          │                                             │
│          ├─ wandb.create_sweep(                        │
│          │     hyperparams.sweep_configuration         │
│          │ )                                           │
│          │                                             │
│          └─ wandb.create_sweep_agent(                  │
│                function=lambda: runner._run_once()     │
│            )                                           │
│             │                                          │
│             │   ┌─────── Agent spawns N runs ────────┐ │
│             │   │                                    │ │
│             └─▶│  Run 1:                            │ │
│                 │    wandb.init_wandb()              │ │
│                 │    executor._run_processes()       │ │
│                 │    wandb.cleanup()                 │ │
│                 │                                    │ │
│                 │  Run 2:                            │ │
│                 │    wandb.init_wandb()              │ │
│                 │    executor._run_processes()       │ │
│                 │    wandb.cleanup()                 │ │
│                 │                                    │ │
│                 │  ...                               │ │
│                 │                                    │ │
│                 │  Run N:                            │ │
│                 │    wandb.init_wandb()              │ │
│                 │    executor._run_processes()       │ │
│                 │    wandb.cleanup()                 │ │
│                 └────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

## Progress Bar Nesting Example

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NESTED PROGRESS BARS                                     │
└─────────────────────────────────────────────────────────────────────────────┘

Visual Output:
┌─────────────────────────────────────────────────────────────────────┐
│ Overall      ████████░░░░░░░░░░░░░░░░░░░░ (2/5) • 00:15:30          │ ← Executor
│ Cleanup      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (0/6) • 00:00:00          │ ← Executor
│ Epoch        ████████████████░░░░░░░░░░░░ (2/3) • Status: Train     │ ← Process
│ Train-Val    ████████████████████████████ (100/100) • 00:02:15      │ ← Process
│ Result       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░ (0/15) • 00:00:00         │ ← Process
└─────────────────────────────────────────────────────────────────────┘

Code Flow:
┌──────────────────────────────────────────────────────────┐
│ PipelineExecutor                                         │
│                                                          │
│   @progress_decorator("overall")  # Top-level bar        │
│   def _execute(task_id, total, progress):                │
│       for idx, process in enumerate(processes):          │
│           process.execute()  # Process creates nested    │
│           progress.advance(task_id)                      │
│                                                          │
│   @progress_decorator("cleanup")  # Top-level bar        │
│   def _cleanup(task_id, total, progress):                │
│       controller.cleanup()                               │
│       progress.advance(task_id)                          │
└──────────────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│ TrainingProcess                                          │
│                                                          │
│   def execute(self):                                     │
│       @self.progress.progress_task("epoch")              │
│       def train_epochs(task_id, total, progress):        │
│           for epoch in range(total):                     │
│               @self.progress.progress_task("train-val")  │
│               def train_val(tid, tot, prog):             │
│                   for batch in dataloader:               │
│                       # Training logic                   │
│                       prog.advance(tid)                  │
│               train_val(num_batches)                     │
│               progress.advance(task_id)                  │
│                                                          │
│       train_epochs(self.epochs)                          │
└──────────────────────────────────────────────────────────┘
```

## Testing Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TESTING LAYERS                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Unit Tests:
┌────────────────────────────────────────────────────┐
│ test_permanences.py                                │
│   • Test each permanence in isolation              │
│   • Mock dependencies                              │
│   • Test lifecycle methods                         │
│                                                    │
│ test_builder.py                                    │
│   • Test config loading                            │
│   • Test class registration                        │
│   • Test permanence/process building               │
│                                                    │
│ test_controller.py                                 │
│   • Test permanence access                         │
│   • Test process iteration                         │
│   • Test lifecycle coordination                    │
│                                                    │
│ test_executor.py                                   │
│   • Test process execution                         │
│   • Test error handling                            │
│   • Mock progress manager                          │
└────────────────────────────────────────────────────┘

Integration Tests:
┌────────────────────────────────────────────────────┐
│ test_pipeline_integration.py                       │
│   • Test Builder → Controller → Executor flow      │
│   • Use mock permanences and processes             │
│   • Verify data flow                               │
│                                                    │
│ test_runner.py                                     │
│   • Test PipelineRunner end-to-end                 │
│   • Mock WandB integration                         │
│   • Test programmatic API                          │
└────────────────────────────────────────────────────┘

Programmatic Testing Example:
┌────────────────────────────────────────────────────┐
│ # No CLI needed!                                   │
│                                                    │
│ def test_pipeline_execution():                     │
│     # Create mock permanences                      │
│     permanences = {                                │
│         "device": MockDevice(),                    │
│         "network": MockNetwork(),                  │
│         "progress_manager": None,  # Optional      │
│     }                                              │
│                                                    │
│     # Create mock process specs                    │
│     process_specs = [                              │
│         ProcessWithParams(                         │
│             MockProcess,                           │
│             {"force": False}                       │
│         )                                          │
│     ]                                              │
│                                                    │
│     # Create controller                            │
│     controller = PipelineController(               │
│         permanences,                               │
│         process_specs                              │
│     )                                              │
│                                                    │
│     # Execute                                      │
│     executor = PipelineExecutor(controller)        │
│     executor.run()                                 │
│                                                    │
│     # Verify results                               │
│     assert MockProcess.executed                    │
└────────────────────────────────────────────────────┘
```

## Key Benefits Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            KEY BENEFITS                                      │
└─────────────────────────────────────────────────────────────────────────────┘

✓ Clear Separation of Concerns
  ├─ CLI: User interaction only
  ├─ Runner: Orchestration
  ├─ Builder: Component construction
  ├─ Controller: Lifecycle management
  └─ Executor: Execution & visualization

✓ Testability
  ├─ Each layer independently testable
  ├─ Mock permanences and processes
  └─ No CLI required for testing

✓ Reusability
  ├─ Programmatic API via PipelineRunner
  ├─ Can embed in other applications
  └─ Flexible executor implementations

✓ Extensibility
  ├─ Easy to add new permanences
  ├─ Easy to add new processes
  ├─ Optional lifecycle hooks
  └─ Plugin architecture ready

✓ Maintainability
  ├─ Single responsibility per class
  ├─ Clear dependencies
  ├─ Well-defined interfaces
  └─ Structured lifecycle

✓ Flexibility
  ├─ Swap executor implementations
  ├─ Optional visualization
  ├─ WandB sweep support
  └─ Nested progress bars
```

---

## Progressive Enhancement: Helper Layer

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HELPER LAYER (NEW!)                                 │
│                         (helpers.py, decorators.py)                         │
│                                                                             │
│  Bridges the gap between scripts and pipelines                              │
│  Works standalone OR with pipeline context                                  │
└─────────────────────────────────────────────────────────────────────────────┘

  SCRIPT MODE (Standalone)             │         PIPELINE MODE
                                       │
┌─────────────────────────────────┐    │    ┌──────────────────────────────┐
│  # researcher_script.py         │    │    │  # Within PipelineExecutor   │
│                                 │    │    │                              │
│  from helpers import (          │    │    │  # Set context               │
│      progress_bar,              │    │    │  set_pipeline_context({      │
│      logger,                    │    │    │      "progress_manager": pm, │
│      device_manager             │    │    │      "wandb_logger": wb,     │
│  )                              │    │    │      "device": dev           │
│                                 │    │    │  })                          │
│  # Works without pipeline!      │    │    │                              │
│  logger.init(project="exp")     │    │    │  # Helpers auto-detect       │
│                                 │    │    │  # pipeline and use it       │
│  for epoch in progress_bar(...):│    │    │                              │
│      device = device_manager    │    │    │  # Same code works!          │
│          .get_device()          │    │    │  for epoch in progress_bar():│
│      logger.log({"loss": loss}) │    │    │      device = device_manager │
│                                 │    │    │          .get_device()       │
└─────────────────────────────────┘    │    │      logger.log({...})       │
                                       │    └──────────────────────────────┘
         Uses Rich & basic WandB       │       Uses Pipeline's permanences


DECORATOR PATTERN: Function → Process
┌────────────────────────────────────────────────────────────────────────────┐
│  @pipeline_process                     # Marks function as pipeline-ready  │
│  def train(epochs: int = 10):         # Parameters become config options   │
│      '''Can run standalone OR in pipeline!'''                              │
│                                                                            │
│      device = device_manager.get_device()  # Works in both modes           │
│      model = MyModel().to(device)                                          │
│                                                                            │
│      for epoch in progress_bar(range(epochs), desc="Training"):            │
│          loss = train_step(model)                                          │
│          logger.log({"loss": loss})                                        │
│                                                                            │
│  # Run as script                                                           │
│  if __name__ == "__main__":                                                │
│      train(epochs=5)  # ← Normal function call                             │
│                                                                            │
│  # Or in config.toml for pipeline                                          │
│  # [processes.training]                                                    │
│  # type = "train"                                                          │
│  # params = { epochs = 10 }                                                │
└────────────────────────────────────────────────────────────────────────────┘


HELPER IMPLEMENTATIONS
┌────────────────────────────────────────────────────────────────────────────┐
│  progress_bar(iterable, desc="Processing")                                 │
│    ├─ No pipeline context: Uses rich.track (tqdm-like)                     │
│    └─ With pipeline context: Uses pipeline's ProgressManager               │
│                                                                            │
│  logger.init(project, entity) / logger.log(metrics)                        │
│    ├─ No pipeline context: Initializes WandB manually                      │
│    └─ With pipeline context: Uses pipeline's WandBManager                  │
│                                                                            │
│  device_manager.get_device()                                               │
│    ├─ No pipeline context: Selects first available GPU                     │
│    └─ With pipeline context: Uses pipeline's Device permanence             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Research-to-Production Journey

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FROM SCRIPT TO PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Level 0: Raw Script                     Changes: 0
┌────────────────────────────────┐
│  for epoch in range(10):       │      • No framework
│      loss = train()            │      • Manual everything
│      print(loss)               │      • Hard to track
└────────────────────────────────┘

            │ Add 1 import, wrap loops
            ▼

Level 1: Progress Bars                  Changes: +2 lines
┌──────────────────────────────────┐
│  from helpers import             │      ✓ Visual progress
│      progress_bar                │      ✓ Time estimates
│                                  │      • Still a script
│  for epoch in progress_bar(...): │
│      loss = train()              │
└──────────────────────────────────┘

            │ Add logger.init() and logger.log()
            ▼

Level 2: Experiment Tracking             Changes: +3 lines
┌──────────────────────────────────┐
│  from helpers import             │      ✓ Progress bars
│      progress_bar, logger        │      ✓ WandB logging
│                                  │      ✓ Experiment tracking
│  logger.init(project="exp")      │      • Still a script
│                                  │
│  for epoch in progress_bar(...): │
│      loss = train()              │
│      logger.log({"loss": loss})  │
└──────────────────────────────────┘

            │ Add device_manager
            ▼

Level 3: Device Management               Changes: +2 lines
┌────────────────────────────────┐
│  from helpers import           │      ✓ Progress + logging
│      progress_bar, logger,     │      ✓ Auto device select
│      device_manager            │      ✓ Multi-GPU support
│                                │      • Still a script
│  device = device_manager       │
│      .get_device()             │
│  model.to(device)              │
└────────────────────────────────┘

            │ Extract functions, add decorator
            ▼

Level 4: Reusable Functions              Changes: Extract to functions
┌─────────────────────────────────┐
│  @pipeline_process              │      ✓ All previous features
│  def train(epochs: int = 10):   │      ✓ Reusable code
│      device = device_manager    │      ✓ Type hints
│          .get_device()          │      ✓ Ready for pipeline
│      for epoch in progress_bar( │      • Still runs as script
│          range(epochs)          │
│      ):                         │
│          loss = train_step()    │
│          logger.log(...)        │
│                                 │
│  if __name__ == "__main__":     │
│      train(epochs=5)            │
└─────────────────────────────────┘

            │ Create config, register in pipeline
            ▼

Level 5: Full Pipeline                   Changes: Config file + class wrapper
┌────────────────────────────────┐
│  # config.toml                 │      ✓ All previous features
│  [processes.training]          │      ✓ Config-driven
│  type = "TrainingProcess"      │      ✓ Permanence lifecycle
│  params = { epochs = 10 }      │      ✓ Production ready
│                                │      ✓ Team collaboration
│  # Run via CLI                 │      ✓ CI/CD integration
│  $ tipi run exp     │
└────────────────────────────────┘
```

## Key Insight: Copy-Paste Compatibility

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              CODE REUSE BETWEEN LEVELS                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Training Logic (Same across levels!)
┌────────────────────────────────────────────────────────────┐
│  def train_step(model, batch, device):                     │
│      inputs, targets = batch                               │
│      inputs = inputs.to(device)                            │
│      targets = targets.to(device)                          │
│                                                            │
│      optimizer.zero_grad()                                 │
│      outputs = model(inputs)                               │
│      loss = criterion(outputs, targets)                    │
│      loss.backward()                                       │
│      optimizer.step()                                      │
│                                                            │
│      return loss.item()                                    │
└────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   Level 2-3          Level 4           Level 5
   (Script)      (Function)          (Process)

  Same code!    Same code!          Same code!
  Just called   Just decorated      Just wrapped
  directly      with @pipeline      in PipelineProcess
                                    class

No rewriting needed - just progressive wrapping!
```

## Documentation References

- **Full Guide**: See `docs/progressive_enhancement.md`
- **Helper API**: See `tipi/helpers.py`
- **Decorators**: See `tipi/decorators.py`
