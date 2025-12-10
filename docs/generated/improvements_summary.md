# Architecture Improvements for Research Workflow

## Problem Statement

The original architecture required too much upfront structure:

- Define Permanence/Process classes
- Create TOML configs
- Register everything in builder
- Go through CLI/runner

**This creates friction for researchers who just want to experiment!**

---

## Solution: Progressive Enhancement Architecture

### Core Principle

**"Start simple, scale when needed"**

Researchers can:

1. Start with plain Python scripts
2. Add features incrementally (progress bars → logging → device management)
3. Extract to functions when code stabilizes
4. Convert to pipeline only when going to production

---

## New Components

### 1. Helper Module (`helpers.py`)

Provides script-level utilities that work **standalone OR with pipeline**:

```python
from tipi.helpers import progress_bar, logger, device_manager

# Works in plain scripts!
for epoch in progress_bar(range(10)):
    device = device_manager.get_device()
    loss = train()
    logger.log({"loss": loss})
```

**Key Features**:

- `progress_bar()`: tqdm-like progress bars (uses Rich standalone, pipeline's ProgressManager when available)
- `logger`: WandB logging (initializes manually or uses pipeline's WandBManager)
- `device_manager`: Smart GPU selection (simple selection standalone, uses pipeline's Device when available)

**How it works**:

- Checks global `_pipeline_context` to detect if running in pipeline
- If context exists, uses pipeline permanences
- If not, provides standalone implementations

### 2. Decorator Module (`decorators.py`)

Makes functions pipeline-ready with zero code changes:

```python
from tipi.decorators import pipeline_process

@pipeline_process
def train(epochs: int = 10):
    """This function works standalone AND in pipeline!"""
    for epoch in helpers.progress_bar(range(epochs)):
        loss = train_step()
        helpers.logger.log({"loss": loss})

# Run as script
if __name__ == "__main__":
    train(epochs=5)

# Or register in pipeline config:
# [processes.training]
# type = "train"
# params = { epochs = 10 }
```

**Key Features**:

- Decorated functions remain callable as normal functions
- Decorator creates a `PipelineProcess` class dynamically
- Function parameters become config parameters
- Automatic pipeline context injection

### 3. Progressive Enhancement Guide (`docs/progressive_enhancement.md`)

Complete guide showing 5 levels of enhancement:

- Level 0: Raw script
- Level 1: Add progress bars (+2 lines)
- Level 2: Add logging (+3 lines)
- Level 3: Better device management (+2 lines)
- Level 4: Extract to reusable functions
- Level 5: Full pipeline (config file + class wrapper)

Each level adds ONE concept, building on previous levels.

---

## Architecture Changes

### Before (Original Proposed)

```
CLI → Runner → Builder → Controller → Executor → Processes
                                                      ↑
                                              Requires class definition
                                              Requires TOML config
                                              Requires registration
```

**Entry barrier**: HIGH (must understand entire pipeline)

### After (With Progressive Enhancement)

```
┌─────────────────────────────────────────────────────┐
│  Script Mode                                        │
│  ──────────────────────────────────────────────     │
│  Plain Python → helpers.py (standalone mode)        │
│                                                     │
│  Entry barrier: ZERO (just import helpers)          │
└─────────────────────────────────────────────────────┘
                      │
                      │ (when ready)
                      ▼
┌─────────────────────────────────────────────────────┐
│  Hybrid Mode                                        │
│  ──────────────────────────────────────────────     │
│  Functions + @pipeline_process → helpers (dual)     │
│                                                     │
│  Entry barrier: LOW (add decorator)                 │
└─────────────────────────────────────────────────────┘
                      │
                      │ (when productionizing)
                      ▼
┌─────────────────────────────────────────────────────┐
│  Pipeline Mode                                      │
│  ──────────────────────────────────────────────     │
│  CLI → Runner → Builder → Controller → Executor     │
│                                    ↓                │
│                            Processes → helpers      │
│                                    (pipeline mode)  │
│                                                     │
│  Entry barrier: MEDIUM (config + classes)           │
└─────────────────────────────────────────────────────┘
```

---

## Technical Implementation

### Pipeline Context Injection

The `PipelineExecutor` sets context before running processes:

```python
class PipelineExecutor:
    def _run_processes(self):
        # Set context for helpers
        from tipi.helpers import set_pipeline_context

        context = {
            "progress_manager": self.controller.get_permanence("progress_manager", None),
            "wandb_logger": self.controller.get_permanence("wandb_logger", None),
            "device": self.controller.get_permanence("device", None),
        }
        set_pipeline_context(context)

        try:
            # Run processes (they use helpers which now see the context)
            for process in self.controller.iterate_processes():
                process.execute()
        finally:
            clear_pipeline_context()
```

### Helper Auto-Detection

Each helper checks for pipeline context:

```python
def progress_bar(iterable, desc="Processing"):
    # Check if running in pipeline
    if _pipeline_context:
        progress_manager = _pipeline_context.get("progress_manager")
        if progress_manager:
            # Use pipeline's progress manager
            return pipeline_progress(iterable, desc, progress_manager)

    # Fallback to rich.track (tqdm-like)
    return track(iterable, description=desc)
```

### Function-to-Process Conversion

The `@pipeline_process` decorator dynamically creates a `PipelineProcess` class:

```python
@pipeline_process
def train(epochs: int = 10):
    # Function body...
    pass

# Decorator generates:
class TrainProcess(PipelineProcess):
    def __init__(self, controller, force=False, epochs=10):
        self.controller = controller
        self.epochs = epochs

    def execute(self):
        set_pipeline_context({...})
        try:
            return train(epochs=self.epochs)  # Call original function
        finally:
            clear_pipeline_context()
```

---

## Benefits

### For Researchers

1. **Zero friction start**: Just write normal Python
2. **Incremental enhancement**: Add features one at a time
3. **No forced migration**: Can stay at any level
4. **Copy-paste friendly**: Code works across levels
5. **Familiar tools**: tqdm-like progress bars, standard WandB

### For Production

1. **Full pipeline features**: When needed
2. **Config-driven**: Easy to manage experiments
3. **Testable**: Each level is testable
4. **Team collaboration**: Standardized structure
5. **CI/CD ready**: CLI integration

### For Both

1. **Same code**: Training logic doesn't change between levels
2. **Progressive complexity**: Match tool to task
3. **Smooth transition**: No rewrites needed
4. **Dual-mode helpers**: Work everywhere

---

## Migration Path

### Existing Scripts → Enhanced Scripts

```python
# Before
for epoch in range(10):
    loss = train()
    print(loss)

# After (add 3 lines)
from tipi.helpers import progress_bar, logger
logger.init(project="exp")

for epoch in progress_bar(range(10)):
    loss = train()
    logger.log({"loss": loss})
```

### Enhanced Scripts → Pipeline-Ready Functions

```python
# Add decorator (1 line)
@pipeline_process
def train(epochs: int = 10):
    # Same code as before!
    pass
```

### Pipeline-Ready Functions → Full Pipeline

```toml
# Create config.toml
[processes.training]
type = "train"  # Function name
params = { epochs = 10 }
```

---

## Comparison with Original Architecture

| Aspect              | Original                | With Progressive Enhancement |
| ------------------- | ----------------------- | ---------------------------- |
| Entry point         | CLI/Runner              | Plain Python script          |
| Initial complexity  | High (classes + config) | Zero (just helpers)          |
| Learning curve      | Steep                   | Gradual                      |
| Research workflow   | Forced into pipeline    | Natural script evolution     |
| Production workflow | Ready                   | Ready (same endpoint)        |
| Code reuse          | Medium                  | High (same code everywhere)  |
| Testing             | Full pipeline only      | Each level testable          |
| Team onboarding     | Full architecture       | Start with helpers           |

---

## Example: Real Research Scenario

### Week 1: New idea

```python
# quick_test.py
from tipi.helpers import progress_bar
for epoch in progress_bar(range(5)):
    print(train())
```

### Week 2: Looks promising

```python
# experiment_v1.py
from tipi.helpers import progress_bar, logger
logger.init(project="new_idea")
for epoch in progress_bar(range(10)):
    logger.log({"loss": train()})
```

### Week 3: Multiple variations

```python
# experiment_v2.py, v3.py, v4.py
# All using helpers - easy to compare in WandB
```

### Month 1: Extract common logic

```python
# train_utils.py
@pipeline_process
def train(learning_rate: float = 0.001):
    # Shared logic
    pass

# experiment_v5.py
from train_utils import train
train(learning_rate=0.01)
```

### Month 2: Production

```toml
# pipeline.toml
[processes.training]
type = "train"
params = { learning_rate = 0.001 }
```

**Same training code from Week 1 to Month 2!**

---

## Implementation Status

### Completed

- ✅ `helpers.py` module with progress_bar, logger, device_manager
- ✅ `decorators.py` module with @pipeline_process
- ✅ `docs/progressive_enhancement.md` full guide
- ✅ Architecture diagrams updated

### TODO (for full implementation)

- [ ] Integrate helper context setting in PipelineExecutor
- [ ] Implement pipeline progress integration in helpers
- [ ] Add tests for dual-mode helpers
- [ ] Create example scripts at each level
- [ ] Update existing processes to use helpers
- [ ] Add CLI flag to detect pipeline mode
- [ ] Create migration guide for existing pipelines

---

## Questions & Answers

**Q: Does this change the core architecture?**
A: No, it adds a new entry layer. Full pipeline architecture remains unchanged.

**Q: Can existing pipelines use helpers?**
A: Yes! Processes can import helpers and they'll automatically use permanences.

**Q: What if someone doesn't want helpers?**
A: Completely optional. Original architecture still works.

**Q: Performance impact?**
A: Minimal. Context checking is a simple dict lookup.

**Q: Does this work with existing permanences?**
A: Yes! Helpers just access them through context.

---

## Recommendation

Implement progressive enhancement in this order:

1. **Phase 1**: Helper module (standalone mode only)

   - Researchers can start using immediately
   - No pipeline changes needed
   - Low risk

2. **Phase 2**: Executor integration

   - Connect helpers to pipeline permanences
   - Existing pipelines work unchanged
   - Medium risk

3. **Phase 3**: Decorator support

   - Enable function-to-process conversion
   - Add to builder registration
   - Medium risk

4. **Phase 4**: Documentation & examples
   - Create example scripts at each level
   - Migration guides
   - Low risk

This allows incremental rollout with early value delivery.
