# Progressive Enhancement: From Script to Pipeline

This guide shows how to smoothly transition from experimental scripts to production-ready pipelines.

## The Journey: 5 Levels of Enhancement

### Level 0: Raw Script (Start Here!)

You're a researcher with an idea. Start with a normal Python script:

```python
# train.py
import torch
from torch.utils.data import DataLoader
from my_model import MyModel

# Simple training script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=32)

print("Starting training...")
for epoch in range(10):
    print(f"Epoch {epoch+1}/10")
    for i, batch in enumerate(dataloader):
        loss = train_step(model, batch, device)
        if i % 10 == 0:
            print(f"Batch {i}, Loss: {loss:.4f}")
```

**Pros**: Quick to write, easy to experiment
**Cons**: No progress bars, no logging, hard to track experiments

---

### Level 1: Add Progress Bars

Want to see progress? Just add one import:

```python
# train.py
import torch
from torch.utils.data import DataLoader
from my_model import MyModel
from tipi.helpers import progress_bar  # ← Add this

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=32)

# Add progress bars with minimal changes
for epoch in progress_bar(range(10), desc="Epochs"):  # ← Changed
    for batch in progress_bar(dataloader, desc="Training"):  # ← Changed
        loss = train_step(model, batch, device)
```

**What you get**:

- Beautiful progress bars (via Rich)
- Automatic time estimation
- Works exactly like `tqdm`
- No pipeline required!

**Changes needed**: 2 lines!

---

### Level 2: Add Experiment Logging

Want to track your experiments in WandB?

```python
# train.py
import torch
from torch.utils.data import DataLoader
from my_model import MyModel
from tipi.helpers import progress_bar, logger  # ← Add logger

# Initialize logging (runs once)
logger.init(project="my_research", entity="my_team")  # ← Add this

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyModel().to(device)
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=32)

for epoch in progress_bar(range(10), desc="Epochs"):
    epoch_loss = 0
    for batch in progress_bar(dataloader, desc="Training"):
        loss = train_step(model, batch, device)
        epoch_loss += loss
        logger.log({"batch_loss": loss})  # ← Add this

    logger.log({"epoch_loss": epoch_loss / len(dataloader)})  # ← Add this
```

**What you get**:

- Automatic WandB logging
- Experiment tracking
- Metric visualization
- Still just a script!

**Changes needed**: 3 lines!

---

### Level 3: Better Device Management

Tired of CUDA boilerplate?

```python
# train.py
import torch
from torch.utils.data import DataLoader
from my_model import MyModel
from tipi.helpers import (
    progress_bar,
    logger,
    device_manager  # ← Add this
)

logger.init(project="my_research", entity="my_team")

# Let the helper pick the best device
device = device_manager.get_device()  # ← Simplified!
model = MyModel().to(device)
dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=32)

for epoch in progress_bar(range(10), desc="Epochs"):
    for batch in progress_bar(dataloader, desc="Training"):
        loss = train_step(model, batch, device)
        logger.log({"loss": loss})
```

**What you get**:

- Automatic best device selection
- Handles multi-GPU scenarios
- Checks VRAM availability
- Still just helpers!

---

### Level 4: Extract Reusable Functions

Your script is working great! Now extract functions for reuse:

```python
# train.py
import torch
from torch.utils.data import DataLoader
from my_model import MyModel
from tipi.helpers import progress_bar, logger, device_manager
from tipi.decorators import pipeline_process  # ← New!

logger.init(project="my_research", entity="my_team")

def train_step(model, batch, device):
    """Your existing training logic"""
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    return loss.item()

@pipeline_process  # ← Add decorator (optional for now)
def train_model(epochs: int = 10):
    """Now this function can become a pipeline process later!"""
    device = device_manager.get_device()
    model = MyModel().to(device)
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=32)

    for epoch in progress_bar(range(epochs), desc="Epochs"):
        for batch in progress_bar(dataloader, desc="Training"):
            loss = train_step(model, batch, device)
            logger.log({"loss": loss})

    return model

# Still works as a script!
if __name__ == "__main__":
    trained_model = train_model(epochs=10)
    torch.save(trained_model.state_dict(), "model.pth")
```

**What you get**:

- Reusable functions
- Ready for pipeline conversion
- Still runs as a script
- Type hints for parameters

**The decorator**: `@pipeline_process` doesn't change behavior yet, but makes the function pipeline-ready!

---

### Level 5: Convert to Full Pipeline

Ready for production? Create a config file and you're done!

#### Step 1: Organize your code

```python
# processes/training.py
from tipi import PipelineProcess
from tipi.helpers import progress_bar, logger

class TrainingProcess(PipelineProcess):
    def __init__(self, controller, force: bool = False, epochs: int = 10):
        super().__init__(controller, force)
        self.epochs = epochs

    def execute(self):
        # Get permanences from pipeline
        device = self.controller.get_permanence("device")
        model = self.controller.get_permanence("network").model
        dataloader = self.controller.get_permanence("data").train_loader

        # Your SAME training logic from Level 4!
        for epoch in progress_bar(range(self.epochs), desc="Epochs"):
            for batch in progress_bar(dataloader, desc="Training"):
                loss = self.train_step(model, batch, device)
                logger.log({"loss": loss})

        return None

    def train_step(self, model, batch, device):
        """Copy from your script - no changes!"""
        # ... same code ...
```

#### Step 2: Create config

```toml
# configs/my_pipeline/execute_pipeline.toml

[permanences.device]
type = "Device"

[permanences.network]
type = "Network"
params = { model = "resnet50", num_classes = 10, pretrained = true }

[permanences.data]
type = "Data"
params = { dataset = "CIFAR10", batch_size = 32 }

[permanences.progress_manager]
type = "ProgressManager"

[permanences.wandb_logger]
type = "WandBManager"
params = { project = "my_research", entity = "my_team" }

[processes.training]
type = "TrainingProcess"
params = { epochs = 10 }

[processes.validation]
type = "ValidationProcess"

[processes.testing]
type = "TestingProcess"
```

#### Step 3: Run via CLI

```bash
# Run the full pipeline
tipi run my_pipeline

# Or with different config
tipi run my_pipeline --config custom_config.toml

# Or programmatically
python -c "
from tipi import PipelineRunner
runner = PipelineRunner('my_pipeline')
runner.run()
"
```

**What you get**:

- Full pipeline management
- Config-driven experiments
- Nested progress bars
- Permanence lifecycle management
- WandB sweep support
- Production-ready code
- Testable components

---

## Comparison Table

| Feature            | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| ------------------ | ------- | ------- | ------- | ------- | ------- | ------- |
| Lines of code      | 15      | 17      | 20      | 20      | 30      | 40+     |
| Progress bars      | ❌      | ✅      | ✅      | ✅      | ✅      | ✅      |
| Experiment logging | ❌      | ❌      | ✅      | ✅      | ✅      | ✅      |
| Device management  | Manual  | Manual  | Manual  | ✅      | ✅      | ✅      |
| Reusable code      | ❌      | ❌      | ❌      | ❌      | ✅      | ✅      |
| Config-driven      | ❌      | ❌      | ❌      | ❌      | ❌      | ✅      |
| Pipeline features  | ❌      | ❌      | ❌      | ❌      | ❌      | ✅      |
| Testable           | ❌      | ❌      | ❌      | ❌      | ⚠️      | ✅      |
| Production-ready   | ❌      | ❌      | ❌      | ❌      | ⚠️      | ✅      |

---

## Key Principles

### 1. **Zero Friction Start**

- Researchers start with normal scripts
- No framework knowledge required
- No config files needed initially

### 2. **Progressive Enhancement**

- Each level adds ONE new concept
- Previous levels still work
- No forced migration

### 3. **Gradual Type Safety**

- Start untyped (scripts)
- Add types when extracting functions (Level 4)
- Full type safety in pipeline (Level 5)

### 4. **Helpers Work Everywhere**

- `progress_bar()` works in scripts and pipelines
- `logger` works standalone and with pipeline
- `device_manager` adapts to context

### 5. **Copy-Paste Friendly**

- Training logic from Level 3 works in Level 5
- Functions can be copied between scripts and processes
- Minimal refactoring needed

---

## Example: Real Research Workflow

### Day 1: New Idea

```python
# quick_experiment.py
for epoch in range(5):
    loss = train()
    print(loss)
```

### Day 2: Looks Promising

```python
# quick_experiment.py
from tipi.helpers import progress_bar, logger

logger.init(project="new_idea")

for epoch in progress_bar(range(10)):
    loss = train()
    logger.log({"loss": loss})
```

### Week 1: Multiple Experiments

```python
# experiment_v1.py, experiment_v2.py, experiment_v3.py
# All using helpers - easy to compare in WandB
```

### Week 2: Extract Common Code

```python
# train_utils.py
@pipeline_process
def train_model(learning_rate: float = 0.001):
    # Shared training logic
    pass

# experiment_v4.py
from train_utils import train_model
train_model(learning_rate=0.01)
```

### Month 1: Production Pipeline

```toml
# configs/production/pipeline.toml
[processes.training]
type = "train_model"  # Your function!
params = { learning_rate = 0.001 }
```

```bash
tipi run production
```

---

## Migration Checklist

Moving from Level N to Level N+1:

### Level 0 → 1: Add Progress Bars

- [ ] Import `progress_bar` from helpers
- [ ] Wrap your loops with `progress_bar()`
- [ ] Add descriptive names

### Level 1 → 2: Add Logging

- [ ] Import `logger` from helpers
- [ ] Call `logger.init()` once at start
- [ ] Add `logger.log()` calls for metrics

### Level 2 → 3: Better Device Management

- [ ] Import `device_manager` from helpers
- [ ] Replace device selection with `device_manager.get_device()`

### Level 3 → 4: Extract Functions

- [ ] Identify reusable code blocks
- [ ] Extract into functions with type hints
- [ ] Add `@pipeline_process` decorator
- [ ] Test standalone execution

### Level 4 → 5: Full Pipeline

- [ ] Create `processes/` directory
- [ ] Convert functions to `PipelineProcess` classes
- [ ] Create TOML config file
- [ ] Register in pipeline builder
- [ ] Test via CLI

---

## Tips for Researchers

### When to move to the next level?

**Level 0 → 1**: When you're tired of seeing `print()` statements
**Level 1 → 2**: When you lose track of which experiment was which
**Level 2 → 3**: When GPU selection becomes annoying
**Level 3 → 4**: When you copy-paste code between experiments
**Level 4 → 5**: When you need to run in production or share with team

### You don't have to reach Level 5!

- Many experiments stay at Level 2-3
- Only productionize what's proven to work
- Keep prototyping at lower levels

### Mix and match

```python
# experiment_hybrid.py
from tipi.helpers import progress_bar, logger
from my_pipeline.processes.training import train_epoch  # ← From Level 5

# Quick experiment using production code
for epoch in progress_bar(range(3)):
    loss = train_epoch(my_model, data)  # ← Production function
    logger.log({"quick_test": loss})    # ← Script logging
```

---

## Questions?

- **Q: Do I have to use all helpers?**
  A: No! Use only what you need. `progress_bar` alone is useful.

- **Q: Can I use this with my existing code?**
  A: Yes! Just import helpers and start adding them incrementally.

- **Q: What if I don't want WandB?**
  A: `logger` will gracefully degrade to console output.

- **Q: Can I stay at Level 3 forever?**
  A: Absolutely! Only move to Level 5 when you need production features.

- **Q: Does this work with my existing tqdm code?**
  A: Yes! `progress_bar` is a drop-in replacement for `tqdm`.
