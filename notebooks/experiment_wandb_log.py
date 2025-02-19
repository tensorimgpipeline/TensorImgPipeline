import os

import torch

import wandb

# os.environ["WANDB_SILENT"] = "true"
# Initialize wandb
wandb.init(entity="lit-rvc", project="TEST-LOGGING", name="experiment_3")

# Dummy training setup
epochs = 10
train_loader = [torch.randn(32, 10) for _ in range(100)]  # 100 batches of 32 samples
val_loader = [torch.randn(32, 10) for _ in range(20)]  # 20 batches of 32 samples
test_loader = [torch.randn(32, 10) for _ in range(10)]  # 10 batches of 32 samples

# Dummy model and loss function
model = torch.nn.Linear(10, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Global step counter for unified logging
global_step = 0

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, torch.randn_like(output))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log per iteration loss with a global step
        global_step += idx  # Increment global step
        wandb.log({"train_loss": loss.item()}, step=global_step)

    # Mean epoch loss
    train_epoch_loss = total_loss / len(train_loader)

    # Validation loss calculation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            output = model(batch)
            loss = loss_fn(output, torch.randn_like(output))
            val_loss += loss.item()
            global_step += idx  # Increment global step
            wandb.log({"val_loss": loss.item()}, step=global_step)

    val_loss /= len(val_loader)
    wandb.log({"train_epoch_loss": train_epoch_loss, "val_epoch_loss": val_loss}, step=global_step)

# Final Test Loss
model.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        output = model(batch)
        loss = loss_fn(output, torch.randn_like(output))
        test_loss += loss.item()

test_loss /= len(test_loader)
wandb.log({"test_loss": test_loss}, step=global_step)

wandb.finish()
