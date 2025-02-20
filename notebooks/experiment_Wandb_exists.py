import wandb

api = wandb.Api()

entity = "lit-rvc"
project = "sam2segnet"

try:
    getted_project = api.project(f"{entity}/{project}")
    print(f"Project '{project}' exists under entity '{entity}'.")
    print(getted_project)
except wandb.errors.CommError:
    print(f"Project '{project}' does not exist or is inaccessible.")
