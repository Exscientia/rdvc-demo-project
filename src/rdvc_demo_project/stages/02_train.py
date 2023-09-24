import dvc.api
import lightning as L
import numpy as np
import pandas as pd
import torch
from dvclive import Live
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rdvc_demo_project.utils import get_git_root


def get_model(in_channels: int, hidden_channels: list[int]) -> torch.nn.Module:
    layers: list[nn.Module] = [nn.Linear(in_channels, hidden_channels[0])]
    last_channel_size = hidden_channels[0]

    for out_channel_size in hidden_channels[1:]:
        layers.append(nn.ReLU())
        layers.append(nn.Linear(last_channel_size, out_channel_size))
        last_channel_size = out_channel_size

    layers.append(nn.ReLU())
    layers.append(nn.Linear(last_channel_size, 1))

    return nn.Sequential(*layers)


def main() -> None:
    config = dvc.api.params_show()

    dataset_dir = get_git_root() / "data/featurised"
    adme_data = pd.read_parquet(dataset_dir / "train.parquet", engine="pyarrow")
    torch.tensor(np.array(adme_data["fingerprint"].to_list()))
    torch.from_numpy(adme_data["Y"].to_numpy())

    fingerprints = torch.tensor(np.array(adme_data["fingerprint"].to_list())).type("torch.FloatTensor")
    solubilities = torch.from_numpy(adme_data["Y"].to_numpy()).type("torch.FloatTensor")

    dataset = TensorDataset(fingerprints, solubilities)

    # Featurise and save dataset splits
    model = get_model(**config["model"])

    fabric = L.Fabric(**config["fabric"])
    fabric.launch()

    dataloader = DataLoader(dataset, batch_size=config["batch_size"])
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

    model.train()

    with Live(str(get_git_root() / "training")) as live:
        for _ in range(config["num_epochs"]):
            for batch in dataloader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = torch.nn.functional.mse_loss(outputs.squeeze(), targets.view(-1))
                live.log_metric("train/loss", loss.item())
                fabric.backward(loss)

                optimizer.step()


if __name__ == "__main__":
    main()
