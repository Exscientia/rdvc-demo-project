from datasets import DatasetDict
import dvc.api
from molflux import modelzoo as mz
from molflux.datasets import load_dataset_from_store

from rdvc_demo_project.utils import get_git_root


def main() -> None:
    config = dvc.api.params_show()

    # Load dataset
    dataset_dir = get_git_root() / "data/featurised"
    split_dataset = load_dataset_from_store(dataset_dir)
    assert isinstance(split_dataset, DatasetDict)

    # DVCLive deposits training metrics into this directory
    training_dir = get_git_root() / "metrics"
    training_dir.mkdir(exist_ok=True)

    # Train model
    model = mz.load_from_dict(config["model"])
    model.train(train_data=split_dataset["train"])

    # Save model
    model_dir = get_git_root() / "model"
    model_dir.mkdir(exist_ok=True)
    mz.save_to_store(model_dir, model)


if __name__ == "__main__":
    main()
