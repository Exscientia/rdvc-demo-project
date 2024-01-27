import dvc.api
from molflux.datasets import load_dataset, save_dataset_to_store, split_dataset, featurise_dataset
from molflux import splits, features

from rdvc_demo_project.utils import get_git_root


def main() -> None:
    config = dvc.api.params_show()

    # Download dataset
    dataset = load_dataset("esol")

    # Split dataset
    shuffle_strategy = splits.load_from_dict(config["split"])
    split_dataset_ = next(split_dataset(dataset, shuffle_strategy))

    # Featurise and save dataset splits
    featuriser = features.load_from_dict(config["features"])
    featurised_dataset = featurise_dataset(split_dataset_, column="smiles", representations=featuriser)

    # fingerprint_factory = partial(smiles2morgan, **config["fingerprint"])
    dataset_dir = get_git_root() / "data/featurised"
    dataset_dir.mkdir(exist_ok=True)
    save_dataset_to_store(featurised_dataset, dataset_dir, format="parquet", compression="gzip")


if __name__ == "__main__":
    main()
