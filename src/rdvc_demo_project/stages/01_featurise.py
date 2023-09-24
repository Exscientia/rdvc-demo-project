from functools import partial

import dvc.api
from tdc.chem_utils.featurize.molconvert import smiles2morgan
from tdc.single_pred import ADME

from rdvc_demo_project.utils import get_git_root


def main() -> None:
    config = dvc.api.params_show()

    # Download dataset
    data = ADME(name="Solubility_AqSolDB")
    split_data = data.get_split()

    # Featurise and save dataset splits
    fingerprint_factory = partial(smiles2morgan, **config["fingerprint"])
    dataset_dir = get_git_root() / "data/featurised"
    dataset_dir.mkdir(exist_ok=True)

    for split, df in split_data.items():
        df["fingerprint"] = df["Drug"].map(fingerprint_factory)
        df.to_parquet(dataset_dir / f"{split}.parquet", index=False)


if __name__ == "__main__":
    main()
