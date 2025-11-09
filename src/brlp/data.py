import os
from typing import Optional, Union

import pandas as pd
from monai.data import Dataset, PersistentDataset
from monai.transforms import Transform  # 較新寫法

def get_dataset_from_pd(
    df: pd.DataFrame,
    transforms_fn: Transform,
    cache_dir: Optional[str] = None
) -> Union[Dataset, PersistentDataset]:
    """
    If `cache_dir` is provided, returns a `monai.data.PersistentDataset`.
    Otherwise, returns a standard `monai.data.Dataset`.

    Args:
        df (pd.DataFrame): Dataframe describing each image/sample in the dataset.
                           It will be converted to a list[dict] and passed to MONAI.
        transforms_fn (Transform): Composed/transforms pipeline.
        cache_dir (Optional[str]): Cache directory for PersistentDataset. If provided,
                                   it will be created automatically when not exists.

    Returns:
        Union[Dataset, PersistentDataset]: The MONAI dataset instance.

    Raises:
        ValueError: If df is empty or transforms_fn is None.
    """
    if df is None or len(df) == 0:
        raise ValueError("`df` is empty. Please provide a non-empty DataFrame.")
    if transforms_fn is None:
        raise ValueError("`transforms_fn` must not be None.")

    # 轉成 MONAI 期望的 data list[dict]
    data = df.to_dict(orient="records")

    # 如果給了 cache_dir，就自動建立；沒給就用一般 Dataset
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        return PersistentDataset(data=data, transform=transforms_fn, cache_dir=cache_dir)
    else:
        return Dataset(data=data, transform=transforms_fn)
