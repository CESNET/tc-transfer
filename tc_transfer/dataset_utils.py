import os
from functools import lru_cache, partial
from typing import Any, Optional

import numpy as np
import pandas as pd
import tcbench as tcb
from cesnet_datazoo.config import DatasetConfig as CesnetDatasetConfig
from cesnet_datazoo.constants import APP_COLUMN, PPI_COLUMN
from cesnet_datazoo.datasets import CESNET_TLS22
from sklearn.model_selection import train_test_split
from tcbench.libtcdatasets.mirage19_generate_splits import filter_dataset as mirage19_filter
from tcbench.libtcdatasets.mirage19_generate_splits import \
    generate_global_splits as mirage19_generate_splits
from tcbench.libtcdatasets.mirage22_generate_splits import \
    _filter_out_ack_packets as mirage22_filter1
from tcbench.libtcdatasets.mirage22_generate_splits import filter_dataset as mirage22_filter2
from tcbench.libtcdatasets.mirage22_generate_splits import \
    generate_global_splits as mirage22_generate_splits
from tcbench.libtcdatasets.utmobilenet21_generate_splits import _verify_splits
from tcbench.libtcdatasets.utmobilenet21_generate_splits import \
    filter_dataset as utmobilenet21_filter
from tcbench.libtcdatasets.utmobilenet21_generate_splits import \
    generate_splits as utmobilenet21_generate_splits

from experiment_wrapper.structured_config import DatasetLoader

TCBENCH_APP_COLUMN = "app"
PPI_MAX_LEN = 30
MIRAGE_COLUMNS_FOR_UNFILTERED = [
    "row_id",
    "app",
    "pkts_size",
    "pkts_dir",
    "timetofirst",
]

to_tcbench_enum = {
    "UCDAVIS19-Human": tcb.DATASETS.UCDAVISICDM19,
    "UCDAVIS19-Script": tcb.DATASETS.UCDAVISICDM19,
    "UTMOBILENET21": tcb.DATASETS.UTMOBILENET21,
    "MIRAGE19": tcb.DATASETS.MIRAGE19,
    "MIRAGE22": tcb.DATASETS.MIRAGE22,
}

def prepare_preload_data(dataset_preload_path: str, silent: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preload_data = np.load(dataset_preload_path, allow_pickle=True)
    if not silent: print(f"Using preloaded dataset from {dataset_preload_path}")
    return (
        preload_data["train_data"], preload_data["val_data"], preload_data["test_data"],
        preload_data["train_labels"], preload_data["val_labels"], preload_data["test_labels"],
    )

@lru_cache(maxsize=None)
def load_dataset(
    dataset_name: str,
    dataset_loader: DatasetLoader,
    split_id: int,
    preload_base: str,
    notcb_dataset_path: Optional[str] = None,
    iscxvpn_label_column: Optional[str] = None,
    random_split_val_test_fraction: Optional[float] = None,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset_loader == DatasetLoader.TCBENCH:
        if dataset_name.startswith("UCDAVIS19-"):
            dataset_preload_path = os.path.join(preload_base, f"{dataset_name}-{random_split_val_test_fraction}-{split_id}.npz")
        else:
            dataset_preload_path = os.path.join(preload_base, f"{dataset_name}-{split_id}.npz")
        if os.path.exists(dataset_preload_path):
            return prepare_preload_data(dataset_preload_path, silent=silent)
        if dataset_name == "UCDAVIS19-Human":
            ucdavis_test_set = "human"
            ucdavis_val_size = random_split_val_test_fraction
        elif dataset_name == "UCDAVIS19-Script":
            ucdavis_test_set = "script"
            ucdavis_val_size = random_split_val_test_fraction
        else:
            ucdavis_test_set = None
            ucdavis_val_size = 0.0
        assert isinstance(ucdavis_val_size, float)
        if not silent: print(f"Loading {dataset_name} dataset split {split_id} from tcbench")
        train_data, val_data, test_data, train_labels, val_labels, test_labels = load_tcbench_dataset(
            tcbench_enum=to_tcbench_enum[dataset_name],
            split_id=split_id,
            ucdavis_test_set=ucdavis_test_set,
            ucdavis_val_size=ucdavis_val_size,
        )
    elif dataset_loader == DatasetLoader.ISCXVPN2016:
        assert iscxvpn_label_column is not None and notcb_dataset_path is not None and random_split_val_test_fraction is not None
        dataset_preload_path = os.path.join(preload_base, f"ISCXVPN2016-{iscxvpn_label_column}-{random_split_val_test_fraction}-{split_id}.npz")
        if os.path.exists(dataset_preload_path):
            return prepare_preload_data(dataset_preload_path, silent=silent)
        if not silent: print(f"Loading ISCXVPN2016 dataset split {split_id} from parquet")
        train_data, val_data, test_data, train_labels, val_labels, test_labels = load_ISCXVPN2016_dataset(
            dataset_path=notcb_dataset_path,
            split_id=split_id,
            test_val_fraction=random_split_val_test_fraction,
            label_column=iscxvpn_label_column,
        )
    elif dataset_loader == DatasetLoader.CESNET_DATAZOO and dataset_name == "CESNET-TLS22":
        assert notcb_dataset_path is not None
        assert isinstance(train_size, int) and isinstance(val_size, int) and isinstance(test_size, int)
        dataset_preload_path = os.path.join(preload_base, f"CESNET-TLS22-{train_size}-{val_size}-{test_size}-{split_id}.npz")
        if os.path.exists(dataset_preload_path):
            return prepare_preload_data(dataset_preload_path, silent=silent)
        if not silent: print(f"Loading CESNET-TLS22 dataset split {split_id} from datazoo")
        train_data, val_data, test_data, train_labels, val_labels, test_labels = load_cesnet_tls22_from_datazoo(
            dataset_path=notcb_dataset_path,
            split_id=split_id,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
    elif dataset_loader == DatasetLoader.APPCLASSNET:
        assert notcb_dataset_path is not None
        assert isinstance(train_size, int) and isinstance(val_size, int) and isinstance(test_size, int)
        dataset_preload_path = os.path.join(preload_base, f"AppClassNet-{train_size}-{val_size}-{test_size}-{split_id}.npz")
        if os.path.exists(dataset_preload_path):
            return prepare_preload_data(dataset_preload_path, silent=silent)
        if not silent: print(f"Loading AppClassNet dataset split {split_id} from source data")
        train_data, val_data, test_data, train_labels, val_labels, test_labels = load_appclassnet(
            dataset_path=notcb_dataset_path,
            split_id=split_id,
            resplit=True,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")
    if dataset_preload_path is not None:
        np.savez(
            dataset_preload_path,
            train_data=train_data, val_data=val_data, test_data=test_data,
            train_labels=train_labels, val_labels=val_labels, test_labels=test_labels,
        )
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

# Load from tcbench
#
def tcbench_process_ppi(row, is_utmobilenet: bool = False):
    directions = np.where(row["pkts_dir"] == 0, -1, 1)
    sizes = row["pkts_size"]
    if is_utmobilenet:
        # For UTMOBILENET21, time differences are in the "timetofirst" column
        time_differences = row["timetofirst"].copy()
    else:
        time_differences = np.diff(row["timetofirst"], prepend=0)
        assert len(directions) == len(sizes) == len(time_differences)
        assert np.isclose(time_differences.cumsum(), row["timetofirst"]).all()
        if "pkts_iat" in row:
            assert np.isclose(time_differences, row["pkts_iat"]).all()
        time_differences[0] = 0.0
    if "duration" in row:
        assert np.isclose(row["duration"], time_differences.sum())
    time_differences = np.round(time_differences * 1000).astype(int) # convert to ms
    ppi = (time_differences, directions, sizes)
    ppi = np.array(ppi)[:, :PPI_MAX_LEN]
    ppi = np.pad(ppi, pad_width=((0, 0), (0, PPI_MAX_LEN - len(ppi[0]))))
    return ppi

def load_tcbench_dataset(tcbench_enum: tcb.DATASETS,
                         split_id: int,
                         filter_min_packets: int = 10,
                         ucdavis_test_set: Optional[str] = None,
                         ucdavis_val_size: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tcbench_enum == tcb.DATASETS.UCDAVISICDM19:
        if ucdavis_test_set is None:
            raise ValueError("ucdavis_test_set must be either 'script' or 'human' when using the UCDAVIS19 dataset")
        df = tcb.load_parquet(tcbench_enum)
        df_pretraining = df[df["partition"] == "pretraining"]
        if ucdavis_val_size > 0:
            train_indices, val_indices = train_test_split(
                df_pretraining.index,
                test_size=ucdavis_val_size,
                random_state=42 + split_id,
                stratify=df_pretraining[TCBENCH_APP_COLUMN],
            )
            df_train = df_pretraining.loc[train_indices]
            df_val = df_pretraining.loc[val_indices]
        else:
            df_train = df_pretraining
            df_val: Any = pd.DataFrame(columns=df_train.columns)
        df_test = tcb.load_parquet(tcbench_enum, split=ucdavis_test_set)
    else:
        if filter_min_packets == 10: # default value for which splits are available
            df = tcb.load_parquet(tcbench_enum, min_pkts=10)
            df_splits = tcb.load_parquet(tcbench_enum, min_pkts=10, split=True) # type: ignore
        elif filter_min_packets == 0: # get unfiltered datasets
            if tcbench_enum == tcb.DATASETS.UTMOBILENET21:
                df = tcb.load_parquet(tcbench_enum, min_pkts=-1)
                df = utmobilenet21_filter(df, min_pkts=0)
                df_splits = utmobilenet21_generate_splits(df)
            elif tcbench_enum == tcb.DATASETS.MIRAGE19:
                df = tcb.load_parquet(tcbench_enum, min_pkts=-1)
                df = mirage19_filter(df, min_pkts=0)
                df = df[MIRAGE_COLUMNS_FOR_UNFILTERED]
                df_splits = mirage19_generate_splits(df)
            elif tcbench_enum == tcb.DATASETS.MIRAGE22:
                df = tcb.load_parquet(tcbench_enum, min_pkts=-1, columns=["row_id", "packet_data_l4_payload_bytes", "packet_data_iat", "packet_data_packet_dir", "app"])
                df = mirage22_filter1(df)
                df = mirage22_filter2(df, min_pkts=0)
                df = df[MIRAGE_COLUMNS_FOR_UNFILTERED]
                df_splits = mirage22_generate_splits(df)
        else:
            raise ValueError("filter_min_packets must be either 10 or 0")
        _verify_splits(df, df_splits)
        split_indices = df_splits.iloc[split_id]
        train_incides, val_indices, test_indices = split_indices["train_indexes"], split_indices["val_indexes"], split_indices["test_indexes"]
        df_train, df_val, df_test = df.iloc[train_incides], df.iloc[val_indices], df.iloc[test_indices]
    ppi_fn = partial(tcbench_process_ppi, is_utmobilenet=tcbench_enum==tcb.DATASETS.UTMOBILENET21)
    train_data = np.stack(df_train.apply(ppi_fn, axis=1))
    val_data = np.stack(df_val.apply(ppi_fn, axis=1)) if len(df_val) > 0 else np.zeros((0, 3, PPI_MAX_LEN))
    test_data = np.stack(df_test.apply(ppi_fn, axis=1))
    train_labels = df_train[TCBENCH_APP_COLUMN].to_numpy()
    val_labels = df_val[TCBENCH_APP_COLUMN].to_numpy()
    test_labels = df_test[TCBENCH_APP_COLUMN].to_numpy()
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

#
# Load the ISCXVPN2016 dataset
#
def ISCXVPN2016_process_ppi(row):
    if len(row["packet_dir"]) == 0:
        return np.zeros((3, PPI_MAX_LEN)), 0
    sizes = row["L4_payload_bytes"]
    directions = 2 * row["packet_dir"] - 1
    time_differences = np.round(row["iat"] * 1000).astype(int) # convert to ms
    ppi = (time_differences, directions, sizes)
    ppi = np.array(ppi)[:, :PPI_MAX_LEN]
    ppi = np.pad(ppi, pad_width=((0, 0), (0, PPI_MAX_LEN - len(ppi[0]))))
    return ppi, len(row["packet_dir"])

def load_ISCXVPN2016_dataset(dataset_path: str,
                             label_column: str,
                             split_id: int = 0,
                             test_val_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(dataset_path)
    df[["PPI", "PPI_LEN"]] = df.apply(ISCXVPN2016_process_ppi, axis=1, result_type="expand")
    assert (df.PPI_LEN > 0).all()
    keep_columns = ["PPI", label_column]
    df = df[keep_columns]
    df = df.rename(columns={label_column: "LABEL"}) # type: ignore

    train_indices, test_indices = train_test_split(
        df.index,
        test_size=test_val_fraction,
        random_state=42 + split_id,
        stratify=df.LABEL,
    )
    val_size_adjusted = test_val_fraction / (1 - test_val_fraction)
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=val_size_adjusted,
        random_state=43 + split_id,
        stratify=df.loc[train_indices].LABEL,
    )
    train_data = np.stack(df.loc[train_indices].PPI)
    val_data = np.stack(df.loc[val_indices].PPI)
    test_data = np.stack(df.loc[test_indices].PPI)
    train_labels = df.loc[train_indices].LABEL.to_numpy()
    val_labels = df.loc[val_indices].LABEL.to_numpy()
    test_labels = df.loc[test_indices].LABEL.to_numpy()
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

#
# Load the CESNET-TLS22 dataset
#
def load_cesnet_tls22_from_datazoo(dataset_path: str,
                                   dataset_size: str = "S",
                                   split_id: int = 0,
                                   train_size: int = 1_000_000,
                                   val_size: int = 100_000,
                                   test_size: int = 1_000_000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert test_size < 10_000_000
    dataset = CESNET_TLS22(data_root=dataset_path, size=dataset_size, silent=split_id != 0)
    dataset_config = CesnetDatasetConfig(
        dataset=dataset,
        fold_id=split_id,
        batch_size=16384,
        test_batch_size=16384,
        train_period_name="W-2021-40",
        test_period_name="W-2021-41",
        train_size=train_size,
        val_known_size=val_size,
        test_known_size="all",
        train_workers=0,
        val_workers=0,
        test_workers=0,
        need_val_set=True,)
    dataset.set_dataset_config_and_initialize(dataset_config)
    assert dataset.class_info is not None
    df_train = dataset.get_train_df()
    df_val = dataset.get_val_df()
    df_test = dataset.get_test_df().sample(test_size, random_state=42 + split_id)
    train_data = np.stack(df_train[PPI_COLUMN]) # type: ignore
    val_data = np.stack(df_val[PPI_COLUMN]) # type: ignore
    test_data =  np.stack(df_test[PPI_COLUMN]) # type: ignore
    train_labels = dataset.class_info.encoder.inverse_transform(df_train[APP_COLUMN])
    val_labels = dataset.class_info.encoder.inverse_transform(df_val[APP_COLUMN])
    test_labels = dataset.class_info.encoder.inverse_transform(df_test[APP_COLUMN])
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

#
# AppClassNet
#
def load_appclassnet(dataset_path: str,
                     split_id: int = 0,
                     train_size: Optional[int] = None,
                     val_size: Optional[int] = None,
                     test_size: Optional[int] = None,
                     resplit: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_data = np.load(os.path.join(dataset_path, "train_x.npy"))
    train_data = np.stack((np.zeros_like(train_data), np.sign(train_data), np.abs(train_data) * 3000), axis=1)
    train_data = np.pad(train_data, ((0, 0), (0, 0), (0, 10)))
    train_labels = np.load(os.path.join(dataset_path, "train_y.npy"))

    val_data = np.load(os.path.join(dataset_path, "valid_x.npy"))
    val_data = np.stack((np.zeros_like(val_data), np.sign(val_data), np.abs(val_data) * 3000), axis=1)
    val_data = np.pad(val_data, ((0, 0), (0, 0), (0, 10)))
    val_labels = np.load(os.path.join(dataset_path, "valid_y.npy"))

    test_data = np.load(os.path.join(dataset_path, "test_x.npy"))
    test_data = np.stack((np.zeros_like(test_data), np.sign(test_data), np.abs(test_data) * 3000), axis=1)
    test_data = np.pad(test_data, ((0, 0), (0, 0), (0, 10)))
    test_labels = np.load(os.path.join(dataset_path, "test_y.npy"))

    if resplit:
        # Combine train and val and resplit
        combined_data = np.concatenate((train_data, val_data))
        combined_labels = np.concatenate((train_labels, val_labels))
        train_indices, val_indices = train_test_split(
            np.arange(len(combined_data)),
            train_size=train_size,
            test_size=val_size, random_state=42 + split_id,
            stratify=combined_labels,
        )
        train_data = combined_data[train_indices]
        val_data = combined_data[val_indices]
        train_labels = combined_labels[train_indices]
        val_labels = combined_labels[val_indices]

        rng = np.random.default_rng(43 + split_id)
        if test_size is not None and test_size < len(test_data):
            test_indices = rng.choice(len(test_data), test_size, replace=False)
            test_data = test_data[test_indices]
            test_labels = test_labels[test_indices]

    return train_data, val_data, test_data, train_labels, val_labels, test_labels
