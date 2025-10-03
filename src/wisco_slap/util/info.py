import polars as pl

import wisco_slap as wis
import wisco_slap.defs as DEFS


def get_unique_acquisitions_per_experiment(subject, exp):
    acqs = []
    ei = wis.pipes.exp_info.load_exp_info_spreadsheet()
    for loc in (
        ei.filter(pl.col("subject") == subject)
        .filter(pl.col("experiment") == exp)["location"]
        .unique()
    ):
        for acq in (
            ei.filter(pl.col("subject") == subject)
            .filter(pl.col("experiment") == exp)
            .filter(pl.col("location") == loc)["acquisition"]
            .unique()
        ):
            acqs.append(f"{loc}--{acq}")
    return acqs


def load_exp_info_spreadsheet():
    path = f"{DEFS.anmat_root}/exp_info.csv"
    return pl.read_csv(path)
