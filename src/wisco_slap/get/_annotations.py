import polars as pl


def hypno_csv(path: str, reformat: bool = True) -> pl.DataFrame:
    """Load an annotation CSV file into a polars DataFrame."""
    df = pl.read_csv(path)
    if reformat:
        if not all(col in df.columns for col in ["start_s", "end_s", "label"]):
            if all(
                col in df.columns
                for col in ["start_time", "end_time", "state", "duration"]
            ):
                # formatting is already done
                return df
            else:
                raise ValueError(
                    f"Format of csv not recognized, try without reformat: {path}"
                )
        else:
            # reformat the csv
            df = df.rename({
                "label": "state",
                "start_s": "start_time",
                "end_s": "end_time",
            })
            df = df.with_columns(
                (pl.col("end_time") - pl.col("start_time")).alias("duration")
            )
            df = df.sort("start_time")
    return df
