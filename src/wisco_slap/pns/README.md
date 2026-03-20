This module is named 'pns' for "Process and Save". It contains scripts that run pipelines where some processing event occurs once, and then the result is saved.

Ideally, these scripts and functions should all really only be used internally within this module, and not as a part of exploratory data analysis, etc.

Currently, the main and fully-verified scripts are:
- `sync_block_dat.py` - This handles the saving of data associated with a single sync block, namely:
  - the ephys data for that sync block
  - the video frame times (in ephys time base) for that sync block
  - the deeplabcut inference on the video for that sync block
  - the eye_metrics dataframe (derived from the deeplabcut inference) for that sync block
  - the whisker dataframe (derived from a manual mask drawn on the video indicating the whiskers) for that sync block
- `scopex_gen.py` - This handles the saving of fluorescence data associated with a single SLAP2 acquisition (as xarray DataArrays), namely:
  - the synaptic traces (dF) and FO values for both channels
  - the ROI traces, both F and Fsvd, for both channels