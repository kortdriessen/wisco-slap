This module handles files that track and maintain important meta data and related summary information for all experiments.

Specifically, the main files that we need to always be tracking are: 
- acquisition_master.yaml
- sb_scoring_times.yaml
- sync_info.yaml
- dmd_info.yaml
- prepro_info.yaml

Note that the ExSum mirror is also maintained in this module, as it is, in a way, a form of metadata - it is simply meant to maintain an exact copy of the ExperimentSummary files in the data_root, so that we can access a fast, local copy as needed.