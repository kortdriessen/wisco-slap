import wisco_slap as wis
import wisco_slap.defs as DEFS
import pandas as pd
import os

def load_auto_hypno(subject, exp, sync_block, filter_unclear: bool = None, filter_on='smooth'):
    hypno_dir = f'{DEFS.anmat_root}/{subject}/{exp}/scoring_data/sync_block-{sync_block}/hypnograms/model_labelled'
    hypno_name = f'raw_epochs.csv'
    epoch_df = pd.read_csv(f'{hypno_dir}/{hypno_name}')
    if filter_unclear is not None:
        if filter_on == 'smooth':
            epoch_df['label'] = epoch_df['label'].mask(epoch_df[['NREM_smooth', 'REM_smooth', 'Wake_smooth']].max(axis=1) < filter_unclear, 'unclear')
        elif filter_on == 'raw':
            epoch_df['label'] = epoch_df['label'].mask(epoch_df[['P_NREM', 'P_REM', 'P_Wake']].max(axis=1) < filter_unclear, 'unclear')
        else:
            raise ValueError(f"Invalid filter_on: {filter_on}")
    return epoch_df