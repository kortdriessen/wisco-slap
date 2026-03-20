rsync -avz --dry-run --exclude='.venv' driessen2@ad.wisc.edu@tononi-2:/Volumes/npx_nfs/slap/slap_mi_in_the_pupil/ /data/slap_analysis/slap_mi_in_the_pupil/

rsync -avz --dry-run --exclude='.venv' --exclude='pupil_training_videos' driessen2@ad.wisc.edu@tononi-2:/Volumes/npx_nfs/slap/slap_mi_in_the_pupil/ /data/slap_analysis/slap_mi_in_the_pupil/

rsync -avz --exclude='.venv' --exclude='pupil_training_videos' driessen2@ad.wisc.edu@tononi-2:/Volumes/npx_nfs/slap/slap_mi_in_the_pupil/ /data/slap_analysis/slap_mi_in_the_pupil/