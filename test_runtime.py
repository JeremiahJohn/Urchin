from urchin import Urchin
import numpy as np
import pickle

urchin = Urchin(
            path_to_sorted_data='/home/mxwbio/Desktop/Sorting/20210716_2b',
            path_to_raw_data='/home/mxwbio/Data/recordings/2021_07_16_2.raw.h5',
            path_to_json='/home/mxwbio/Desktop/Analysis/urchin/20210716_2_completed.json')

# urchin.stim_bounds List of separated stimuli.
# to split sub-stimuli, use split_into_sub_stimuli
flash_stim_sub = urchin.split_into_sub_stimuli(*urchin.stim_bounds[4])
bar_stim_sub = urchin.split_into_sub_stimuli(*urchin.stim_bounds[3])
check_stim_sub = urchin.split_into_sub_stimuli(*urchin.stim_bounds[-1])
# extract spike times within each sub_stim feature.
# flash_stim_sub_spikes = urchin.extract_spiketimes_within_bounds(
#                             cluster_id=urchin.extracted_clusters,
#                             bounds=flash_stim_sub)
# bar_stim_sub_spikes = urchin.extract_spiketimes_within_bounds(
#                             cluster_id=urchin.extracted_clusters,
#                             bounds=bar_stim_sub)
check_stim_sub_spikes = urchin.extract_spiketimes_within_bounds(
                            cluster_id=urchin.extracted_clusters,
                            bounds=check_stim_sub)
# Given that stim_sub_spikes is a dict with entire stim details
# This finds PSTH
lookup_orientations = [str(o) for o in urchin.find_stim('MovingBar')[0]['orientations']]
wanted_option, orientation_FR = ['180', 'epoch'], [0 for _ in range(len(lookup_orientations))]

# for ind, orientation in enumerate(lookup_orientations):
#     # Calculate average firing rate for each stimulus presentation of orientation.
#     firing_rate = np.array([[urchin.coarse_FR(*bar_stim_sub[rep][option], s_t)
#                     for option, spike_times in epochs.items() if option == orientation
#                     for s_t in spike_times]
#                     for rep, epochs in bar_stim_sub_spikes.items()])
#     # Find average firing rate across all trial repetitions.
#     avg_firing_rate = np.mean(firing_rate, axis=0)
#     orientation_FR[ind] = avg_firing_rate
#
# flash_FR = np.array([[urchin.generate_PSTH(*flash_stim_sub[rep][option], s_t)
#                 for option, spike_times in epochs.items() if option in wanted_option
#                 for s_t in spike_times]
#                 for rep, epochs in flash_stim_sub_spikes.items()])

full_RF = np.array([[urchin.generate_RF(*check_stim_sub[rep][option], rep_num=rep_num, bounded_spike_times=s_t)
                for option, spike_times in epochs.items() if option in wanted_option
                for s_t in spike_times]
                for rep_num, (rep, epochs) in enumerate(check_stim_sub_spikes.items())])
full_RF = np.mean(full_RF, axis=0)
# squish_RF = np.array([urchin.compress_RF(rf) for rf in full_RF])

# one_clust = check_stim_sub_spikes['rep_0']['epoch'][76]
# sample_RF = urchin.generate_RF(*check_stim_sub['rep_0']['epoch'], rep_num=0, bounded_spike_times=one_clust, to_collapse=True)
np.save('full_RF.npy', squish_RF)
# np.save('clust_22.npy', one_clust)
# Calculate average across trials for each cluster.
# flash_FR = np.mean(flash_FR, axis=0)
# np.save('avg_flash_FR.npy', flash_FR)
# Because orientations is ragged between each presented orientation.
# np.save('avg_orientation_FR.npy', orientation_FR)
# with open('spikes_first_checks.pkl', 'wb') as f:
#     pickle.dump(check_stim_sub_spikes['rep_0']['epoch'], f)
