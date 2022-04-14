import os
import json
import numpy as np
import matplotlib.pyplot as plt
import h5py
from phylib.io.model import TemplateModel

class Urchin:

    def __init__(self, path_to_sorted_data, path_to_json):
        self.path_to_sorted_data = path_to_sorted_data
        self.extracted_clusters = None
        self.dataset = h5py.File(path_to_sorted_data,'r')
        with open(path_to_json) as config_file:
            self.config = json.load(config_file)
        # For TTL pulse. raw bits and times for extrapolation
        self.pulses, self.times = None, None # Modified in load_TTL
        self.on_times, self.off_times = None, None # Modified in _prune_TTL

    def _extract_clusters(self):
        """ Go through given data directory and assemble list of cluster_id's
            as a 2x2 matrix with row index as group_idx-1 and elements in row as
            'good' labelled cluster_id's for group_idx.
        """

        # Step one: get list of groups from path
        sub_directories = os.listdir(self.path_to_sorted_data)
        groups = [sub_directory for sub_directory in sub_directories if sub_directory.isnumeric()]

        self.extracted_clusters = [0 for x in range(len(groups))]
        # Step two: access 'cluster_info.tsv' from inside path/{group}/traces/traces.GUI/
        for group_idx in groups:
            with open(f'{self.path_to_sorted_data}/{group_idx}/traces/traces.GUI/cluster_info.tsv') as tab_info:
                group_ls = []
                info = csv.reader(tab_info,delimiter='\t',quotechar='"')

                #Step three: extract good cluster_id's and append to mapped extracted_clusters list
                for row in info:
                    group_ls.append(row[0]) if ('good' in row) and row[0].isnumeric() else None

                self.extracted_clusters[int(group_idx.lstrip('0'))-1] = group_ls

    def extract_spiketimes(self,group_idx,cluster_id):
        """ Extract spike times from given group_idx (group directory) and cluster_id."""

        # 'grp_spiketimes' is a list of every spike time in the group
        # 'grp_spikeclusters' is a list mapping cluster_id to the times in 'grp_spiketimes'
        grp_spiketimes = np.load(f'{self.path_to_sorted_data}/{group_idx}/traces/traces.GUI/spike_times.npy') # I think this and clusters is all thats needed to extract spike times
        grp_spikeclusters = np.load(f'{self.path_to_sorted_data}/{group_idx}/traces/traces.GUI/spike_clusters.npy')

        grp_spikeinfo = np.vstack((grp_spiketimes,grp_spikeclusters)).T
        # To extract spike_times of a given cluster_id, the above lists need to be sorted by 'grp_spikeclusters'
        # and then extracted for element matching cluster_id.
        # Sorting is not neccessary, but will it boost speed?
        grp_spikeinfo = grp_spikeinfo[grp_spikeinfo[:,1].argsort()]

        return grp_spikeinfo[grp_spikeinfo[:,1]==int(cluster_id),0]

    def _find_cluster_info(self):
        """ Used to extract information for electrical image (ei), such as
            best channels and associated electrode positions for cluster_id.
        """

    def _prune_TTL(self,ttl):
        """ Used by self.generate_RF to fill in missing and remove extra TTL pulses."""
        # TTL pulse frequency limit.
        max_pulse_freq = 70
        strange_on_times = self.on_times = self.times[self.pulses > 0] / 20000
        # Eliminate pulses within maximum frequency limit. Include the first time with 'to_begin'.
        self.on_times = self.on_times[np.diff(strange_on_times) >= 1/max_pulse_freq]

    def _extract_stim_times(self):
        """ Used to separate stimulus timing lengths from JSON config."""
        # Stimuli list and options that lengthens epoch_num.
        # Here, epoch is each variation in stimuli, by intensity or orientation
        stim_list = config['loggedStimuli']
        stim_options = ['stepSizes', 'orientations', 'gratingOrientations']
        stim_sum = []
        # Iterate through stimuli and extract actual times to stim_sum
        # ('stimulus name', total time)
        for stim in stim_list:
            epoch_time, epoch_num, interval_time = 0, 1, 0
            stim_option = next((option for option in stim_options if option in stim), None)
            # stim_option finds stimulus specific key in stim dict.
            epoch_time += stim['_actualPreTime']+stim['_actualStimTime']+stim['_actualTailTime']
            epoch_time += stim['_actualInterStimulusInterval'] if '_actualInterStimulusInterval' in stim else 0
            if 'stimulusReps' in stim:
                epoch_num = stim['stimulusReps'] * len(stim[stim_option]) if stim_option is not None else stim['stimulusReps']
                interval_time += stim['stimulusReps'] * stim['_actualInterFamilyInterval'] if '_actualInterFamilyInterval' in stim else 0
            stim_sum.append((stim['protocolName'], epoch_num * epoch_time + interval_time))
        return stim_sum

    def generate_RF(self):
        """ Generate receptive field for given spike times of cluster_id."""

    def load_TTL(self):
        """ Load TTL bits to hold times of HIGH / LOW pulse changes."""
        # From maxwell package: '/sig' subdataset is uint16. Left bit shift and bitwise OR.
        # [1027,0] is first item in last row of dataset. left bit shift multip. by 2**16.
        # Bitwise OR is just addition of [1027,0]*2**16 and [1026,0].
        first_frame_num = (self.dataset['sig'][1027,0] << 16) | self.dataset['sig'][1026,0]
        self.pulses, self.times = self.dataset['bits']['bits'], self.dataset['bits']['frameno'] - first_frame_num
        on_times, off_times = self.times[self.pulses == 128], self.times[self.pulses == 0]
        return on_times, off_times
