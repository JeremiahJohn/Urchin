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
        # For information about stimuli presented in experiment.
        with open(path_to_json) as config_file:
            self.config = json.load(config_file)
        self.stim_sum, self.stim_bounds = self._extract_stim_times(), None # returned by split_into_stimuli
        # For TTL pulse. raw bits and times for extrapolation
        # need to assert that dataset is correct shape.
        self.pulses, self.times = self.load_TTL()
        self.on_times, self.off_times = None, None # Modified in _prune_TTL
        self._prune_TTL()

    def load_TTL(self):
        """ Load TTL bits to hold times of HIGH / LOW pulse changes.

            Returns
            -------
            tuple: two lists
                times for which TTL pulse reads HIGH (on) or LOW (off)
        """
        # From maxwell package: '/sig' subdataset is uint16. Left bit shift and bitwise OR.
        # [1027,0] is first item in last row of dataset. left bit shift multip. by 2**16.
        # Bitwise OR is just addition of [1027,0]*2**16 and [1026,0].
        first_frame_num = (self.dataset['sig'][1027,0] << 16) | self.dataset['sig'][1026,0]
        self.pulses, self.times = self.dataset['bits']['bits'], self.dataset['bits']['frameno'] - first_frame_num
        on_times, off_times = self.times[self.pulses == 128], self.times[self.pulses == 0]
        return on_times, off_times

    def split_into_stimuli(self):
        """ Split TTL pulses to bound each stimuli in experiment.

            Returns
            -------
            list
                2D list with [stimuli_name, index_of_start, index_of_stop] entries.
                index start and stop is inclusive (since pulses examined are only HIGH (on_times))
        """
        intervals = [(ind, interval) for ind, interval in enumerate(np.diff(self.on_times)) if interval >=25]

        # Several inner functions to help separate:
        #   (_find_pauses,_find_non_pause_bounds,_split_pause_chunks)
        def _separate_stims(start_bound, stop_bound, _duration):
            # Separate combination of stims into individuals.
            # stop_bound will be upcoming pause, which will stay the same, but
            # start_bound MUST be shifted to end index of previous stimulus.
            time_to_find = self.on_times[start_bound]+ _duration
            times_to_search = self.on_times[start_bound:stop_bound+1]
            # The space between TTL pulses (~half the frame rate) is 0.0834.
            # rtol is 0.003, or 3.58% of the pulse spacing.
            found_time = np.isclose(times_to_search, time_to_find, rtol=0.003, atol=0.01)
            ind_found_time = np.where(found_time == True)[0]
            return start_bound + ind_found_time[0] if ind_found_time.size != 0 else None

        def _find_stim_bounds(start_bound, stop_bound, bound_stims):
            # Find bounds of each stimulus, Pause or non-pause.
            # Use non_pause_bounds to search for complete or additive combination of stims.
            # Use _separate_stims to find stim start and stop, regardless
            # of individual or combination of stim(s) in non-pause bound.
            stim_ends = []
            for ind, (name, sus_stim) in enumerate(bound_stims):
                if name == 'Pause':
                    # Since non_pause_bounds does not include pause indices (beginning), adjust so pause can be found too.
                    # This is for the sake of accurate start_bound shifting.
                    stim_ends.append([name, start_bound-2, _separate_stims(start_bound-2, stop_bound, sus_stim)])
                else:
                    stim_ends.append([name, start_bound, _separate_stims(start_bound, stop_bound, sus_stim)])

                start_bound += stim_ends[ind][2]-start_bound if stim_ends[ind][2] is not None else 0

            return stim_ends

        remove_nones = lambda a_l: [a for a in a_l if a is not None]
        peel_nesting = lambda a_l: [a_s for a in a_l for a_s in a]
        # Begin separating.
        # slice stim_sum as last three items are not necessary when finding pauses.
        pause_bounds = remove_nones([self._find_pauses(ind,interval,self.stim_sum[:2]) for ind, interval in intervals])
        non_pause_bounds = peel_nesting([self._find_non_pause_bounds(ind, bound, pause_bounds) for ind, bound in enumerate(pause_bounds)])
        chunks = self._split_pause_chunks(self.stim_sum)
        stim_test = peel_nesting([_find_stim_bounds(*bound, stim) for bound, stim in zip(non_pause_bounds, chunks)])

        return stim_test


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

    def generate_RF(self):
        """ Generate receptive field for given spike times of cluster_id."""

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
                info = csv.reader(tab_info,delimiter='\t',quotechar='"')

                #Step three: extract good cluster_id's and append to mapped extracted_clusters list
                group_ls = [row[0] for row in info if 'good' in row and row[0].isnumeric()]

                self.extracted_clusters[int(group_idx.lstrip('0'))-1] = group_ls

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

    def _find_pauses(self, ind, interval, stim_sum):
        """ Find pauses given stim_sum (_extract_stim_times), self.on_times intervals >= 25s,
            and current index in intervals.
        """
        found_pause = [math.isclose(sus_pause,interval, rel_tol=0.1) for _,sus_pause, *_ in stim_sum]
        return [ind, ind+1] if any(found_pause) else None

    def _find_non_pause_bounds(self, ind, bound, pause_bounds):
        """ Find bounds outside of known pauses and append as non_pause to search for stim.
            Add to start, as non-pause bound starts at end of pause, but stim starts
            one TTL pulse after.
        """
        if ind < (len(pause_bounds)-1):
            if (bound[1] == 0 and ind==0):
                return [[0, bound[0]-1], [bound[1]+1, pause_bounds[ind+1][0]-1]]
            else:
                return [[bound[1]+1, pause_bounds[ind+1][0]-1]]
        else:
            return [[bound[1]+1, len(self.on_times)-1]] if bound[1] != len(self.on_times)-1 else None

    def _split_pause_chunks(self, stim_sum):
        """ Reorganize stim_sum (_extract_stim_times) to separate stimuli between each pause.
            Each chunk should start with a new Pause.
        """
        pause_in_stim_sum = [ind for ind, stim in enumerate(stim_sum) for sub in stim if sub == 'Pause']
        # Use indices and loop from 0 to start of pause_in_stim_sum, then [0]:[1] till [len(pause_in_stim_sum)-1]:len(stim_sum)
        pause_chunks = []
        if not pause_in_stim_sum[0]:
            for ind, cut in enumerate(pause_in_stim_sum):
                if (ind != len(pause_in_stim_sum)-1):
                    pause_chunks.append(stim_sum[cut:pause_in_stim_sum[ind+1]])
                else:
                    pause_chunks.append(stim_sum[cut:len(stim_sum)])

            return pause_chunks
        else:
            for ind, cut in enumerate(pause_in_stim_sum):
                if ind == 0:
                    pause_chunks.append(stim_sum[0:cut])
                elif (ind != len(pause_in_stim_sum)-1):
                    pause_chunks.append(stim_sum[cut:pause_in_stim_sum[ind+1]])
                else:
                    pause_chunks.append(stim_sum[cut:len(stim_sum)])

            return pause_chunks

    def _extract_stim_times(self):
        """ Used to separate stimulus timing lengths from JSON config."""
        # Stimuli list and options that lengthens epoch_num.
        # Here, epoch is each variation in stimuli, by intensity or orientation
        stim_list = config['loggedStimuli']
        stim_options = ['stepSizes', 'orientations', 'gratingOrientations']
        stim_sum = []
        # Iterate through stimuli and extract actual times to stim_sum
        # ('stimulus name', total time, num of epochs, time per epoch, sum of times between epoch)
        # total time = number of epochs * time per epoch + sum of times between each epoch
        for stim in stim_list:
            epoch_time, epoch_num, interval_time = 0, 1, 0
            stim_option = next((option for option in stim_options if option in stim), None)
            # stim_option finds stimulus specific key in stim dict.
            epoch_time += stim['_actualPreTime']+stim['_actualStimTime']+stim['_actualTailTime']
            epoch_time += stim['_actualInterStimulusInterval'] if '_actualInterStimulusInterval' in stim else 0
            if 'stimulusReps' in stim:
                epoch_num = stim['stimulusReps'] * len(stim[stim_option]) if stim_option is not None else stim['stimulusReps']
                interval_time += stim['stimulusReps'] * stim['_actualInterFamilyInterval'] if '_actualInterFamilyInterval' in stim else 0
            stim_sum.append((stim['protocolName'], epoch_num * epoch_time + interval_time, epoch_num, epoch_time, (interval_time/stim['stimulusReps'] if 'stimulusReps' in stim else interval_time) ))
        return stim_sum
