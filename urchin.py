import os
import json
import math
import csv
from itertools import compress
import numpy as np
import matplotlib.pyplot as plt
import h5py
from phylib.io.model import TemplateModel

class Urchin:

    def __init__(self, path_to_sorted_data, path_to_raw_data, path_to_json):
        self.path_to_sorted_data = path_to_sorted_data
        self.extracted_clusters = self._extract_clusters()
        # Raw data for TTL pulse info.
        self.dataset = h5py.File(path_to_raw_data,'r')
        # For TTL pulse. raw bits and times for extrapolation
        # need to assert that dataset is correct shape.
        self.pulses, self.times = self.load_TTL() # Modified in load_TTL
        self.on_times, self.off_times = None, None # Modified in _prune_TTL
        self._prune_TTL()
        # For information about stimuli presented in experiment.
        with open(path_to_json) as config_file:
            self.config = json.load(config_file)
        self.stim_sum = self._extract_stim_times()
        self.stim_bounds = self.split_into_stimuli() # returned by split_into_stimulist

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
        pulses, times = self.dataset['bits']['bits'], self.dataset['bits']['frameno'] - first_frame_num
        # on_times, off_times = self.times[self.pulses == 128], self.times[self.pulses == 0]
        return pulses, times

    def find_stim(self, stim_name):
        """ Get JSON config for particular stim given name."""
        for stim in self.config['loggedStimuli']:
            if (stim['protocolName'] == stim_name):
                return stim

    def split_into_stimuli(self):
        """ Split TTL pulses to bound each stimuli in experiment.

            Returns
            -------
            list
                2D list with [stimuli_name, index_of_start, index_of_stop, [timing_info]] entries.
                index start and stop is inclusive (since pulses examined are only HIGH (on_times))
        """
        # TODO: bring back _find_pauses and _find_non_pause_bounds and bring out _separate_stims and _find_stim_bounds
        intervals = [(ind, interval) for ind, interval in enumerate(np.diff(self.on_times)) if interval >=25]

        # Several helper functions to help separate:
        #   (_find_pauses, _find_non_pause_bounds, _split_pause_chunks, _separate_stims, _find_stim_bounds)

        remove_nones = lambda a_l: [a for a in a_l if a is not None]
        peel_nesting = lambda a_l: [a_s for a in a_l for a_s in a]
        # Begin separating.
        # slice stim_sum, as timing_info (last item) is not necessary when finding pauses.
        # Find bounds of each Pause in experiment
        pause_bounds = remove_nones([self._find_pauses(ind,interval,self.stim_sum) for ind, interval in intervals])
        # Find bounds between each Pause
        non_pause_bounds = peel_nesting([self._find_non_pause_bounds(ind, bound, pause_bounds) for ind, bound in enumerate(pause_bounds)])
        # Get all stimuli in bounds between Pause (non_pause_bounds), including leftmost Pause.
        chunks = self._split_pause_chunks(self.stim_sum)
        # Use total time of stimulus calculated in _extract_stim_times to find bounds for each stimulus.
        stim_bounds = peel_nesting([self._find_stim_bounds(*bound, stim) for bound, stim in zip(non_pause_bounds, chunks)])

        return stim_bounds

    def split_into_sub_stimuli(self, name, start_bound, stop_bound, timing_info):
        """ For most stimuli, the actual stimulus is repeated in each epoch; sub-stimulus bounds must be extracted.
            To determine how many bounds to make (epoch_num), and the time of each bound (epoch_time).

            Parameters
            ----------
            timing_info: list
                        [epoch_num, epoch_time, inter_rep_time]

            Returns
            -------
            dict
                Nested, with keys for each repetition ('rep_{i}').
                Within each rep_i is a dict with keys ({stim_option in stim_options} or 'epoch').
        """
        epoch_num, epoch_time, inter_stim_time, inter_rep_time = timing_info
        # To get label for each epoch by stimulus options (i.e. orientation, intensity)
        valid_option = lambda opt, st: list(compress(opt, [op in st for op in opt]))
        # chunk()
        def chunk(ar, n):
            for i in range(0, len(ar), n):
                yield ar[i:i+n]

        stim_options = ['stepSizes', 'orientations', 'gratingOrientations']
        stim_log = ['flashLog', 'orientationLog']
        current_stim = self.find_stim(name)
        # Get valid options (i.e. orientation, intensity) for current_stim from stim_options
        stim_options = current_stim[valid_option(stim_options,current_stim)[0]] if valid_option(stim_options,current_stim) else 1
        # Get log of all options displayed during experiment based on valid stim_option for current_stim
        stim_log = current_stim[valid_option(stim_log,current_stim)[0]] if valid_option(stim_log,current_stim) else 1
        # Split up the log of options played by number of repetitions
        # heuristic is that each option plays once per rep.
        stim_log = list(chunk(stim_log, len(stim_options))) if stim_options !=1 else None
        # Total number of epochs (every time option changes) for duration of experiment, int divided by number of option changes.
        rep_num = epoch_num // len(stim_options) if stim_options != 1 else epoch_num

        sub_rep_ends, sub_epoch_ends = [], []
        for _ in range(rep_num):
            sub_epoch_ends = []
            if inter_rep_time != 0:
                # Finds index in on_times matching inter_rep_time (first thing to happen in rep)
                inter_rep_ind = self._separate_stims(start_bound, stop_bound, inter_rep_time)
                start_bound += inter_rep_ind-start_bound if inter_rep_ind is not None else 0

            for option in range( (len(stim_options) if isinstance(stim_options,list) else stim_options) ):
                if inter_stim_time != 0:
                    # Finds index in on_times within epoch_time matching inter_rep_time, (first thing to happen in epoch_time)
                    inter_stim_ind = self._separate_stims(start_bound, stop_bound, inter_stim_time)
                    start_bound += inter_stim_ind-start_bound if inter_stim_ind is not None else 0

                sub_epoch_ends.append([start_bound, self._separate_stims(start_bound, stop_bound, epoch_time-inter_stim_time)])
                start_bound += sub_epoch_ends[option][1]-start_bound if sub_epoch_ends[option][1] is not None else 0

            sub_rep_ends.append(sub_epoch_ends)

        sub_ends = {f'rep_{ind_r}':{f'{opt_label}':e_bounds for opt_label, e_bounds in zip(
                                     ( stim_log[ind_r] if stim_options != 1 else ['epoch']*stim_options ),
                                     epochs_bounds)}
                       for ind_r, epochs_bounds in enumerate(sub_rep_ends)}

        return sub_ends

    def extract_spiketimes_within_bounds(self, cluster_id, bounds, group_idx=None):
        """ Extract spike times that occur between bound in bounds.

            Parameters
            ----------
            cluster_id: int or list
                        The cluster(s) from which spike_times will be extracted. If list,
                        style must match self.extracted_clusters. i.e. length of 'total number of groups'
                        and cluster_ids in list at index corresponding to group_idx.
            bounds: dict or list pair.
                        Provided from split_into_sub_stimuli, can be one repetition, all repetitions, or
                        one bound range (i.e. [start_bound, stop_bound]).
            Returns
            -------
            dict or ndarray
                            The spike times within specified bound(s). Matches order of 'cluster_id' parameter.
        """
        # Mask finds times within bounds.
        mask = lambda a_l, st, stp: a_l[np.logical_and(a_l >= self.on_times[st], a_l <= self.on_times[stp])]
        peel_nesting = lambda a_l: [a_s for a in a_l for a_s in a]
        # Grouper extracts spike_times for clusters in 'cluster_id' (assumes style of self.extracted_clusters)
        def _grouper(c_idxs):
            # g_id + 1 as g_id is zero indexed, but group folders are 1 indexed.
            # The order of cluster_ids in c_idx is maintained below, to allow
            # mapping of unlabeled spike_times to cluster_id.
            return [self._extract_spiketimes(g_id+1, c_ids) for g_id, c_ids in enumerate(c_idxs) if c_ids != 0]

        if isinstance(bounds, dict) and isinstance(cluster_id, int):
            # Multiple sub-stimuli bounds are provided, and spike_times holds one cluster.
            assert group_idx is not None
            spike_times = self._extract_spiketimes(group_idx, [cluster_id])
            return {rep:{option:mask(spike_times, bound[0], bound[1]) for option, bound in epochs.items()} for rep, epochs in bounds.items()}
        elif isinstance(bounds, dict) and isinstance(cluster_id, list):
            # Multiple sub-stimuli bounds are provided, but spike_times holds multiple clusters.
            # peel nesting because of format output of _extract_spiketimes.
            total_spike_times = peel_nesting(_grouper(cluster_id))
            return {rep:
                {option:[mask(spike_times, bound[0], bound[1]) for spike_times in total_spike_times] for option, bound in epochs.items()}
                for rep, epochs in bounds.items()}
        elif isinstance(cluster_id, list):
            # Single bounds is provided, but spike_times holds multiple clusters.
            start_bound, stop_bound = bounds
            total_spike_times = peel_nesting(_grouper(cluster_id))
            return [mask(spike_times, start_bound, stop_bound) for spike_times in total_spike_times]
        else:
            # Single bounds and single cluster.
            assert group_idx is not None
            start_bound, stop_bound = bounds
            spike_times = self._extract_spiketimes(group_idx, [cluster_id])
            return mask(spike_times, start_bound, stop_bound)

    def generate_PSTH(self, start_ind, stop_ind, bounded_spike_times, window_size=50, step=10):
        """ Generate peri-stimulus time histogram for given spike_times bounded by stimulus epoch times.
            Parameters
            ----------
            start_time, stop_time: float
                                The time bound is the start (or stop) time for the beginning (or end) of the stimulus.
                                This is irrespective of the first and last spikes in this duration (see note below).
            bounded_spike_times: numpy.ndarray
                                An array of spike_times for a cluster that falls between sub-stimulus bounds.
                                Each row in value for key 'option' in stim_options returned from extract_spiketimes_within_bounds.
            window_size: int
                        Size (in ms) of window to look through when striding across bounded_spike_times.
            step: int
                Size (in ms) of step that window should move when striding.

            Returns
            -------
            list
                list of counts of spikes occurring within striding window
        """
        # start_time, stop_time = bounded_spike_times[0], bounded_spike_times[-1:]
        # Note: the above should not be how start and stop is found, as this limits search to the first spike in the time duration of
        # the stimulus to the last spike. Instead, the bounds should be the start and stop of the stim, irrespective of any spikes.
        start_time, stop_time = self.on_times[start_ind], self.on_times[stop_ind]

        mask = lambda a_l, st, stp, at_end=False: np.logical_and(a_l >= st, a_l < stp) if not at_end else np.logical_and(a_l >= st, a_l <= stp)
        # chunk to return generator with premade windows to look at bounded_spike_times through.
        def chunk(strt_b, stp_b, win_s, s):
            # Won't work cleanly when (stp_b - win_s) mod(s) != 0
            # Get n number of intervals to divide up time, before edge of window goes out of bounds.
            n = int(((stp_b - win_s) - strt_b) // s)
            cn = strt_b
            for i in range(n+1):
                b_b = cn, cn+win_s
                cn += s
                # Make last bin equal to end of previous bin till stp_b.
                yield b_b if i != n else (cn, stp_b)

        windows = list(chunk(start_time, stop_time, window_size*10**-3, step*10**-3))

        counts = [np.count_nonzero(mask(bounded_spike_times, start_t, stop_t) == True) if ind != len(windows)-1
            else np.count_nonzero(mask(bounded_spike_times, start_t, stop_t, at_end=True) == True)
            for ind, (start_t, stop_t) in enumerate(windows)]
        return np.array(counts) / (window_size*10**-3)


    def generate_RF(self):
        """ Generate receptive field for given spike times of cluster_id."""

    def _extract_spiketimes(self, group_idx, cluster_ids):
        """ Extract spike times from given group_idx (group directory) and cluster_id.
            Parameters
            ---------
            group_idx: int
                        This is the int representation of the group, without leading zeros
                        e.g. group '001' on disk is 1 in parameter.
            cluster_id: int
                        This is a list of cluster_ids (requiring group_idx) to minimize the
                        number of redundant calls to np.load
            Returns
            -------
            numpy.ndarray
                        1D Array with spike_times (in seconds) for a given cluster_id
        """
        meta_spike = lambda g_i: (
            np.load(f'{self.path_to_sorted_data}/{g_i:03d}/traces/traces.GUI/spike_clusters.npy'),
            np.load(f'{self.path_to_sorted_data}/{g_i:03d}/traces/traces.GUI/spike_times.npy'))

        def _extract(s_clust, s_times, c_id):
            # Combine time meta info (cluster_id) with times.
            s_inf = np.vstack((s_clust, s_times)).T
            # Divide by 20000, as this is refresh rate of MEA, and times are in (seconds from start)*refresh rate.
            return s_inf[s_inf[:,0] == int(c_id), 1] / 20000

        # 'grp_spiketimes' is a list of every spike time in the group
        # 'grp_spikeclusters' is a list mapping cluster_id to the times in 'grp_spiketimes'
        grp_spikeclusters, grp_spiketimes = meta_spike(group_idx)
        return [_extract(grp_spikeclusters, grp_spiketimes, cluster_id) for cluster_id in cluster_ids]

    def _extract_clusters(self):
        """ Go through given data directory and assemble list of cluster_id's
            as a 2x2 matrix with row index as group_idx-1, and each row element as
            'good' labelled cluster_id's for group_idx.

            Note that cluster_id's are specific to its group, not across groups.
        """

        # Step one: get list of groups from path
        sub_directories = os.listdir(self.path_to_sorted_data)
        groups = [sub_directory for sub_directory in sub_directories if sub_directory.isnumeric()]

        extracted_clusters = [0 for x in range(len(groups))]
        # Step two: access 'cluster_info.tsv' from inside path/{group}/traces/traces.GUI/
        for group_idx in groups:
            try:
                with open(f'{self.path_to_sorted_data}/{group_idx}/traces/traces.GUI/cluster_info.tsv') as tab_info:
                    info = csv.reader(tab_info, delimiter='\t', quotechar='"')

                    #Step three: extract good cluster_id's and append to mapped extracted_clusters list
                    group_ls = [int(row[0]) for row in info if 'good' in row and row[0].isnumeric()]

                    if group_ls:
                        extracted_clusters[int(group_idx.lstrip('0'))-1] = group_ls
            except Exception as e:
                continue
        return extracted_clusters

    def _find_cluster_info(self):
        """ Used to extract information for electrical image (ei), such as
            best channels and associated electrode positions for cluster_id.
        """

    def _prune_TTL(self):
        """ Used by self.generate_RF to fill in missing and remove extra TTL pulses."""
        # TTL pulse frequency limit.
        max_pulse_freq = 70
        strange_on_times = self.on_times = self.times[self.pulses > 0] / 20000
        strange_off_times = self.off_times = self.times[self.pulses == 0] / 20000
        # Eliminate pulses within maximum frequency limit. Include the first time with 'to_begin'.
        self.on_times = self.on_times[np.diff(strange_on_times, prepend = 1/max_pulse_freq) >= 1/max_pulse_freq]
        self.off_times = self.off_times[np.diff(strange_off_times, prepend = 1/max_pulse_freq) >= 1/max_pulse_freq]

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
        pause_in_stim_sum = [ind for ind, stim in enumerate(stim_sum) if 'Pause' in stim]
        # Use indices and loop from 0 to start of pause_in_stim_sum, then [0]:[1] till [len(pause_in_stim_sum)-1]:len(stim_sum)
        pause_chunks = []
        if not pause_in_stim_sum[0]:
            # Case for when Pause is at beginning of experiment.
            for ind, cut in enumerate(pause_in_stim_sum):
                # Cuts out all stimuli between Pauses, or from last Pause till the end of the experiment.
                if (ind != len(pause_in_stim_sum)-1):
                    pause_chunks.append(stim_sum[cut:pause_in_stim_sum[ind+1]])
                else:
                    pause_chunks.append(stim_sum[cut:len(stim_sum)])

            return pause_chunks
        else:
            # Case for when Pause is not at beginning of experiment.
            for ind, cut in enumerate(pause_in_stim_sum):
                # Cuts out all stimuli between Pauses, from beginning till first Pause, or from last Pause
                # till the end of the experiment.
                if ind == 0:
                    pause_chunks.append(stim_sum[0:cut])
                elif (ind != len(pause_in_stim_sum)-1):
                    pause_chunks.append(stim_sum[cut:pause_in_stim_sum[ind+1]])
                else:
                    pause_chunks.append(stim_sum[cut:len(stim_sum)])

            return pause_chunks

    def _separate_stims(self, start_bound, stop_bound, _duration, re_depth=0):
        """ Separate combination of stims into individuals.
            stop_bound will be upcoming pause, which will stay the same, but
            start_bound MUST be shifted to end index of previous stimulus.
        """
        # For interStimulusInterval, need to separate interval from actual stimulus.
        time_to_find = self.on_times[start_bound]+ _duration
        times_to_search = self.on_times[start_bound:stop_bound+1]
        # The space between TTL pulses (~half the frame rate) is 0.0834.
        # rtol is 0.003, or 3.58% of the pulse spacing.
        found_time = np.isclose(times_to_search, time_to_find, rtol=0.003, atol=0.01)
        ind_found_time = np.where(found_time == True)[0]
        # Add some resiliency in separating. Recursion allows shift of bounds by one to skip
        # a bad start_bound and hopefully find the _duration.
        if ind_found_time.size != 0:
            return start_bound + ind_found_time[0]
        elif re_depth != 1:
            return self._separate_stims(start_bound+1, stop_bound, _duration, re_depth=1)
        else:
            return None

    def _find_stim_bounds(self, start_bound, stop_bound, bound_stims):
        """ Find bounds of each stimulus, Pause or non-pause.
            Use non_pause_bounds to search for complete or additive combination of stims.
            Use _separate_stims to find stim start and stop, regardless
            of individual or combination of stim(s) in non-pause bound.

            Parameters
            ----------
            bound_stims: list
                        This is a list of all stimuli found between two Pauses (bounded
                        by start_bound and stop_bound), which includes the leftmost Pause.
        """
        stim_ends = []
        for ind, (name, sus_stim, *timing_info) in enumerate(bound_stims):
            if name == 'Pause':
                # Since non_pause_bounds does not include pause indices (beginning), adjust so pause can be found too.
                # This is for the sake of accurate start_bound shifting.
                stim_ends.append([name, start_bound-2, self._separate_stims(start_bound-2, stop_bound, sus_stim), timing_info])
            else:
                stim_ends.append([name, start_bound, self._separate_stims(start_bound, stop_bound, sus_stim), timing_info])

            start_bound += stim_ends[ind][2]-start_bound if stim_ends[ind][2] is not None else 0

        return stim_ends

    def _extract_stim_times(self):
        """ Used to separate stimulus timing lengths from JSON config."""
        # Stimuli list and options that lengthens epoch_num.
        # Here, epoch is each variation in stimuli, by intensity or orientation
        stim_list = self.config['loggedStimuli']
        stim_options = ['stepSizes', 'orientations', 'gratingOrientations']
        stim_sum = []
        # Iterate through stimuli and extract actual times to stim_sum
        # ('stimulus name', total time, num of epochs, time per epoch, sum of times between epoch)
        # total time = number of epochs * time per epoch + sum of times between each epoch
        for stim in stim_list:
            epoch_time, epoch_num, inter_rep_time = 0, 1, 0
            stim_option = next((option for option in stim_options if option in stim), None)
            # stim_option finds stimulus specific key in stim dict.
            epoch_time += stim['_actualPreTime']+stim['_actualStimTime']+stim['_actualTailTime']
            # This is time within epoch between stimulus presentation.
            inter_stim_time = stim['_actualInterStimulusInterval'] if '_actualInterStimulusInterval' in stim else 0
            epoch_time += inter_stim_time
            if 'stimulusReps' in stim:
                epoch_num = stim['stimulusReps'] * len(stim[stim_option]) if stim_option is not None else stim['stimulusReps']
                inter_rep_time += stim['stimulusReps'] * stim['_actualInterFamilyInterval'] if '_actualInterFamilyInterval' in stim else 0
                # Last three elements are timing info: epoch_num, epoch_time (including interStimulusInterval) and interFamilyInterval (if relevent).
            stim_sum.append((stim['protocolName'], epoch_num * epoch_time + inter_rep_time, epoch_num, epoch_time, inter_stim_time, (inter_rep_time/stim['stimulusReps'] if 'stimulusReps' in stim else inter_rep_time) ))
        return stim_sum
