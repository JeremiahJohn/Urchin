import os
import json
import math
import random
import csv
from itertools import compress
import numpy as np
from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
# from numba import jit
# import matplotlib.pyplot as plt
import h5py
# from phylib.io.model import TemplateModel

class Urchin:

    def __init__(self, path_to_sorted_data, path_to_raw_data, path_to_json):
        self.path_to_sorted_data = path_to_sorted_data
        self.extracted_clusters = self._extract_clusters()
        ## Raw data for TTL pulse info.
        self.dataset = h5py.File(path_to_raw_data,'r')
        ## For TTL pulse. raw bits and times for extrapolation
        # need to assert that dataset is correct shape.
        self.pulses, self.times = self.load_TTL() # Modified in load_TTL
        self.on_times, self.off_times = None, None # In seconds, modified in _prune_TTL
        self._prune_TTL()
        # ****
        self.on_times = np.insert(self.on_times, 1474, self.on_times[1473]+5) # NOTE: current TTL pulses are missing critical pulse at index 1474 at end of tail_time for rep_1 of checkerboard.
        # ****
        self._working_stim = () # Used to store modified stim_times in generate_RF
        ## For information about stimuli presented in experiment.
        with open(path_to_json) as config_file:
            self.config = json.load(config_file)
        self.checks = { f'c_{c_n}':None for c_n in range(len(self.find_stim('CheckerboardReceptiveField'))) }
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
        # for stim in self.config['loggedStimuli']:
        #     if (stim['protocolName'] == stim_name):
        #         return stim
        stim_list = [stim for stim in self.config['loggedStimuli'] if stim['protocolName'] == stim_name]
        return stim_list

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
        current_stim = self.find_stim(name)[0]
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
                # _separate_stims returns start and stop bounds, in case start_bound had to be modified to capture duration.
                inter_rep_ind = self._separate_stims(start_bound, stop_bound, inter_rep_time)[1]
                start_bound += inter_rep_ind-start_bound if inter_rep_ind is not None else 0

            for option in range( (len(stim_options) if isinstance(stim_options,list) else stim_options) ):
                if inter_stim_time != 0:
                    # Finds index in on_times within epoch_time matching inter_rep_time, (first thing to happen in epoch_time)
                    # _separate_stims returns start and stop bounds, in case start_bound had to be modified to capture duration.
                    inter_stim_ind = self._separate_stims(start_bound, stop_bound, inter_stim_time)[1]
                    start_bound += inter_stim_ind-start_bound if inter_stim_ind is not None else 0
                # _separate_stims returns start and stop bounds, in case start_bound had to be modified to capture duration.
                modified_start_bound, duration_stop_bound = self._separate_stims(start_bound, stop_bound, epoch_time-inter_stim_time)
                sub_epoch_ends.append([modified_start_bound, duration_stop_bound])
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
        # Mask finds times within bounds. bounds[1] is the actual stop. No need for +1, as we are not using slices.
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
            start_ind, stop_ind: int
                                The indices of the time bound is the start (or stop) time for the beginning (or end) of the stimulus.
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

    def sort_PSTH(self, multi_PSTH):
        """ Group PSTHs together to make analysis (hopefully) easier by using a
            distance matrix with the cosine distance method followed by ward clustering.
            Parameters
            ----------
            multi_PSTH: ndarray
                        This is an array of shape (num of clusters, num of windows), holding the PSTH for
                        multiple clusters in each row of array.

            Returns
            -------
            list
                List of the grouped PSTH responses of len (total_num_groups), with each index being a set of grouped PSTHs.
        """
        # A cosine distance method will return NaN due to divide by zero for null vectors in PSTH. Get around this by changing null to ones vectors.
        multi_PSTH[np.all(multi_PSTH == 0, axis=1)] = [1 for _ in range(multi_PSTH.shape[1])]
        # Cosine distance is used as this highlights closeness of PSTH distributions in vector space, while not focusing on magnitude.
        distance_matrix = pdist(multi_PSTH, 'cosine')
        # Hierarchical clustering is used because we would have to use elbow method to figure out optimal number of groups for k-means, and also
        # because of the curse of dimensionality for k-means clustering (I think).
        # ward was chosen mainly because it seemed to find the most clusters.
        linkage = ward(distance_matrix)
        # From the linkage matrix, flat clustering can be preformed with a threshold of 0.8.
        # The smaller the threshold, the less accepting groups will be. Perhaps this threshold might need to be modified for other datasets.
        groups_ref = fcluster(linkage, 0.8, criterion='distance')
        # Sort multi_PSTH based on group references in groups_ref.
        grouped_responses = [multi_PSTH[groups_ref == group_num] for group_num in range(1, groups_ref.max()+1)]
        return grouped_responses

    def coarse_FR(self, start_ind, stop_ind, bounded_spike_times):
        """ Calculate average firing rate for given spike_times bounded by stimulus epoch times.
            Parameters
            ----------
            start_time, stop_time: int
                                The indices of the time bound is the start (or stop) time for the beginning (or end) of the stimulus.
                                This is irrespective of the first and last spikes in this duration (see note below).
            bounded_spike_times: numpy.ndarray
                                An array of spike_times for a cluster that falls between sub-stimulus bounds.
                                This can be each row in value for key 'option' in stim_options returned from extract_spiketimes_within_bounds.

            Returns
            -------
            float
                Average firing rate in terms of spikes per second over the sub-stim duration.
        """
        # start_time, stop_time = bounded_spike_times[0], bounded_spike_times[-1:]
        # Note: the above should not be how start and stop is found, as this limits search to the first spike in the time duration of
        # the stimulus to the last spike. Instead, the bounds should be the start and stop of the stim, irrespective of any spikes.
        start_time, stop_time = self.on_times[start_ind], self.on_times[stop_ind]
        # The number of spikes in the broad sub-stimulus time period, to estimate spikes/sec at worse resolution than PSTH.
        return len(bounded_spike_times)/(stop_time - start_time)

    def vector_sum_DSI(self, cluster_rates, thetas):
        """ Compute vector sum DSI as fraction of resultant magnitude in FR magnitudes for each direction.
            Parameters
            ----------
            cluster_rates: ndarray
                        This is a 1D array of coarse firing rates for one cluster's response to each moving bar
                        orientation in experiment with rows of array mapped to 'thetas'.
            thetas: list
                        A list of orientation angles in radians that maps to 'cluster_rates'.

            Returns
            -------
            tuple
                (DSI, preferred_orientation) where DSI is norm(vector_sum_resultant) / total_magnitude
                and preferred_orientation is arctan(y/x) for x,y in vector_sum_resultant.
        """
        # Sum of FR in each orientation, then the vector_sum coordinates.
        total_magnitude, resultant = cluster_rates.sum(), self._compute_vector_sum(cluster_rates, thetas)
        # L2 norm of resultant
        return np.linalg.norm(resultant)/total_magnitude, np.arctan(resultant[1]/resultant[0])

    def generate_RF(self, start_ind, stop_ind, rep_num, bounded_spike_times, order=0, to_collapse=False, axis=1):
        """ Generate receptive field for given spike times of cluster_id.
            Parameters
            ----------
            start_ind, stop_ind: int
                                The indices of the time bound is the start (or stop) time for the beginning (or end) of
                                the epoch (one epoch per rep) within the stimulus. This is the same as generate_PSTH or coarse_FR.
            rep_num: int
                    The current rep number (zero-indexed) for corresponding start_ind and stop_ind. This is used to extract the correct
                    page within the checkerboard array.
            bounded_spike_times: numpy.ndarray
                                An array of spike_times for a cluster that falls between sub-stimulus bounds.
                                This can be each row in value for key 'option' in stim_options returned from
                                extract_spiketimes_within_bounds, to enable batch processing.
            order: int
                    This is the index of the stim in question, as per order of appearance in JSON config.
                    Default is 0 which isn't used when only one CheckerboardReceptiveField was displayed during experiment.
            to_collapse: bool
                    Options to choose between temporally collapsed, 2D spike-triggered average (True), or 3D STA (False).
            axis: int
                    Frame axis to collapse when returning 2D output. Must be 1 or 2, with default as 1.

            Returns
            -------
            numpy.ndarray
                        Either: 3D array (n ms, frame height, frame width) containing the 2D, spike-triggered average of
                        the checkerboard and spike_times, where each page is n ms before the spike.
                        OR: 2D array (frame_height, frame_width) containing the temporally collapsed spike-triggered average,
                        where each column corresponds the average across axis=2 of one frame.
        """
        # First need to construct the huge 3D array that is the movie of the checkerboard stimulus.
        # Use numba maybe as separate, optimized function to create array. This is because the rng used by
        # python's random is a mersenne twister, but numpy is PCG i.e. different paths from the same seed.
        p_floor = lambda a, p=0: np.floor(a * 10**p) / 10**p
        ## Extract correct stim based on 'order' parameter.
        stim_info = self.find_stim('CheckerboardReceptiveField')[order]
        # Find dimensions of the 2D frame
        check_coords = np.array(stim_info['checkCoordinates'])
        frame_height, frame_width = check_coords[check_coords[:,0] == check_coords[0,0]].shape[0], check_coords[check_coords[:,1] == check_coords[0,1]].shape[0]
        # The number of frames before and after the spike to capture.
        frames_before = 30
        frames_after = 10
        # Quick check if there are no spikes.
        if len(bounded_spike_times) == 0:
            sta = np.zeros((frames_before+frames_after, frame_height, frame_width))
            sta_temporal = np.mean(sta, axis=axis).T
            return sta_temporal if to_collapse else sta

        ## Create checkerboard for this stim if it hasn't been made yet.
        if self.checks[f'c_{order}'] is None:
            # Create 3D checkerboard with known (pages, rows, and column), and the random seed.
            # This contains checks for epochs outside of stim_time. To boost efficiency, it might be best to separate this step from the others.
            checkerboard = Urchin._checkerboard(
                                        stim_info['stimulusReps'],
                                        np.ceil(stim_info['_stimTimeNumFrames'] / stim_info['frameDwell']).astype(int),
                                        len(check_coords),
                                        stim_info['randomSeed']).astype(int)

            self.checks[f'c_{order}'] = checkerboard
        else:
            checkerboard = self.checks[f'c_{order}']

        ## Fix dropped TTL pulses in on_times during this epoch production.
        stim_times = self.on_times[start_ind+1:stop_ind] # start_ind + 1 as bounds should include pre and tail_time.
        # An attempt to improve efficiency in batch calls.
        if (len(self._working_stim) == 0 or self._working_stim[0] != rep_num):
            expected_frame_interval = stim_info["frameDwell"] / stim_info["_FR"] # frameDwell is in frames, not seconds.
            frame_intervals = np.diff(stim_times)
            poor_samples_mask = ~np.isclose(frame_intervals, expected_frame_interval, rtol=0.001, atol=0.01) # times when interval is not close to expected.
            poor_samples_inds = np.where(poor_samples_mask)[0]
            # int division to find how many frames were dropped, based on expected_frame_interval.
            missing_frames = ((p_floor(frame_intervals[poor_samples_mask], p=3)*100) // (expected_frame_interval*100))
            # Fill in the times of the missing frames based on the values in missing_frames.
            if len(missing_frames) != 0:
                # Indices are all referenced to stim_times, a subset of self.on_times. Slice gets the middle and omits end bounds of linspace.
                missing_times = [np.linspace(stim_times[ind], stim_times[ind+1], missed+2)[1:-1] for ind, missed in zip(poor_samples_inds, missing_frames.astype(int))]
                # To take advantage of numpy insert, we need to flatten missing_times, then map the index to the corresponding time.
                size_missing_times = [len(time) for time in missing_times]
                poor_samples_inds += 1 # Add one as np.insert will place value to replace specificed index, and current inds are left hand bound of time interval.
                # Use mapping to insert flattened missing_times into stim_times. Note that np.ravel is faster than hstack. But this is fine for one execution.
                stim_times = np.insert(stim_times, np.repeat(poor_samples_inds, size_missing_times), np.hstack(missing_times))
            self._working_stim = (rep_num, stim_times)
        else:
            stim_times = self._working_stim[1]

        ## Spike-triggered average of pixel values on each frame based on spike_times in bounded_spike_times.
        # Because stim_times is being used to map to win flips in checkerboard axis=1, they have to be
        # of the same size. However, they are often different sizes, with the trend of more TTL pulses than
        # win flips. So kludge is to resize stim_times by cutting out extra pulses in the beginning of the stim.
        extra = len(stim_times) - (np.ceil(stim_info['_stimTimeNumFrames'] / stim_info['frameDwell']).astype(int))
        stim_times_pruned = stim_times[:-extra] if extra != 0 else stim_times
        assert len(stim_times) != 0
        assert len(stim_times_pruned) != 0
        # Get frames_before + frames_after number of frames before each spike in total spikes within epoch (rep) bounds.
        # Skip spikes close to the start of the epoch (rep).
        sta = np.array([ ( np.vstack([checkerboard[rep_num][stim_times_pruned < s_t][-frames_before:], checkerboard[rep_num][stim_times_pruned >= s_t][:frames_after]])
                            if (len(checkerboard[rep_num][stim_times_pruned < s_t][-frames_before:]) == frames_before) and (len(checkerboard[rep_num][stim_times_pruned >= s_t][:frames_after]) == frames_after)
                            else np.vstack([ np.zeros((frames_before, len(check_coords))), np.zeros((frames_after, len(check_coords))) ]) )
                        for s_t in bounded_spike_times])

        # Calculate mean of pixel values -30: frames behind spike, and reshape to look like frame.
        sta = np.transpose(np.mean(sta, axis=0).reshape(frames_before+frames_after, frame_width, frame_height), axes=(0,2,1))
        # As 'sta' is a 3D array across time, we can compress to 2D to examine temporal changes in pixel intensity
        # of receptive field. We compress by mean across columns.
        sta_temporal = np.mean(sta, axis=axis).T
        return sta_temporal if to_collapse else sta

    def compress_RF(self, full_sta, axis=1):
        """ Compress spike-triggered average computed in generate_RF when performing batch operations.
            Parameters
            ----------
            full_sta: ndarray
                    A 3D array representing the full spike-triggered average, where each page is a frame
                    closer to the frame immediately before a spike, and value at each page is the row x column
                    average frame pixel intensities.
            axis: int
                    The axis (row or column) to compress the full_sta along.

            Returns
            -------
            numpy.ndarray
                    A 2D, compressed represntation of the spike-triggered average, with axis=2 representing temporal
                    changes (closer to the spike) and axis=1 representing spatial pixel changes on one frame (position).
        """
        # Transpose as the result of np.mean is a vertically stacked array, with each layer being a frame.
        return np.mean(full_sta, axis=axis).T


    def _extract_spiketimes(self, group_idx, cluster_ids):
        """ Extract spike times from given group_idx (group directory) and cluster_id.
            Parameters
            ---------
            group_idx: int
                        This is the int representation of the group, without leading zeros
                        e.g. group '001' on disk is passed as 1 in parameter.
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

    def _compare_PSTH(self, x, y, compare_with='cos'):
        """ Use cosine similarity to compare two PSTHs from two clusters. This is a personal
            implementation of cosine distance, but actual clustering should use cluster_PSTH.
        """
        # Euclidean distance falls with intensity of firing.
        euclidean_dist = lambda x,y: np.sqrt(np.sum((x-y)**2))
        # Cosine similarity is fine if firing rate is different between x and y, but
        # compares the spacing of those times more. (0 is dissimilar and 1 is the same)
        cos_sim = lambda x,y: np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
        return cos_sim(x,y) if compare_with == 'cos' else euclidean_dist(x,y)

    def _compute_vector_sum(self, cluster_rates, thetas):
        """ Calculate the vector sum of polar plot FR of one cluster with the inclusion of non-preferred directions.
            Finds the cartesian coordinates for this resultant, and also the total magnitude of each orientation-FR
            vector, so as to aid computation of vector_DSI.
        """
        assert cluster_rates.shape[0] == len(thetas)
        # Transform polar coordinates to cartesian, so as to calculate resultant easily.
        cartesian_transform = lambda rho, theta: [np.cos(theta)*rho, np.sin(theta)*rho]
        # Convert cluster FR to cartesian, mapped to 'thetas'.
        transformed_FR = np.array([cartesian_transform(rho, theta) for rho, theta in zip(cluster_rates, thetas)])
        return transformed_FR.sum(axis=0)

    @staticmethod
    def _checkerboard(epochs, frames, pixels, seed):
        """ Create checkerboard array to recreate checkerboard stimulus. This is optimized with
            numba's jit.
            Differing rng:
            python: 6m 28s
            numpy: 18.15s
            numba: ~13s
        """
        # Note that when using numba, nopython mode has limitations on what can be compiled, so astype(int) must be
        # run on 'board' when _checkerboard is called.
        # Eliminated nested for loops in favor of list comprehension with numpy's reshape
        # 283ms vs ~630ms
        # board = np.empty((epochs, frames, pixels), dtype=np.float64)
        random.seed(seed)
        board = np.array([random.random() for _ in range(epochs*frames*pixels)]).reshape((epochs, frames, pixels))
        return ((board < 0.5).astype(int) - 0.5)*2

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
            return start_bound, start_bound + ind_found_time[0]
        elif re_depth != 1:
            return self._separate_stims(start_bound+1, stop_bound, _duration, re_depth=1)
        else:
            return None, None

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
                # _separate_stims returns start and stop bounds, in case start needs to be shifted to capture correct duration.
                stim_ends.append([name, start_bound-2, self._separate_stims(start_bound-2, stop_bound, sus_stim)[1], timing_info])
            else:
                stim_ends.append([name, start_bound, self._separate_stims(start_bound, stop_bound, sus_stim)[1], timing_info])

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
