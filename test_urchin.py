import numpy as np
import json
import math
import random
import pickle
from itertools import compress
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.cluster.hierarchy import ward, average, single, centroid, fcluster
from numba import jit
from scipy.spatial.distance import pdist
import seaborn as sns; sns.set_theme()

%matplotlib inline
%config InlineBackend.figure_format = 'svg'

# Load configuration files
ttl_data = np.load("ss/analysis/data/ttl_data.npy")
config = None
with open("/Users/jeremiahjohn/Library/CloudStorage/Box-Box/Dunn Lab/Users/Rigs/MEA/20210716/20210716_2_completed.json") as config_file:
    config = json.load(config_file)

# Test for _extract_stim_times
stim_list = config['loggedStimuli']
stim_options = ['stepSizes', 'orientations', 'gratingOrientations']
stim_sum = []
for stim in stim_list:
    epoch_time, epoch_num, inter_rep_time = 0, 1, 0
    stim_option = next((option for option in stim_options if option in stim), None)

    epoch_time += stim['_actualPreTime']+stim['_actualStimTime']+stim['_actualTailTime']
    inter_stim_time = stim['_actualInterStimulusInterval'] if '_actualInterStimulusInterval' in stim else 0
    epoch_time += inter_stim_time
    if 'stimulusReps' in stim:
        epoch_num = stim['stimulusReps'] * len(stim[stim_option]) if stim_option is not None else stim['stimulusReps']
        inter_rep_time += stim['stimulusReps'] * stim['_actualInterFamilyInterval'] if '_actualInterFamilyInterval' in stim else 0
    stim_sum.append((stim['protocolName'], epoch_num * epoch_time + inter_rep_time, epoch_num, epoch_time, inter_stim_time, (inter_rep_time/stim['stimulusReps'] if 'stimulusReps' in stim else inter_rep_time) ))

print(stim_sum)
print(f'total: {sum([time for _,time in stim_sum])}')

# Clean TTL pulses of extra and unecessary pulses.
times, pulses = ttl_data[:,0], ttl_data[:,1]
strange_on_times, strange_on_pulses = times[pulses > 0] / 20000, pulses[pulses > 0]
prune_times, prune_pulses = strange_on_times[np.diff(strange_on_times, prepend=1/70) >= 1/70], strange_on_pulses[np.diff(strange_on_times, prepend=1/70) >= 1/70]

# Match times extracted from _extract_stim_times to TTL.
intervals = [(ind, interval) for ind, interval in enumerate(np.diff(prune_times)) if interval >=25]
pause_bounds, non_pause_bounds, stim_bounds = [], [], []
# Detects if obvious interval gap matches with estimated Pause stim time.
# Attempt to improve with list comprehension via functions
def _find_pauses(ind, interval, stim_sum):
    #Find pauses given stim_sum, prune_times intervals >= 25s, and current index in intervals.
    found_pause = [math.isclose(sus_pause,interval, rel_tol=0.1) for _,sus_pause, *_ in stim_sum]
    return [ind, ind+1] if any(found_pause) else None

# Using known indices and queue place for Pause in TTL times, order the remaining stimuli.
# Find bounds outside of known pauses and append as non_pause to search for stim.
def _find_non_pause_bounds(ind, bound):
    # I've returned a 2D list for the list comprehension to work.
    # At this point maybe the for loop is easier to read:
    # for ind, bound in enumerate(pause_bounds):
    #     non_pause_bounds.append([0, bound[1]]) if bound[1] != 0 and ind == 0 else None
    #     if ind is not (len(pause_bounds)-1):
    #         non_pause_bounds.append([bound[2], pause_bounds[ind+1][1]])
    #     else:
    #         non_pause_bounds.append([bound[2], len(prune_times)-1]) if bound[2] != len(prune_times)-1 else None
    # Add to start, as non-pause bound starts at end of pause, but stim starts
    # one TTL pulse after.
    if ind < (len(pause_bounds)-1):
        if (bound[1] == 0 and ind==0):
            return [[0, bound[0]-1], [bound[1]+1, pause_bounds[ind+1][0]-1]]
        else:
            return [[bound[1]+1, pause_bounds[ind+1][0]-1]]
    else:
        return [[bound[1]+1, len(prune_times)-1]] if bound[1] != len(prune_times)-1 else None

def _count_stim_type(stim_name, stim_l):
    # Counts for list of type [('stim_name', stim_time)]
    return [stim_s.count(stim_name) for stim_s in stim_l].count(1)

def _split_pause_chunks():
    pause_in_stim_sum = [ind for ind, stim in enumerate(stim_sum) if 'Pause' in stim]
    #Use indices and loop from 0 to start of pause_in_stim_sum, then [0]:[1] till [len(pause_in_stim_sum)-1]:len(stim_sum)
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

def _separate_stims(start_bound, stop_bound, _duration, re_count=0):
    # separate combination of stims into individuals.
    # stop_bound will be upcoming pause, which will stay the same for
    # every function call in block of loop.
    # The space between TTL pulses (~half the frame rate) is 0.0834.
    # rtol is 0.003, or 3.58% of the pulse spacing.
    # atol is 0.01.
    time_to_find = prune_times[start_bound]+ _duration
    times_to_search = prune_times[start_bound:stop_bound+1]
    found_time = np.isclose(times_to_search, time_to_find, rtol=0.003, atol=0.01) if re_count != 2 else np.isclose(times_to_search, time_to_find, rtol=0.0096, atol=0.01)
    ind_found_time = np.where(found_time == True)[0]
    return (start_bound, start_bound + ind_found_time[0]) if ind_found_time.size != 0 else _separate_stims(start_bound+1, stop_bound, _duration, re_count=1) if re_count !=1 else (None, None)

# Use non_pause_bounds to search for complete or additive combination of stims.
def _find_stim_bounds(start_bound, stop_bound, bound_stims):
    # Find bounds of each non-pause stimulus.
    # Use _separate_stims to find stim start and stop, regardless
    # of individual or combination of stim(s) in non-pause bound.
    # Using match instead for only individual stim might speed things up.
    # [[name, _separate_stims(start_bound, stop_bound, sus_stim)] for name, sus_stim in bound_stims]
    stim_ends = []
    for ind, (name, sus_stim, *timing_info) in enumerate(bound_stims):
        if name == 'Pause':
            # Since non_pause_bounds does not include pause indices (beginning), adjust so pause can be found too.
            # This is for the sake of accurate start_bound shifting.
            str_bnd, stp_bnd = _separate_stims(start_bound-2, stop_bound, sus_stim)
            stim_ends.append([name, str_bnd, stp_bnd, timing_info])
        else:
            str_bnd, stp_bnd = _separate_stims(start_bound, stop_bound, sus_stim)
            stim_ends.append([name, str_bnd, stp_bnd, timing_info])

        start_bound += stim_ends[ind][2]-start_bound if stim_ends[ind][2] is not None else 0

    # bound_stims_ind = [[name, start_bound+stim_ends[ind-1], stim_end] if ind > 0
    #                       else [name, start_bound, stim_end]
    #                       for ind, (name, stim_end) in enumerate(stim_ends)]

    return stim_ends

def _find_stim(stim_name):
    opt = [stim for stim in stim_list if stim['protocolName'] == stim_name]
    return opt
    # for stim in stim_list:
    #     if (stim['protocolName'] == stim_name):
    #         return stim

def _find_sub_stim_bounds(name, start_bound, stop_bound, timing_info):
    # For most stimuli, the actual stimulus is repeated in each epoch; sub-stimulus bounds must be extracted.
    # To determine how many bounds to make (epoch_num), and the time of each bound (epoch_time).
    # stim_bound is (name, start_bound, stop_bound, [epoch_num, epoch_time, interval_time])
    epoch_num, epoch_time, inter_stim_time, interval_time = timing_info
    # To get label for each epoch by change to stimulus (i.e. orientation, intensity)
    valid_option = lambda opt, st: list(compress(opt, [op in st for op in opt]))
    def chunk(ar, n):
        for i in range(0, len(ar), n):
            yield ar[i:i+n]

    stim_options = ['stepSizes', 'orientations', 'gratingOrientations']
    stim_log = ['flashLog', 'orientationLog']
    current_stim = _find_stim(name)[0]
    stim_options = current_stim[valid_option(stim_options,current_stim)[0]] if valid_option(stim_options,current_stim) else 1
    stim_log = current_stim[valid_option(stim_log,current_stim)[0]] if valid_option(stim_log,current_stim) else 1
    stim_log = list(chunk(stim_log, len(stim_options))) if stim_options !=1 else None
    rep_num = epoch_num // len(stim_options) if stim_options != 1 else epoch_num

    sub_rep_ends, sub_epoch_ends = [], []
    for _ in range(rep_num):
        sub_epoch_ends = []
        if interval_time != 0:
            _, inter_rep_ind = _separate_stims(start_bound, stop_bound, interval_time)
            start_bound += inter_rep_ind - start_bound if inter_rep_ind is not None else 0
        for option in range( (len(stim_options) if isinstance(stim_options,list) else stim_options) ):
            # inter_stim_time
            if inter_stim_time != 0:
                test_strt, inter_stim_ind = _separate_stims(start_bound, stop_bound, inter_stim_time)
                start_bound += inter_stim_ind-start_bound if inter_stim_ind is not None else 0

            mod_start_bound, duration_bound = _separate_stims(start_bound,stop_bound, epoch_time-inter_stim_time)
            sub_epoch_ends.append([mod_start_bound, duration_bound])
            start_bound += duration_bound-start_bound if duration_bound is not None else 0

        sub_rep_ends.append(sub_epoch_ends)

    sub_ends = {f'rep_{ind_r}':{f'{opt_label}':e_bounds for opt_label, e_bounds in zip(
                                 ( stim_log[ind_r] if stim_options != 1 else ['epoch']*stim_options ),
                                 epochs_bounds)}
                   for ind_r, epochs_bounds in enumerate(sub_rep_ends)}

    # sub_stim_ends = {f'epoch_{ind}':epoch_bounds for ind, epoch_bounds in zip(range(epoch_num), sub_stim_ends)}
    return sub_ends

def spiketimes_within_bounds(cluster_id, bounds, group_idx=None):
    """ Extract spike times that occur between bound in bounds."""
    # Mask finds times within bounds.
    mask = lambda a_l, st, stp: np.logical_and(a_l >= prune_times[st], a_l <= prune_times[stp])
    peel_nesting = lambda a_l: [a_s for a in a_l for a_s in a]
    # Grouper extracts spike_times for clusters in 'cluster_id' (assumes style of self.extracted_clusters)
    def _grouper(c_idxs):
        # g_id + 1 as g_id is zero indexed, but group folders are 1 indexed.
        # The order of cluster_ids in c_idx is maintained below, to allow
        # mapping of unlabeled spike_times to cluster_id.
        return [_extract_spiketimes(g_id+1, c_ids) for g_id, c_ids in enumerate(c_idxs) if c_ids != 0]

    if isinstance(bounds, dict) and isinstance(cluster_id, int):
        # Multiple sub-stimuli bounds are provided, and spike_times holds one cluster.
        assert group_idx is not None
        spike_times = _extract_spiketimes(group_idx, [cluster_id])
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
        spike_times = _extract_spiketimes(group_idx, [cluster_id])
        return mask(spike_times, start_bound, stop_bound)

def create_PSTH( actual_start_time, actual_stop_time, bounded_spike_times, window_size=50, step=10):
    start_time, stop_time = actual_start_time, actual_stop_time

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
    return (np.array(counts) / (window_size*10**-3)), windows

def _compute_vector_sum(cluster_rates, thetas):
    assert cluster_rates.shape[0] == len(thetas)
    # Transform polar coordinates to cartesian, so as to calculate resultant easily.
    cartesian_transform = lambda rho, theta: [np.cos(theta)*rho, np.sin(theta)*rho]
    # Convert cluster FR to cartesian, mapped to 'thetas'.
    transformed_FR = np.array([cartesian_transform(rho, theta) for rho, theta in zip(cluster_rates, thetas)])
    return transformed_FR.sum(axis=0)

def vector_sum_DSI(cluster_rates, thetas):
    # Sum of FR in each orientation, then the vector_sum coordinates.
    total_magnitude, resultant = cluster_rates.sum(), _compute_vector_sum(cluster_rates, thetas)
    # L2 norm of resultant
    return np.linalg.norm(resultant)/total_magnitude, np.arctan(resultant[1]/resultant[0])

def _checkerboard(epochs, frames, pixels, seed):
    # seed = 0.6107989573688088
    board = np.empty((epochs, frames, pixels), dtype=np.float64)
    random.seed(seed)
    for i in range(epochs):
        for j in range(frames):
            for k in range(pixels):
                board[i,j,k] = (int(random.random() < 0.5) - 0.5)*2
    return board

def generate_RF(start_ind, stop_ind, rep_num, bounded_spike_times, order=0, to_collapse=False, axis=1):
    p_floor = lambda a, p=0: np.floor(a * 10**p) / 10**p
    ## Extract correct stim based on 'order' parameter.
    stim_info = _find_stim('CheckerboardReceptiveField')[order]
    # Find dimensions of the 2D frame
    check_coords = np.array(stim_info['checkCoordinates'])
    frame_height, frame_width = check_coords[check_coords[:,0] == check_coords[0,0]].shape[0], check_coords[check_coords[:,1] == check_coords[0,1]].shape[0]

    ## Create checkerboard for this stim if it hasn't been made yet.
    checkerboard = _checkerboard(
                                stim_info['stimulusReps'],
                                np.ceil(stim_info['_stimTimeNumFrames'] / stim_info['frameDwell']).astype(int),
                                len(check_coords),
                                stim_info['randomSeed']).astype(int)

    ## Fix dropped TTL pulses in on_times during this epoch production.
    stim_times = prune_times[start_ind+1:stop_ind] # start_ind + 1 as bounds should include pre and tail_time.
    expected_frame_interval = stim_info["frameDwell"] / stim_info["_FR"] # frameDwell is in frames, not seconds.
    frame_intervals = np.diff(stim_times)
    poor_samples_mask = ~np.isclose(frame_intervals, expected_frame_interval, rtol=0.001, atol=0.01) # times when interval is not close to expected.
    poor_samples_inds = np.where(poor_samples_mask)[0]
    # int division to find how many frames were dropped, based on expected_frame_interval.
    missing_frames = ((p_floor(frame_intervals[poor_samples_mask], p=2)*100) // (expected_frame_interval*100))
    # Fill in the times of the missing frames based on the values in missing_frames.
    if len(missing_frames) != 0:
        # Indices are all referenced to stim_times, a subset of self.on_times. Slice gets the middle and omits end bounds of linspace.
        missing_times = [np.linspace(stim_times[ind], stim_times[ind+1], missed+2)[1:-1] for ind, missed in zip(poor_samples_inds, missing_frames.astype(int))]
        # To take advantage of numpy insert, we need to flatten missing_times, then map the index to the corresponding time.
        size_missing_times = [len(time) for time in missing_times]
        poor_samples_inds += 1 # Add one as np.insert will place value to replace specificed index, and current inds are left hand bound of time interval.
        # Use mapping to insert flattened missing_times into stim_times. Note that np.ravel is faster than hstack. But this is fine for one execution.
        stim_times = np.insert(stim_times, np.repeat(poor_samples_inds, size_missing_times), np.hstack(missing_times))

    ## Spike-triggered average of pixel values on each frame based on spike_times in bounded_spike_times.
    # Because stim_times is being used to map to win flips in checkerboard axis=1, they have to be
    # of the same size. However, they are often different sizes, with the trend of more TTL pulses than
    # win flips. So kludge is to resize stim_times by cutting out extra pulses in the beginning of the stim.
    extra = len(stim_times) - (np.ceil(stim_info['_stimTimeNumFrames'] / stim_info['frameDwell']).astype(int))
    stim_times_pruned = stim_times[:-extra] if extra != 0 else stim_times
    frames_before = 30
    # Get 30 frames before each spike in total spikes within epoch (rep) bounds. Skip spikes close to the start of the epoch (rep).
    sta = np.array([( np.vstack([checkerboard[rep_num][stim_times_pruned < s_t][-frames_before:], checkerboard[rep_num][stim_times_pruned >= s_t][:10]])
                        if (len(checkerboard[rep_num][stim_times_pruned < s_t][-frames_before:]) == frames_before) and (len(checkerboard[rep_num][stim_times_pruned >= s_t][:10]) == 10)
                        else np.vstack([np.zeros((frames_before, 600)), np.zeros((10, 600)) ]) )
                    for s_t in bounded_spike_times ])
    # Calculate mean of pixel values -30: frames behind spike, and reshape to look like frame.
    #sns.heatmap(sta.mean(axis=0)[38].reshape(30,20).T)
    sta = np.transpose(np.mean(sta, axis=0).reshape(frames_before+10, frame_width, frame_height), axes=(0,2,1))
    # As 'sta' is a 3D array across time, we can compress to 2D to examine temporal changes in pixel intensity
    # of receptive field. We compress by mean across columns.
    sta_temporal = np.mean(sta, axis=axis).T
    return sta_temporal if to_collapse else sta

peel_nesting = lambda a_l: [a_s for a in a_l for a_s in a]
pause_bounds = [_find_pauses(ind,interval,stim_sum) for ind, interval in intervals if _find_pauses(ind,interval,stim_sum) != None]
non_pause_bounds = [_find_non_pause_bounds(ind, bound) for ind, bound in enumerate(pause_bounds)]
non_pause_bounds = [non_pause for non_pause_bound in non_pause_bounds for non_pause in non_pause_bound]
chunks = _split_pause_chunks()
stim_bounds = peel_nesting([_find_stim_bounds(*bound, stim) for bound, stim in zip(non_pause_bounds, chunks)])
print(stim_bounds[-1])
sub_stim_bounds = _find_sub_stim_bounds(*stim_bounds[-1])
print(sub_stim_bounds)
# NOTE missing critical TTL pulse at index 1474, bounding tail_time of rep_1
temp_prune_times = np.insert(prune_times, 1474, prune_times[1473]+5)
plt.eventplot(temp_prune_times[131:3532])
poor_board = np.diff(prune_times[132:808])
sample_RF = np.load("ss/analysis/data/sample_RF.npy")
with open("ss/analysis/data/spikes_first_checks.pkl", 'rb') as f:
    spikes_first_checks = pickle.load(f)
spike_times_22 = np.load("ss/analysis/data/clust_22.npy")
[ind for ind, s_t in enumerate(spikes_first_checks) if len(s_t) >= 1000]
sample_22_RF = generate_RF(131,808,0,spikes_first_checks[3], to_collapse=True, axis=1)
sns.heatmap(sample_22_RF, cmap='viridis')
## Test RF computation ##
# len(generate_RF(1475,2148))
checks = { f'c_{c_n}':None for c_n in range(len(_find_stim('CheckerboardReceptiveField'))) }
check_coords = np.array(_find_stim('CheckerboardReceptiveField')[0]['checkCoordinates'])
frame_height, frame_width = check_coords[check_coords[:,0] == check_coords[0,0]].shape[0], check_coords[check_coords[:,1] == check_coords[0,1]].shape[0]
plt.scatter(check_coords[:,0], check_coords[:,1])

print(sub_stim_bounds)
# The spikes occurring within bound of stimuli time can now be extracted per cluster.
print(f'epoch: {(prune_times[1474]-prune_times[1473])} and actual: {stim_bounds[1][3][1]}')
prune_times[43:47]
# load example spike_times for cluster 10
normalize = lambda a_l: (a_l - min(a_l)) / (max(a_l) - min(a_l))
cl_spiketimes = np.load("ss/analysis/data/cluster_10.npy") / 20000
print(f'start: {prune_times[123]-prune_times[122]}')
_mask_test = lambda a_l, st, stp: np.logical_and(a_l > prune_times[st], a_l < prune_times[stp])
cl_epoch = cl_spiketimes[_mask_test(cl_spiketimes, 46, 48)]
heatmap, window_ref = create_PSTH(prune_times[46], prune_times[48], cl_epoch)
plt.plot(heatmap)
# Looking at where pre-time, stim-time, and post-time happens on heatmap.
print(f'start: {window_ref[208][1] - window_ref[-1][0]}')
print(f'shape: {len(heatmap)}')
ht = heatmap.reshape(1, len(heatmap))
# fig, ax = plt.subplots(figsize=(1,10))
sns.heatmap(ht, yticklabels=False, linewidths=0)
counts, bins = np.histogram(cl_epoch, bins='auto')
## total heatmap test ##
total_heatmap = np.load("ss/analysis/data/total_heatmap.npy")
initial_heatmap = sns.heatmap(total_heatmap, yticklabels=False, linewidths=0)
# Save: initial_heatmap.get_figure().savefig('initial_heatmap.png', dpi=400)

## Test orientation based FR ##
orientation_FR = np.load("ss/analysis/data/avg_orientation_FR.npy", allow_pickle=True)
max_orientation_FR = np.load("ss/analysis/data/PSTH_avg_orientation_FR.npy")
orientation_counts = np.load("ss/analysis/data/avg_orientation_counts.npy")
flash_FR = np.load("ss/analysis/data/avg_flash_FR.npy")
with open("ss/analysis/data/spikes_first_flash.pkl", 'rb') as f:
    spiketimes_first_flash = pickle.load(f)
sns.heatmap(flash_FR[50:59], yticklabels=False, linewidths=0)
orientation_FR[:,60]
orientation_rad = np.array(_find_stim('MovingBar')[0]['orientations'])*np.pi/180
cartesian_transform = lambda rho, theta: (rho*math.cos(theta), rho*math.sin(theta))
# c: 179, 4, 9, 24, 26, 32, 53, 68, 69, 70 (33, 34, 39, 50, 52)
c_i = 170
# cartesian_orientation = np.array([cartesian_transform(rho, theta) for theta, rho in zip(orientation_rad, orientation_FR[:,c_i])])
# plt.plot(cartesian_orientation[:,0], cartesian_orientation[:,1])
plt.polar(np.append(orientation_rad, orientation_rad[0]), np.append(orientation_FR[:,c_i], orientation_FR[0,c_i]))

## DSI test ##
total_dsi = np.array([vector_sum_DSI(orientation_FR[:,tag], orientation_rad)[0] for tag in range(orientation_FR.shape[1])])
good_ds = np.where(total_dsi >=0.20)[0]
good_ds.shape
sns.histplot(total_dsi)

## Comparison test ##
euclidean_dist = lambda x,y: np.sqrt(np.sum((x-y)**2))
cos_sim = lambda x,y: np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
cos_sim(total_heatmap[47], total_heatmap[42])

## Clustering with scipy ##
# using flash_FR, a misnomer as it is the average PSTHs for every cluster.
# remove null flash responses
pruned_total_heatmap = flash_FR
pruned_total_heatmap[np.all(pruned_total_heatmap == 0, axis=1)] = [1 for _ in range(flash_FR.shape[1])]
# cosine distance method creates divide by zero NaN when there are null vectors in data.
d_ma = pdist(pruned_total_heatmap, 'cosine')
d_link = ward(d_ma)
# c1: threshold 0.3 with ward clustering and hamming distance metric.
# c2: threshold 0.8 with ward clustering and cosine distance.
sorted_responses = fcluster(d_link, 0.8, criterion='distance')
clustered_heatmap = [flash_FR[sorted_responses == g_num] for g_num in range(1, sorted_responses.max()+1)]
np.where(sorted_responses==12)[:10]
len(clustered_heatmap[3])
# c2:
# ON_transient (16,8,9): [16][[0,8,9,-2]]
# ON_sustained 11: [11][[2,3,8,13]]
# OFF_transient (0,1,4): [0][[0,1]], [1][-1:], [4][0]
# OFF_sustained 5: [5][[1,2,3,-1]]
# ON-OFF (3,6,7,9,10): [6][4], [9][2], [10][[6,-5]]
# ON DS flash response characterized in group 13, 14.
# ON DS-like. 'turbo' colormap 13, 14: [13][[0,1,3]] [14][[1,3,4]]
# np.vstack([[16][[0,8,9,-2]], [13][[3]], [14][[1,3,4]], [0][[0,1]], [1][-1:], [4][0], [5][[1,2,3,-1]], [6][4], [9][2], [10][[6,-5]]])
sns.heatmap(pruned_total_heatmap[31:35],
            vmin=0,
            cmap="rocket",
            cbar_kws={"label": "spikes/s", "shrink": 0.8},
            rasterized=True,
            linewidth=0, xticklabels=0, yticklabels=0)
plt.savefig("ON_burst.svg", format="svg")
#c1: ON sustained 16, 8, 7. ON burst, then sustained 12
given_clust = np.where(sorted_responses == 1)
sns.heatmap(total_heatmap[143:150], linewidths=0)
# sns.heatmap(total_heatmap[148:150], linewidths=0)
# ind_li = np.where(sorted_responses == 12)[0]
# plt.eventplot(spiketimes_first_flash[ind_li[4]])

## PSTH test ##
m_cont = sio.loadmat('/Users/jeremiahjohn/Library/CloudStorage/Box-Box/Dunn Lab/Users/Jeremiah/exampleSpikeData.mat')
print(m_cont['firingRates'].shape)
print(f'start: {m_cont["spikeTimes"][0][0]}')
ex_s_t = m_cont['spikeTimes'][0] / 1000
ex_heatmap, ex_window_ref = create_PSTH(0, ex_s_t[-1]+0.5, ex_s_t, window_size=40)
print(f'ex_heatmap: {ex_heatmap.shape} and ans: {m_cont["firingRates"][0].shape}')
plt.plot(m_cont['firingRates'][0])
plt.plot(ex_heatmap)
# ---------------------
#   Testing components
# ---------------------
def _test_chunk(s_b, st_b, w_s, s):
    # Won't work cleanly when (st_b - w_s) mod(s) != 0
    t_s = int(((st_b - w_s) - s_b) // s)
    cn = s_b
    for i in range(t_s+1):
        b_b = cn, cn+w_s
        cn += s
        yield b_b if i != t_s else (cn, st_b)
list(_test_chunk(0,3,0.05,0.01))
rng = np.random.default_rng(1)
l_arr = ((rng.random((3,4,4)) > 0.5).astype(int)-0.5)*2


%%timeit
_checkerboard(2,6,5,0.030470408872349197)
random.seed(0.030470408872349197)
[(int(random.random() < 0.5)-0.5)*2 for _ in range(24)]
# time per epoch = pre_time + stim_time + tail_time
# num of epochs = stimulus_reps + step_sizes
#inter family time = len(step_sizes) * inter_family_interval
print(f'{prune_times[-1:]-prune_times[0]} and {(times[-1:]-times[0])/20000}')
print(intervals)
plt.scatter(prune_times,prune_pulses)
