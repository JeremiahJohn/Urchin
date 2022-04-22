import numpy as np
import json
import math
from itertools import compress
import matplotlib.pyplot as plt
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
    epoch_time, epoch_num, interval_time = 0, 1, 0
    stim_option = next((option for option in stim_options if option in stim), None)

    epoch_time += stim['_actualPreTime']+stim['_actualStimTime']+stim['_actualTailTime']
    epoch_time += stim['_actualInterStimulusInterval'] if '_actualInterStimulusInterval' in stim else 0
    if 'stimulusReps' in stim:
        epoch_num = stim['stimulusReps'] * len(stim[stim_option]) if stim_option is not None else stim['stimulusReps']
        interval_time += stim['stimulusReps'] * stim['_actualInterFamilyInterval'] if '_actualInterFamilyInterval' in stim else 0
    stim_sum.append((stim['protocolName'], epoch_num * epoch_time + interval_time, epoch_num, epoch_time, (interval_time/stim['stimulusReps'] if 'stimulusReps' in stim else interval_time) ))

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
    pause_in_stim_sum = [ind for ind, stim in enumerate(stim_sum) for sub in stim if sub == 'Pause']
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

def _separate_stims(start_bound, stop_bound, _duration):
    # separate combination of stims into individuals.
    # stop_bound will be upcoming pause, which will stay the same for
    # every function call in block of loop.
    # The space between TTL pulses (~half the frame rate) is 0.0834.
    # rtol is 0.003, or 3.58% of the pulse spacing.
    time_to_find = prune_times[start_bound]+ _duration
    times_to_search = prune_times[start_bound:stop_bound+1]
    found_time = np.isclose(times_to_search, time_to_find, rtol=0.003, atol=0.01)
    ind_found_time = np.where(found_time == True)[0]
    return start_bound + ind_found_time[0] if ind_found_time.size != 0 else None

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
            stim_ends.append([name, start_bound-2, _separate_stims(start_bound-2, stop_bound, sus_stim), timing_info])
        else:
            stim_ends.append([name, start_bound, _separate_stims(start_bound, stop_bound, sus_stim), timing_info])

        start_bound += stim_ends[ind][2]-start_bound if stim_ends[ind][2] is not None else 0

    # bound_stims_ind = [[name, start_bound+stim_ends[ind-1], stim_end] if ind > 0
    #                       else [name, start_bound, stim_end]
    #                       for ind, (name, stim_end) in enumerate(stim_ends)]

    return stim_ends

def _find_stim(stim_name):
    for stim in stim_list:
        if (stim['protocolName'] == stim_name):
            return stim

def _find_sub_stim_bounds(name, start_bound, stop_bound, timing_info):
    # For most stimuli, the actual stimulus is repeated in each epoch; sub-stimulus bounds must be extracted.
    # To determine how many bounds to make (epoch_num), and the time of each bound (epoch_time).
    # stim_bound is (name, start_bound, stop_bound, [epoch_num, epoch_time, interval_time])
    epoch_num, epoch_time, interval_time = timing_info
    # To get label for each epoch by change to stimulus (i.e. orientation, intensity)
    valid_option = lambda opt, st: list(compress(opt, [op in st for op in opt]))
    def chunk(ar, n):
        for i in range(0, len(ar), n):
            yield ar[i:i+n]

    stim_options = ['stepSizes', 'orientations', 'gratingOrientations']
    stim_log = ['flashLog', 'orientationLog']
    current_stim = _find_stim(name)
    stim_options = current_stim[valid_option(stim_options,current_stim)[0]] if valid_option(stim_options,current_stim) else 1
    stim_log = current_stim[valid_option(stim_log,current_stim)[0]] if valid_option(stim_log,current_stim) else 1
    stim_log = list(chunk(stim_log, len(stim_options))) if stim_options !=1 else None
    rep_num = epoch_num // len(stim_options) if stim_options != 1 else epoch_num

    sub_rep_ends, sub_epoch_ends = [], []
    for _ in range(rep_num):
        sub_epoch_ends = []
        for option in range( (len(stim_options) if isinstance(stim_options,list) else stim_options) ):
            sub_epoch_ends.append([start_bound, _separate_stims(start_bound,stop_bound, epoch_time)])
            start_bound += sub_epoch_ends[option][1]-start_bound if sub_epoch_ends[option][1] is not None else 0
            if interval_time != 0:
                sub_epoch_ends.append([start_bound, _separate_stims(start_bound,stop_bound, interval_time)])
                start_bound += sub_epoch_ends[option][1]-start_bound if sub_epoch_ends[option][1] is not None else 0

        sub_rep_ends.append(sub_epoch_ends)

    sub_ends = {f'rep_{ind_r}':{f'{opt_label}':e_bounds for opt_label, e_bounds in zip(
                                 ( stim_log[ind_r] if stim_options != 1 else ['epoch']*stim_options ),
                                 epochs_bounds)}
                   for ind_r, epochs_bounds in enumerate(sub_rep_ends)}

    # sub_stim_ends = {f'epoch_{ind}':epoch_bounds for ind, epoch_bounds in zip(range(epoch_num), sub_stim_ends)}
    return sub_ends

def spiketimes_within_bounds(spike_times, bounds):
    """ Extract spike times that occur between bound in bounds."""
    # Mask finds times within bounds.
    mask = lambda a_l, st, stp: np.logical_and(a_l >= prune_times[st], a_l <= prune_times[stp])
    # Grouper extracts spike_times for clusters in 'cluster_id' (assumes stye of self.extracted_clusters)
    def _grouper(c_idxs):
        for g_id, c_ids in enumerate(c_idxs):
            if c_ids != 0:
                for c_id in c_ids:
                    # g_id + 1 as g_id is zero indexed, but group folders are 1 indexed.
                    yield _extract_spiketimes(g_id+1, c_id)

    if isinstance(bounds, dict) and isinstance(cluster_id, int):
        # Multiple sub-stimuli bounds are provided, and spike_times holds one cluster.
        assert group_idx is not None
        spike_times = _extract_spiketimes(group_idx, cluster_id)
        return {option:spike_times[mask(spike_times, bound[0], bound[1])] for option, bound in bounds.items()}
    elif isinstance(bounds, dict) and isinstance(cluster_id, list):
        # Multiple sub-stimuli bounds are provided, but spike_times holds multiple clusters.
        return {option:[s_t[mask(s_t, bound[0], bound[1])] for s_t in list(_grouper(cluster_id))] for option, bound in bounds.items()}
    elif isinstance(cluster_id, list):
        # Single bounds is provided, but spike_times holds multiple clusters.
        asssert group_idx is not None
        start_bound, stop_bound = bounds
        spike_times = _extract_spiketimes(group_idx, cluster_id)
        return [spike_times[mask(spike_times, start_bound, stop_bound)] for spike_time in spike_times]
    else:
        # Single bounds and single cluster.
        assert group_idx is not None
        start_bound, stop_bound = bounds
        spike_times = _extract_spiketimes(group_idx, cluster_id)
        return spike_times[mask(spike_times, start_bound, stop_bound)]

def create_PSTH(bounded_spike_times, window_size=50, step=10):
    start_time, stop_time = bounded_spike_times[0], bounded_spike_times[-1:]

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


peel_nesting = lambda a_l: [a_s for a in a_l for a_s in a]
pause_bounds = [_find_pauses(ind,interval,stim_sum) for ind, interval in intervals if _find_pauses(ind,interval,stim_sum) != None]
non_pause_bounds = [_find_non_pause_bounds(ind, bound) for ind, bound in enumerate(pause_bounds)]
non_pause_bounds = [non_pause for non_pause_bound in non_pause_bounds for non_pause in non_pause_bound]
chunks = _split_pause_chunks()
stim_bounds = peel_nesting([_find_stim_bounds(*bound, stim) for bound, stim in zip(non_pause_bounds, chunks)])
print(stim_bounds[4])
sub_stim_bounds = _find_sub_stim_bounds(*stim_bounds[4])
print(sub_stim_bounds)
# The spikes occurring within bound of stimuli time can now be extracted per cluster.
print(f'epoch: {(prune_times[48]-prune_times[46]) - stim_bounds[3][3][1]}')

# load example spike_times for cluster 10
normalize = lambda a_l: (a_l - min(a_l)) / (max(a_l) - min(a_l))
cl_spiketimes = np.load("ss/analysis/data/cluster_10.npy") / 20000
cl_epoch = cl_spiketimes[np.logical_and(cl_spiketimes >= prune_times[120], cl_spiketimes <= prune_times[127])]
heatmap = create_PSTH(cl_epoch)
ht = heatmap.reshape(1, len(heatmap))
fig, ax = plt.subplots(figsize=(1,10))
sns.heatmap(ht, xticklabels=False, yticklabels=False)
counts, bins = np.histogram(cl_epoch, bins='auto')

epoch_t = spiketimes_within_bounds(cl_spiketimes,sub_stim_bounds['rep_1'])
print(epoch_t['epoch'].shape)
plt.hist(cl_epoch_n, bins='auto')



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
# time per epoch = pre_time + stim_time + tail_time
# num of epochs = stimulus_reps + step_sizes
#inter family time = len(step_sizes) * inter_family_interval
print(f'{prune_times[-1:]-prune_times[0]} and {(times[-1:]-times[0])/20000}')
print(intervals)
plt.scatter(prune_times,prune_pulses)
