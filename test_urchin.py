import numpy as np
import json
import math
from itertools import compress
import matplotlib.pyplot as plt

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
    stim_sum.append((stim['protocolName'], epoch_num * epoch_time + interval_time))

print(stim_sum)
print(f'total: {sum([time for _,time in stim_sum])}')

# Clean TTL pulses of extra and unecessary pulses.
times, pulses = ttl_data[:,0], ttl_data[:,1]
strange_on_times, strange_on_pulses = times[pulses > 0] / 20000, pulses[pulses > 0]
prune_times, prune_pulses = strange_on_times[np.diff(strange_on_times) >= 1/70], strange_on_pulses[np.ediff1d(strange_on_times, to_begin=1/70) >= 1/70]

# Match times extracted from _extract_stim_times to TTL.
intervals = [(ind, interval) for ind, interval in enumerate(np.diff(prune_times)) if interval >=25]
found_pauses = []
# Detects if obvious interval gap matches with estimated Pause stim time.
for ind, interval in intervals:
    found_pause = [math.isclose(sus_pause,interval, rel_tol=0.1) for _,sus_pause in stim_sum]
    found_pauses.append(('Pause', ind, interval)) if any(found_pause) else None
# Using known indices and queue place for Pause in TTL times, order
# the remaining stimuli.


# ---------------------
#   Testing components
# ---------------------

# time per epoch = pre_time + stim_time + tail_time
# num of epochs = stimulus_reps + step_sizes
#inter family time = len(step_sizes) * inter_family_interval
print(f'{prune_times[-1:]-prune_times[0]} and {(times[-1:]-times[0])/20000}')
print(intervals)
plt.scatter(prune_times,prune_pulses)
