# Urchin
A tool to perform batch analyses on spike-sorted multielectrode array data.
## How to run
Since this is not a package at this time, the file urchin.py contains the Urchin class
used to run analysis. Necessary dependencies include numpy, scipy, and h5py.

The urchin class takes three parameters:

|       Parameter      |                Description               |
| -------------------- | ---------------------------------------- |
| path_to_sorted_data  | Directory with spike-sorted data         |
| path_to_raw_data     | Path to raw .h5 data                     |
| path_to_json         | Path to JSON stimulus config file        |

The analysis scheme happens as follows:
1. Each stimuli in the config file must be assigned to bounds dictated by the TTL pulses.
   - This is run when Urchin is constructed, and stored in the object parameter `stim_bounds`, where each value in the list is a separate stimulus.

2. Any sub-stimuli timing features must now be found, which may or may not be present depending on the stimulus. For example, a MovingBar stimuli has multiple sub-stimuli for each direction of the bar.
   - These sub-stimuli features can be determined by passing in the index in `stim_bounds` of the stimuli to analyze to the function `split_into_sub_stimuli`. The output of this function is used to perform batch analysis.

3. Using the output of `split_into_sub_stimuli`, the spike times for each 'good' cluster marked on Phy must be extracted within the timing bounds of the sub-stimuli.
   - Pass this output into the function `extract_spiketimes_within_bounds`, which extracts spike times within the bounds of the sub-stimuli.

4. At this point, several spike count based analyses can be performed, such as the PSTH for a cluster, the receptive field of the cluster, the vector DSI of a cluster, or the average firing rate of a cluster over the sub-stimulus time period.

## Available Analyses
### generate_PSTH
### generate_RF
### vector_DSI
### coarse_FR
