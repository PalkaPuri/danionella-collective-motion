import numpy as np
from scipy.signal import find_peaks

def _proofread_detected_peaks_valleys(peaks,valleys,values):
    '''
    Proofread detected peaks and valleys to select alternating valleys & peaks, resulting in complete burst-and-glide cycles.
    Parameters:
        peaks, valleys - arrays of time indices of detected peaks and valleys
        values - velocity values in segment 
    Returns:
        proofread_cycles -- array of shape (N_cycles, 3) consisting of time indices of consecutive valley, peak, 
        valley constituting a complete burst-and-glide cycle
    '''
    if len(peaks) == 0 or len(valleys) < 2:
        return np.array([]).reshape(-1,3)

    proofread_peaks = []
    proofread_valleys = []

    # start sequence with a valley
    current_valley_idx = 0
    current_peak_idx = np.flatnonzero(peaks>valleys[0])[0] #start with the peak that follows the first valley

    while current_valley_idx < len(valleys) - 1 and current_peak_idx < len(peaks):
        # all detected peaks before next valley
        candidate_peak_idxs = current_peak_idx + np.flatnonzero(peaks[current_peak_idx:] < valleys[current_valley_idx+1])
        if len(candidate_peak_idxs):
            # select the highest peak
            best_candidate = np.argmax(values[peaks[candidate_peak_idxs]])
            current_peak_idx = candidate_peak_idxs[best_candidate]

            # save the valley and peak pair
            proofread_peaks.append(peaks[current_peak_idx])
            proofread_valleys.append(valleys[current_valley_idx])

            #print(current_peak_idx,current_valley_idx)
            # update idxs
            current_valley_idx += 1
            current_peak_idx = candidate_peak_idxs[-1] + 1

        else: #this implies that we have multiple consecutive valleys
            candidate_valley_idxs = current_valley_idx+ np.flatnonzero(valleys[current_valley_idx:] < peaks[current_peak_idx])
            # select the deepest valley
            best_candidate = np.argmin(values[valleys[candidate_valley_idxs]])
            current_valley_idx = candidate_valley_idxs[best_candidate]

            # save the peak,valley pair
            proofread_peaks.append(peaks[current_peak_idx])
            proofread_valleys.append(valleys[current_valley_idx])

            #print(current_peak_idx,current_valley_idx)
            # update idxs
            current_valley_idx = candidate_valley_idxs[-1] + 1
            current_peak_idx +=1 

    # after adding pairs if there is a valley left at the end, add it
    if current_valley_idx < len(valleys):
        proofread_valleys.append(valleys[current_valley_idx])

    # convert to arrays
    proofread_peaks = np.array(proofread_peaks).astype(int)
    proofread_valleys = np.array(proofread_valleys).astype(int)

    # ensure that the sequence ends with a valley, i.e. we have complete burst-glide cycles within each segment
    if proofread_valleys[-1] < proofread_peaks[-1] :
        proofread_peaks = proofread_peaks[:-1] # remove the last peak

    # save the cycles
    proofread_cycles = np.empty((len(proofread_peaks),3),dtype=int)
    proofread_cycles[:,0] = proofread_valleys[:-1]
    proofread_cycles[:,1] = proofread_peaks
    proofread_cycles[:,2] = proofread_valleys[1:]
    
    return proofread_cycles


def find_complete_burst_glide_cycles(v, min_burst_duration, min_glide_duration, prominence_threshold):
    ''' 
    Annotate burst & glide cycles from smoothed velocity v. 
    Note v has shape (T,), and is the velocity of one specific fish in the experiment.
    Function returns an array with three columns : 
    cycle start (i.e. valley), peak locations, cycle end (i.e valley) (in frames) for each identified cycle 
    '''

    # Find continuous segments of non-nan data
    is_nan = np.isnan(v)
    split_points = np.where(is_nan[1:] != is_nan[:-1])[0] + 1

    # Split the array at change points
    segments = np.split(v, split_points)
    frames = np.split(np.arange(len(v)),split_points)

    burst_glide_cycles = [np.array([]).reshape(-1,3)]

    for seg,f_idx in zip(segments,frames):
        if len(seg) > 0 and np.all(np.isfinite(seg)):
            # Find peaks/valleys in this segment
            peaks, _ = find_peaks(seg,prominence=prominence_threshold,width=min_burst_duration,distance=min_glide_duration+min_burst_duration)
            valleys, _ = find_peaks(-seg,prominence=prominence_threshold,width=min_glide_duration,distance=min_burst_duration+min_glide_duration)

            # proofread within each segment and get cycles
            segment_bg_cycles = _proofread_detected_peaks_valleys(np.array(peaks),np.array(valleys),seg)
            
            # IMPORTANT! convert from segment index to frames index
            if len(segment_bg_cycles):
                segment_bg_cycles = f_idx[segment_bg_cycles]
                burst_glide_cycles.append(segment_bg_cycles)        
    return np.concatenate(burst_glide_cycles).astype(int)

def compute_pairwise_response_features(focal_fish_idx, neighbor_idxs, burst_glide_cycles_all, head_angles_all, x_all, y_all, v_all, arena_radius, fps=121, seed=42):
    rng = np.random.default_rng(seed)

    # start and peak of response for focal fish
    cycle_starts = burst_glide_cycles_all[focal_fish_idx][:,0] # start of burst
    peaks = burst_glide_cycles_all[focal_fish_idx][:,1] # end of burst
    N = len(cycle_starts)

    # compute each feature and save to dictionary
    features = {}

    for i,idx in enumerate(neighbor_idxs):
        #pairwise distance
        pairwise_distance = np.sqrt((x_all[focal_fish_idx,:]-x_all[idx,:])**2 + (y_all[focal_fish_idx,:]-y_all[idx,:])**2)
        features[f"initial_L_{i}"] = pairwise_distance[cycle_starts]
        features[f"delta_L_{i}"] = pairwise_distance[peaks] - pairwise_distance[cycle_starts]

        # reaction time: time since last peak of neighbor
        neighbor_peaks = burst_glide_cycles_all[idx][:,1]
        t_ind = np.searchsorted(a=neighbor_peaks, v =cycle_starts) - 1
        features[f"reaction_time_{i}"] = (cycle_starts- neighbor_peaks[t_ind])/fps + (rng.random(N)-0.5)/fps # uniform jitter by +/- 0.5 frame
        # set this to nan if no neighbor peak before focal fish burst
        features[f"reaction_time_{i}"][t_ind<0] = np.nan

        # polarization
        polarization = np.sqrt(np.mean(np.cos(head_angles_all[[focal_fish_idx,idx],:]),axis=0)**2 + np.mean(np.sin(head_angles_all[[focal_fish_idx,idx],:]),axis=0)**2)
        features[f"initial_m_{i}"] = polarization[cycle_starts]
        features[f"delta_m_{i}"] = polarization[peaks] - polarization[cycle_starts]

        # cosine of avoidance angle : dot product initial focal fish heading and vector joining the fish pair
        phi = np.arctan2(y_all[idx,cycle_starts]-y_all[focal_fish_idx,cycle_starts],x_all[idx,cycle_starts]-x_all[focal_fish_idx,cycle_starts])
        features[f"cosine_avoidance_{i}"] = np.cos(head_angles_all[focal_fish_idx,cycle_starts]-phi)

    # cumulative change in heading angle over the burst
    dtheta_per_frame = head_angles_all[focal_fish_idx,1:] - head_angles_all[focal_fish_idx,:-1]
    # keep the smaller change in angle
    dtheta_per_frame[dtheta_per_frame<-np.pi] = dtheta_per_frame[dtheta_per_frame<-np.pi] + 2*np.pi
    dtheta_per_frame[dtheta_per_frame>np.pi] = dtheta_per_frame[dtheta_per_frame>np.pi] - 2*np.pi
    cumsum_dtheta_per_frame = np.append(0,np.cumsum(dtheta_per_frame))
    features["cum_delta_theta"] = cumsum_dtheta_per_frame[peaks] - cumsum_dtheta_per_frame[cycle_starts]

    # length of burst
    features["T_burst"] = (peaks - cycle_starts)/fps + (rng.random(N)-0.5)/fps # uniform jitter by +/- 0.5 frame

    # initial distance to wall 
    features["distance_to_wall"] = arena_radius - np.sqrt((x_all[focal_fish_idx,cycle_starts] - arena_radius)**2 + (y_all[focal_fish_idx,cycle_starts] - arena_radius)**2)

    # delta_v
    features["delta_v"] = v_all[focal_fish_idx,peaks] - v_all[focal_fish_idx, cycle_starts]

    # sort all neighbor-wise features by initial distance to focal fish
    neighbor_distances = np.array([features[f"initial_L_{i}"] for i in range(len(neighbor_idxs))])
    sorted_neighbor_idxs = np.argsort(neighbor_distances,axis=0)

    features_sorted = features.copy()
    neighborwise_features = ["initial_L","delta_L","reaction_time","initial_m","delta_m","cosine_avoidance"]
    for f in neighborwise_features:
        f_values = np.array([features[f"{f}_{i}"] for i in range(len(neighbor_idxs))])
        f_values_sorted = np.take_along_axis(f_values, sorted_neighbor_idxs, axis=0)
        for i in range(len(neighbor_idxs)):
            features_sorted[f"{f}_{i}"] = f_values_sorted[i]

    return features_sorted

    