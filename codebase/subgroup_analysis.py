import numpy as np

# ---- identify interacting subgroups of fish based on their positions in current frame ---- 
def get_graph(x,y,L):
    '''
    Input:
        x,y : (N,) [centroid position of all fish at a given time]
        L : interaction distance (mm)
    Returns:
        graph: dict with keys as node indices and values as lists of interacting node indices
    '''
    x = x.reshape(-1)
    y = y.reshape(-1)

    N = x.shape[0]
    graph = {int(node): None for node in range(N)}
    
    for i in range(N):
        distance = (x - x[i])**2 + (y - y[i])**2
        distance[i] = np.nan
        graph[i] = np.flatnonzero(distance<L**2)
    return graph

def _dfs(graph, node, visited, subgroup):
    '''
    Depth-first search to find all connected nodes in a graph starting from a given node.
    '''
    visited[node] = True
    subgroup.append(node)

    for neighbor in graph[node]:
        if not visited[neighbor]:
            _dfs(graph, neighbor, visited, subgroup)

def find_connected_subgroups(graph):
    '''
    This function finds all connected subgroups in a graph using depth-first search (DFS).
    '''
    num_nodes = len(graph)
    visited = [False] * num_nodes
    subgroups = []

    for node in range(num_nodes):
        if not visited[node]:
            subgroup = []
            _dfs(graph, node, visited, subgroup)
            subgroups.append(subgroup)

    return subgroups

# --- identify subgroups of interacting fish in an experiment/simulation & frames for which they exist ---
def _save_and_remove_nonexistent_subgroups(active_subgroups, current_subgroups, archived_subgroups, active_start_frames, current_frame, archived_frames):
    '''
    This function identifies subgroups of fish that no longer exist in the current frame 
     - removes them from "active_subgroups" and adds them to "archived_subgroups"
     - also saves the frame numbers that they started and ended at, which will be used for further analysis

    Inputs:
        active_subgroups : LIST of active subgroups from last frame
        current_subgroups : LIST of subgroups in current frame
        archived_subgroups : LIST of subgroups that no longer exist
        active_start_frames : LIST of starting frame number of each active subgroup
        current_frame : current frame number
        archived_frames : LIST of start & end frame numbers of each archived subgroup. 
            Note on indexing: the last frame that the subgroup exists at is (end-1)! 
    
    Returns:
        active_subgroups, archived_subgroups, active_start_frames, archived_frames
    '''

    # groups to be archived if they no longer exist in current frame
    subgroups_to_archive = [(i, subgroup) for (i, subgroup) in enumerate(active_subgroups) if subgroup not in current_subgroups]
    
    for i, subgroup in subgroups_to_archive:
        archived_subgroups.append(subgroup)
        archived_frames.append([active_start_frames[i], current_frame])  # note start and end frames of each subgroup being archived
        active_subgroups.remove(subgroup)
        active_start_frames[i] = np.nan
    
    # delete starting frames of subgroups that have been archived
    active_start_frames = [frame for frame in active_start_frames if np.isfinite(frame)]
    return active_subgroups, archived_subgroups, active_start_frames, archived_frames

def _add_new_subgroups_to_active(active_subgroups, current_subgroups, active_start_frames, current_frame):
    """
    This function adds any newly formed subgroups to the active subgroups list.
    """
    new_subgroups = [subgroup for subgroup in current_subgroups if subgroup not in active_subgroups]
    for subgroup in new_subgroups:
        active_subgroups.append(subgroup)
        active_start_frames.append(current_frame)
    
    return active_subgroups, active_start_frames


def annotate_subgroups_framewise(x, y, L):
    '''
    This function identifies subgroups of interacting fish, and tracks the start and end frames for each subgroup. 
    Inputs:
        x,y : (N, T) arrays of x and y positions of fish across time
        L : interaction distance
    Returns:
        archived_subgroups : list of all subgroups that existed across time, where each subgroup is a list of fish indices
        archived_frames : list of start and end frames of each subgroup in archived_subgroups. Note that the last frame that the subgroup exists at is (end-1)!
    '''
    N, T = x.shape
    active_subgroups = [] # list of subgroups that are currently active
    active_start_frames = [] # list of starting frames of each active subgroup
    archived_subgroups = [] # list of subgroups that have been archived (i.e. no longer exist in current frame)
    archived_frames = [] # list of start and end frames of each archived subgroup

    for current_frame in range(T):
        if np.all(np.isfinite(x[:,current_frame])): # only analyze frames where all fish positions are valid
            graph = get_graph(x[:, current_frame], y[:, current_frame], L)
            current_subgroups = find_connected_subgroups(graph)

            # check if any active subgroups from last frame no longer exist in current frame, and if so, move them to archived subgroups list
            active_subgroups, archived_subgroups, active_start_frames, archived_frames = _save_and_remove_nonexistent_subgroups(active_subgroups, current_subgroups, archived_subgroups, active_start_frames, current_frame, archived_frames)

            # check if any new subgroups have formed in current frame that were not in active subgroups list, and if so, add them to the list along with their starting frame number
            active_subgroups, active_start_frames = _add_new_subgroups_to_active(active_subgroups, current_subgroups, active_start_frames, current_frame)
        else:
            # if any fish positions are invalid in current frame, we will not analyze this frame and will consider all active subgroups to be ended at the previous frame
            current_subgroups = []
            active_subgroups, archived_subgroups, active_start_frames, archived_frames = _save_and_remove_nonexistent_subgroups(active_subgroups, current_subgroups, archived_subgroups, active_start_frames, current_frame, archived_frames)

    # after looping through all frames, any remaining active subgroups should be moved to archived subgroups list with end frame as T
    current_frame = T
    for i, subgroup in enumerate(active_subgroups):
        archived_subgroups.append(subgroup)
        archived_frames.append([active_start_frames[i], current_frame])  # note start and end frames of each subgroup being archived

    return archived_subgroups, np.vstack(archived_frames)

def get_num_neighbours(x,y,L):
    '''
    Input
        x,y : (N, T) arrays of x and y positions of fish across time
        L : interaction distance
    Output
        n_neighbours: array of number of neighbours of each fish. shape= (N, frames)
    '''
    
    N,T = x.shape
    valid_frames = np.flatnonzero(np.all(np.isfinite(x + y),axis=0))
    neighbours = np.zeros((N,T))
    for i in range(N):
        distance = (x - x[i,:])**2 + (y - y[i,:])**2
        distance[i,:] = np.inf
        neighbours[i,:] = np.count_nonzero(distance<L**2,axis=0)
    return neighbours[:,valid_frames].reshape(-1)

# --- assign a numeric system state variable to each frame based on the subgroup configuration & enumerate transitions ---
def _generate_sum_partitions(n, start=1, path=None, partitions=None):
    ''' Generate all possible partitions of a number n into a sum of positive integers. '''
    if path is None:
        path = []
    if partitions is None:
        partitions = []
    if n == 0:
        partitions.append(path.copy())
        return
    for i in range(start, n + 1):
        _generate_sum_partitions(n - i, i, path + [i], partitions)
    return partitions

def _assign_state_var(curr_state, state_list):
    for i,state in enumerate(state_list):
        if sorted(curr_state)==state:
            return i
    return -1

def _compute_transitions(state_variable, num_states):
    transition_matrix = np.zeros((num_states, num_states))
    for i in range(len(state_variable)-1):
        if state_variable[i] != -1 and state_variable[i+1] != -1:
            transition_matrix[state_variable[i], state_variable[i+1]] += 1
    return transition_matrix

def assign_state_variable_and_get_transitions(x,y,L,states_list=None):
    '''
    This function assigns a numeric system state variable to each frame based on the subgroup configuration in that frame, and computes the transition probabilities between states.
    Inputs:
        x,y : (N, T) arrays of x and y positions of fish across time
        L : interaction distance
        states_list : list of all possible states (optional). If not provided, it will be generated based on the number of fish.
    Returns:
        state_variable : (T,) array of state variable for each frame
        transition_matrix : (num_states, num_states) array of transition counts between states
    '''

    N, T = x.shape
    if states_list is None:
        states_list = _generate_sum_partitions(N)

    system_state_var = np.full(T, -1)  # Initialize with -1 for invalid frames
    valid_frames = np.flatnonzero(np.all(np.isfinite(x), axis=0) & np.all(np.isfinite(y), axis=0))

    # only find system state for valid frames
    for current_frame in valid_frames:
        graph = get_graph(x[:, current_frame], y[:, current_frame], L)
        current_subgroups = find_connected_subgroups(graph)
        subgroup_sizes = [len(subgroup) for subgroup in current_subgroups]
        system_state_var[current_frame] = _assign_state_var(subgroup_sizes, states_list)

    num_states = len(states_list)
    transition_matrix = _compute_transitions(system_state_var, num_states)
    return system_state_var, transition_matrix

# --- spatial density plots ---
def convert_to_focal_distance(i,x,y,theta):
    '''
    Input
        i : focal fish index
        x,y,theta : arrays, each with shape (N,T)
        
    Output
        dFB: front-back distance of all fish wrt focal fish : (N,T)
        dLR: left-right distance of all fish wrt focal fish : (N,T)
        
        [set dFB,dLR for the focal fish itself to be nan]
        
    '''
    N,T = x.shape
    dFB = np.empty((N,T)) #front-back
    dFB[:] = np.nan
    
    dLR = np.empty((N,T)) #left-right 
    dLR[:] = np.nan
    
    dx = x - x[i,:]
    dy = y - y[i,:]
    d = np.sqrt(dx**2 + dy**2)
    
    dx[i,:] = np.nan
    dy[i,:] = np.nan
    d[i,:] = np.nan
    
    viewing_angles = theta[i,:] - np.arctan2(dy,dx)
    
    dFB = d * np.cos(viewing_angles) #dFB>0 implies neighbour in front
    dLR = d * np.sin(viewing_angles) #dLR>0 implies neighbour to the right

    return dFB, dLR