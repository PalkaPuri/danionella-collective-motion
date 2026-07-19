import numpy as np
from scipy.optimize import curve_fit, least_squares

def get_polarization_by_subgroup_size(selected_subgroups, selected_frames, head_angles,N=4):
    '''
    Calculate polarization for each frame a subgroup exists for. Results are grouped by subgroup size.
    Input:
        selected_subgroups : list of subgroups to analyze; each subgroup consists of a list of fish indices
        selected_frames : list of (start,end) frame indices for each subgroup in selected_subgroups. 
        head_angles : (N,T) array of heading angles of all fish across time
    Returns:
        polarization_by_size : dictionary where keys are subgroup sizes and values are arrays with framewise mx,my for each subgroup
    '''
    polarization_by_size = {size: [] for size in range(2, N+1)} # initialize dictionary to store polarization by subgroup size
    for subgroup, frames in zip(selected_subgroups, selected_frames):
        size = len(subgroup)
        group_heads = head_angles[subgroup, frames[0]:frames[1]]
        mx = np.mean(np.cos(group_heads),axis=0)
        my = np.mean(np.sin(group_heads),axis=0)
        polarization_by_size[size].append(np.column_stack((mx,my)))
    return polarization_by_size

def _autocorr(x,dt):
    x1 = x[:-dt]
    x2 = x[dt:]
    return np.corrcoef(x1, x2)[0, 1]

def get_autocorrelation_and_avg_timescale(subgroup_polarization, dt_array):
    '''
    Calculate autocorrelation of m_x and m_y for each subgroup.
    Input:
        subgroup_polarization : list of arrays of framewise mx,my for each subgroup
        dt_array : array of time lags to calculate autocorrelation for (in frames)
    Returns:
        autocorrelation : (2, num_subgroups, len(dt_array)) array of autocorrelation of m_x and m_y for each subgroup at each time lag
        avg_timescale :  decay timescale of autocorrelation averaged across all subgroups & m_x,m_y (in frames)
    '''
    autocorrelation = np.zeros((2, len(subgroup_polarization), len(dt_array)))
    for i, subgroup in enumerate(subgroup_polarization):
        mx = subgroup[:, 0]
        my = subgroup[:, 1]
        
        for j, dt in enumerate(dt_array):
            if dt == 0:
                autocorrelation[:, i, j] = 1
            else:
                autocorrelation[0, i, j] = _autocorr(mx, dt)
                autocorrelation[1, i, j] = _autocorr(my, dt)

    # calculate decay timescale for the average
    def exp_decay(x, timescale):
        return np.exp(-x / timescale)
    
    avg_autcorr = np.mean(autocorrelation, axis=(0,1))  # average over subgroups and m_x/m_y
    nan_mask = np.isfinite(avg_autcorr)
    fit, _ = curve_fit(exp_decay, dt_array[nan_mask], avg_autcorr[nan_mask])
    return autocorrelation, fit[0]

def get_first_jump_moments(subgroup_polarization, dt, fps=121):
    '''
    Input:
        subgroup_polarization: list of arrays of framewise mx,my for each subgroup
        dt: time step (in frames) for first jump moments calculation
        fps : frame rate (Hz)

    Returns:
        jump_moments: array with 4 columns: mx, my, dmx, dmy
        '''

    jump_moments = []
    for m in subgroup_polarization:
        mx = m[:,0]
        my = m[:,1]
        dmx = (mx[dt:] - mx[:-dt])/dt*fps
        dmy = (my[dt:] - my[:-dt])/dt*fps
        
        jump_moments.append(np.column_stack(
            (mx[:-dt], my[:-dt], dmx, dmy)))
        
    if len(jump_moments):      
        # stack all jump moments from all subgroups        
        return np.vstack(jump_moments)
    else:
        return np.empty((0, 4))

def get_second_jump_moments(subgroup_polarization, dt, fps=121):
    ''' 
    Input:
        subgroup_polarization: list of arrays of framewise mx,my for each subgroup
        dt: time step (in frames) for second jump moments calculation
        fps : frame rate (Hz)
    Returns:
        jump_moments: array with 5 columns: mx, my, dmxx, dmyy, dmxy
    '''

    jump_moments = []
    for m in subgroup_polarization:
        mx = m[:,0]
        my = m[:,1]
        dmxx = (mx[dt:] - mx[:-dt])*(mx[dt:] - mx[:-dt])/dt*fps
        dmyy = (my[dt:] - my[:-dt])*(my[dt:] - my[:-dt])/dt*fps
        dmxy = (mx[dt:] - mx[:-dt])*(my[dt:] - my[:-dt])/dt*fps
        
        jump_moments.append(np.column_stack((mx[:-dt], my[:-dt], dmxx, dmyy, dmxy)))

    if len(jump_moments):  
        return np.vstack(jump_moments)
    else:
        return np.empty((0, 5))
    
def _avg_over_m(mx,my,F,nbins):
    '''
    Average F(mx,my) over bins in mx and my. Helper function for fitting alpha and beta from first and second jump moments.

    Input
        mx, my, F : 1D arrays of data points of same length
        nbins : number of bins to use for mx and my

    Returns:
        avg_F = (nbins, nbins) array. 
            avg_F[i,j] is  F averaged over data points whose my falls in the i-th bin and mx falls in the j-th bin.
    '''
    bin_edges = np.linspace(-1,1,nbins+1)
    bin_indices_x = np.digitize(mx,bin_edges) - 1
    bin_indices_y = np.digitize(my,bin_edges) - 1
    avg_F = np.empty((len(bin_edges)-1,len(bin_edges)-1))
    avg_F[:] = np.nan
    for i in range(nbins):
        for j in range(nbins):
            mask = np.logical_and(bin_indices_y==i, bin_indices_x==j)
            if np.any(mask):
                avg_F[i,j] = np.nanmean(F[mask])
    return avg_F

def _Bij_inverse_function(dmxx, dmyy, dmxy):
    '''
    Convert second jump moments to B_xx, B_yy, B_xy. Helper function for fitting alpha and beta from first and second jump moments.

    Input:
        dmxx,dmyy,dmxy: arrays or matrices of second jump moments 
    Returns:
        B_xx,B_yy,B_xy: same dimension as the inputs
    '''

    B_xy2 = dmxy**2 / ((dmxx+dmyy)+2*np.sqrt(dmxx*dmyy-dmxy**2))
    B_xx2 = (dmxx + np.sqrt(dmxx*dmyy-dmxy**2))**2/((dmxx+dmyy)+2*np.sqrt(dmxx*dmyy-dmxy**2))
    B_yy2 = (dmyy + np.sqrt(dmxx*dmyy-dmxy**2))**2/((dmxx+dmyy)+2*np.sqrt(dmxx*dmyy-dmxy**2))
    return B_xx2, B_yy2, B_xy2
    
def fit_alpha_beta_jointly_from_first_second_jump_moments(first_jump_moments, second_jump_moments, N, nbins=20):
    '''
    Fit model parameters alpha and beta jointly from first and second jump moments data.
    Input:
        first_jump_moments : array with 4 columns: mx, my, dmx, dmy
        second_jump_moments : array with 5 columns: mx, my, dmxx, dmyy, dmxy
        N : number of fish in the group
        nbins : number of bins to use for averaging over mx and my
    Returns:
        alpha_fit : fitted value of alpha
        beta_fit : fitted value of beta
        r_squared : average R^2 across all equations and components 
    '''    

    # ---------------------------------------------
    # Step 1. Average jump moment data 
    # ---------------------------------------------
    mx_first = _avg_over_m(first_jump_moments[:,0], first_jump_moments[:,1], first_jump_moments[:,0], nbins)
    my_first = _avg_over_m(first_jump_moments[:,0], first_jump_moments[:,1], first_jump_moments[:,1], nbins)
    dmx = _avg_over_m(first_jump_moments[:,0], first_jump_moments[:,1], first_jump_moments[:,2], nbins)
    dmy = _avg_over_m(first_jump_moments[:,0], first_jump_moments[:,1], first_jump_moments[:,3], nbins)

    mx_second = _avg_over_m(second_jump_moments[:,0], second_jump_moments[:,1], second_jump_moments[:,0], nbins)
    my_second = _avg_over_m(second_jump_moments[:,0], second_jump_moments[:,1], second_jump_moments[:,1], nbins)
    m2 = mx_second**2 + my_second**2
    dmxx = _avg_over_m(second_jump_moments[:,0], second_jump_moments[:,1], second_jump_moments[:,2], nbins)
    dmyy = _avg_over_m(second_jump_moments[:,0], second_jump_moments[:,1], second_jump_moments[:,3], nbins)
    dmxy = _avg_over_m(second_jump_moments[:,0], second_jump_moments[:,1], second_jump_moments[:,4], nbins)

    B_xx2, B_yy2, _ = _Bij_inverse_function(dmxx, dmyy, dmxy)
    
    # ---------------------------------------------
    # Step 2. Define residual function for joint 
    # fitting of alpha and beta
    # ---------------------------------------------
    
    def f(x, alpha):
        """Equation for first jump moments"""
        return -alpha * x  
    
    def g(x, alpha, beta):
        """Equation for second jump moments"""
        return (alpha + (1-x) * beta) / N
    
    
    def residuals(params, mx_first, my_first, dmx, dmy, m2, B_xx2, B_yy2):
        """
        Compute residual vector for joint non-linear least squares fitting
        Input:
            params : [alpha, beta] - parameters to fit
            mx_first, my_first : 2D arrays of averaged mx and my for first jump moments
            dmx, dmy : 2D arrays of averaged dmx and dmy for first jump moments
            m2 : 2D array of mx^2 + my^2 for second jump moments
            B_xx2, B_yy2 : 2D arrays of B_xx and B_yy for second jump moments                   
        Returns:
            residuals : array
                Residual values for each data point and equation [res1_1, res1_2, ..., res2_1, res2_2, ...]
        """
        alpha, beta = params
        
        # Calculate residuals for each equation and x,y components
        res1_x = dmx - f(mx_first, alpha)  # First jump moments x-component
        res1_y = dmy - f(my_first, alpha)  # First jump moments y-component
        res2_x = B_xx2 - g(m2, alpha, beta)  # Second jump moments x-component
        res2_y = B_yy2 - g(m2, alpha, beta)  # Second jump moments y-component
        
        # Return ALL individual residuals 
        return np.concatenate([res1_x.ravel(),
                                res1_y.ravel(),
                                res2_x.ravel(),
                                res2_y.ravel()])
    
    # -----------------------------------------------
    # Step 3. Fit parameters by minimizing residuals 
    # across all equations and data points
    # -----------------------------------------------
    initial_guess = [0.1, 0.1]  # Initial guess for [alpha, beta]
    bounds = ([0, 0], [np.inf, np.inf])  # Bounds for parameters
    nan_mask_1 = np.logical_and(np.isfinite(dmx),np.isfinite(dmy))
    nan_mask_2 = np.logical_and(np.isfinite(B_xx2),np.isfinite(B_yy2))
    result = least_squares(residuals, initial_guess, args=(mx_first[nan_mask_1], 
                                                           my_first[nan_mask_1], 
                                                           dmx[nan_mask_1], 
                                                           dmy[nan_mask_1], 
                                                           m2[nan_mask_2], 
                                                           B_xx2[nan_mask_2], 
                                                           B_yy2[nan_mask_2]), bounds=bounds)

    alpha_fit = result.x[0]
    beta_fit = result.x[1]
    
    # -----------------------------------------------
    # Step 4. Calculate R^2 for each equation
    # -----------------------------------------------

    dmx_fit = f(mx_first, result.x[0])
    nan_mask = np.isfinite(dmx)
    ss_residuals_x = np.sum((dmx[nan_mask] - dmx_fit[nan_mask])**2)
    ss_total_x = np.sum((dmx[nan_mask] - np.nanmean(dmx))**2)
    r_squared_1_x = 1 - (ss_residuals_x / ss_total_x)

    dmy_fit = f(my_first, result.x[0])
    ss_residuals_y = np.sum((dmy[nan_mask] - dmy_fit[nan_mask])**2)
    ss_total_y = np.sum((dmy[nan_mask] - np.nanmean(dmy))**2)
    r_squared_1_y = 1 - (ss_residuals_y / ss_total_y)

    B_xx2_fit = g(m2, result.x[0], result.x[1])
    nan_mask = np.isfinite(B_xx2)
    ss_residuals_x = np.sum((B_xx2[nan_mask] - B_xx2_fit[nan_mask])**2)
    ss_total_x = np.sum((B_xx2[nan_mask] - np.nanmean(B_xx2))**2)
    r_squared_2_x = 1 - (ss_residuals_x / ss_total_x)

    B_yy2_fit = g(m2, result.x[0], result.x[1])
    nan_mask = np.isfinite(B_yy2)
    ss_residuals_y = np.sum((B_yy2[nan_mask] - B_yy2_fit[nan_mask])**2)
    ss_total_y = np.sum((B_yy2[nan_mask] - np.nanmean(B_yy2))**2)
    r_squared_2_y = 1 - (ss_residuals_y / ss_total_y)

    return alpha_fit, beta_fit, (r_squared_1_x + r_squared_1_y + r_squared_2_x + r_squared_2_y)/4


def get_single_fish_dtheta_vs_dt(single_fish_subgroups, subgroup_frames, head_angles, dt_array):
    '''
    Calculate chnage in head angle dtheta for single fish subgroups (Ng=1) for time lags dt_array.
    
    Input:
        single_fish : list of single fish subgroups
        single_fish_frames : list of (start,end) frame indices for each single fish subgroup
        head_angles : (N,T) array of heading angles of all fish across time
        dt_array : array of time lags (in frames) to calculate dtheta for
    Returns:
        dtheta_vs_dt : array with 2 columns: dt, dtheta pooled across all single fish subgroups
    '''
    dtheta_vs_dt = []

    for (fish,times) in zip(single_fish_subgroups, subgroup_frames):
        lifetime = times[1] - times[0]
        head_angles_single_fish = head_angles[fish, times[0]:times[1]].reshape(-1) 
        dt_array_truncated = dt_array[dt_array < lifetime]
        for dt in dt_array_truncated:
            dtheta = head_angles_single_fish[dt:] -  head_angles_single_fish[:-dt]
            # wrap dtheta to be in the range [-pi, pi]
            dtheta[dtheta<-np.pi] = dtheta[dtheta<-np.pi]+2*np.pi
            dtheta[dtheta>np.pi] = dtheta[dtheta>np.pi]-2*np.pi  
            dtheta_vs_dt.append(np.column_stack((np.full(len(dtheta), dt), dtheta)))
    return np.concatenate(dtheta_vs_dt,axis=0)

def get_MSD_dtheta_vs_dt(dtheta_vs_dt):
    '''
    Calculate mean squared change in head angle <dtheta^2> for each time lag dt
    
    Input:
        dtheta_vs_dt : array with 2 columns: dt, dtheta pooled across all single fish subgroups
    Returns:
        MSD_vs_dt : array with 2 columns: dt, <dtheta^2>
        n_data_points : array with 2 columns: dt, number of data points used to calculate <dtheta^2> for each dt
    '''
    dt_array = np.unique(dtheta_vs_dt[:,0])
    MSD_vs_dt = np.zeros((len(dt_array), 2))
    n_data_points = np.zeros((len(dt_array), 2))
    for i, dt in enumerate(dt_array):
        dtheta = dtheta_vs_dt[dtheta_vs_dt[:,0]==dt,1]
        MSD_vs_dt[i] = [dt, np.nanmean(dtheta**2)]
        n_data_points[i] = [dt, len(dtheta)]
    return MSD_vs_dt, n_data_points

def fit_MSD_slope_linear_regime(MSD_vs_dt, dt_linear, fps=121):
    '''
    Fit slope of MSD vs dt in the linear regime (dt > dt_linear)
    
    Input:
        MSD_vs_dt : array with 2 columns: dt (in frames), <dtheta^2>
        dt_linear : time lag (in s) above which to fit linear regime
        fps : frames per second
    Returns:
        slope : fitted slope of MSD vs dt in the linear regime
    '''
    # convert all times to seconds!
    dt = MSD_vs_dt[:,0] / fps
    MSD = MSD_vs_dt[:,1]
    mask = dt > dt_linear
    slope, _ = np.polyfit(dt[mask], MSD[mask], 1)
    return slope

def fit_s_epsilon_jointly_bounded(alpha, slope, s_upper_bound):
    def residuals(params):
        s, epsilon  = params
        # Calculate residuals for each equation 
        res1 = (slope-s*epsilon)
        res2 = alpha - s * (1 - np.exp(-epsilon/2))
        # Return both residuals as a vector
        return np.array([res1, res2])  
    initial_guess = [1, 1]  # Initial guess for [s, epsilon]
    bounds = ([0, 0], [s_upper_bound, np.inf])  # Bounds for parameters
    result = least_squares(residuals, initial_guess, bounds=bounds)
    return result.x