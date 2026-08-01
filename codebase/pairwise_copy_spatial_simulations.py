import numpy as np


class Simulate_Fish_School:
    def __init__(self, N, v, R, s, eps, c, Tmax, dt, L, seed=0):
        '''
        Inputs:
            N : # of fish
            v: (fixed) velocity values for each fish (N,) 
            R: radius of arena
            s: angular diffusion rate (N,)
            eps: variance of angular diffusion events (N,)
            c: copying rate (N,)
            Tmax: simulation time
            dt: time interval
            L : interaction distance (same for all fish)
            seed: random seed for initialization and stochastic events
        '''
        self.N = N
        self.v = v
        self.R = R
        self.s = s
        self.eps = eps
        self.c = c
        self.Tmax = Tmax
        self.dt = dt
        self.L = L
        self.rng = np.random.default_rng(seed)
        self.t = 0
        
    def get_initial_positions(self):
        '''
        Return:
            x0,y0,theta0 : arrays of shape (N,). random initial positions and headings
            Here, (0,0) is the center of the circular arena. We initialize positions uniformly in the arena, and headings uniformly in [0,2pi]
        '''
        r1 = self.rng.random(self.N)
        r2 = 2*np.pi*self.rng.random(self.N)
        x0 = self.R*r1*np.cos(r2)
        y0 = self.R*r1*np.sin(r2)
        theta0 = 2*np.pi*self.rng.random(self.N)
        return x0,y0,theta0
    
    def update_positions(self):
        t = self.t
        x_temp = self.x[:,t] + self.v*np.cos(self.theta[:,t])*self.dt
        y_temp = self.y[:,t] + self.v*np.sin(self.theta[:,t])*self.dt
        self.x[:,t+1], self.y[:,t+1], self.theta[:,t+1] = self.reflective_bound(x_temp, y_temp, self.theta[:,t])
        return
    
    def reflective_bound(self,x,y,theta):
        '''
        Input:
            x,y, theta : position and heading angle of each fish at a time-point. Shape (N,)
        Return:
            x_new, y_new, theta_new : arrays after implementing reflective boundary conditions
        '''
        x_new = x
        y_new = y
        theta_new = theta

        distance_to_center = np.sqrt(x**2 + y**2) 

        for i,d in enumerate(distance_to_center):
            if d>self.R:
                # Calculate the unit vector pointing from the center of the arena to the fish
                ux = x[i] / d
                uy = y[i] / d

                # Calculate the projection of the heading direction onto the unit vector
                h_parallel = np.cos(theta[i]) * ux + np.sin(theta[i]) * uy

                # Calculate the reflection of the heading vector
                hx_new = np.cos(theta[i]) - 2 * h_parallel * ux
                hy_new = np.sin(theta[i]) - 2 * h_parallel * uy

                #Calculate new heading angle
                theta_new[i] = np.arctan2(hy_new,hx_new)

                # Set the new position to the intersection point with the arena boundary
                x_new[i] = self.R * ux
                y_new[i] = self.R * uy

        return x_new,y_new,theta_new
    
    def get_copying_pairs(self):
        '''
        Return:
            pairs : list of all allowed pairs for copying interaction
            rates : array of copying rate associated with each pair
        '''
        # get current positions
        x = self.x[:,self.t]
        y = self.y[:,self.t]

        pairs = []
        rates = []
        
        for i in range(self.N):
            distances = np.sqrt((x-x[i])**2 + (y-y[i])**2)
            for j in range(i+1,self.N):
                if distances[j]<self.L:
                    pairs.append([i,j])
                    rates.append(self.c[i])
                    pairs.append([j,i])
                    rates.append(self.c[j])
        
        rates = np.array(rates)
        return pairs, rates
    
    def run_simulation(self):
        Tind_max = np.ceil(self.Tmax/self.dt).astype(int)
        self.x = np.empty((self.N, Tind_max+1 )) # N x time_points
        self.y = np.empty((self.N, Tind_max+1 ))
        self.theta = np.empty((self.N, Tind_max+1 ))
        self.x[:,0], self.y[:,0], self.theta[:,0] = self.get_initial_positions()
        diffusion_probabilities = self.s / np.sum(self.s)
        
        while self.t < Tind_max:
            pairs,copy_rates = self.get_copying_pairs()
            copy_probabilities = copy_rates / np.sum(copy_rates)
            
            # sample number of copying and diffusion events
            n_copy = self.rng.poisson(lam = np.sum(copy_rates)*self.dt)
            n_diffuse = self.rng.poisson(lam = np.sum(self.s)*self.dt)
            
            # implement random copying and diffusion events
            if n_copy:
                pair_idxs = self.rng.choice(len(pairs),size=n_copy,replace=True, p = copy_probabilities)
                for idx in pair_idxs:
                    pair = pairs[idx]
                    # fish pair[0] copies fish pair[1]'s heading
                    self.theta[pair[0], self.t] = self.theta[pair[1], self.t]
                    
            if n_diffuse:
                fish_idxs = self.rng.choice(self.N,size=n_diffuse,replace=True, p = diffusion_probabilities)
                for idx in fish_idxs:
                    self.theta[idx, self.t] += self.rng.normal(loc=0,scale=np.sqrt(self.eps[idx]))
            
            # generate new positions
            self.update_positions()

            # update time index
            self.t+=1

        return self.x,self.y,self.theta
    


class Simulate_Fish_School_Excluded_Volume(Simulate_Fish_School):
    def __init__(self, N, v, R, s, eps, c, Tmax, dt, L, R_ex, omega, seed=0):
        '''
        Inputs:
            N : # of fish
            v: (fixed) velocity values for each fish (N,) 
            R: radius of arena
            s: angular diffusion rate (N,)
            eps: variance of angular diffusion events (N,)
            c: copying rate (N,)
            Tmax: simulation time
            dt: time interval
            L : interaction distance (1,)
            R_ex: excluded volume radius (1,)
            omega: strength of excluded volume repulsion (1,)
            seed: random seed for initialization and stochastic events
        '''
        super().__init__(N, v, R, s, eps, c, Tmax, dt, L, seed)
        self.R_ex = R_ex
        self.omega = omega
        
    def get_initial_positions(self):
        '''
        Return:
            x0,y0,theta0 : arrays of shape (N,). random initial positions and headings
            Here, (0,0) is the center of the circular arena. We initialize positions uniformly in the arena, and headings uniformly in [0,2pi]
        '''
        # set initial positions by rejection sampling to ensure no fish start within R_ex of each other
        x0 = np.empty(self.N)
        y0 = np.empty(self.N)
        for i in range(self.N):
            while True:
                r1 = self.rng.random()
                r2 = 2*np.pi*self.rng.random()
                x_temp = self.R*r1*np.cos(r2)
                y_temp = self.R*r1*np.sin(r2)
                if i==0:
                    x0[i] = x_temp
                    y0[i] = y_temp
                    break
                else:
                    distances = np.sqrt((x0[:i]-x_temp)**2 + (y0[:i]-y_temp)**2)
                    if np.all(distances>self.R_ex):
                        x0[i] = x_temp
                        y0[i] = y_temp
                        break
        theta0 = 2*np.pi*self.rng.random(self.N)
        return x0,y0,theta0
    
    def implement_excluded_volume_interactions(self):
        ''' 
        check distances between all pairs of fish at current time. 
        if any distance < R_ex, revert that pair to previous positions and turn heading angles away from each other
        '''
        revert = np.zeros(self.N,dtype=np.bool_)
        total_turn_magnitude = np.zeros(self.N)
        
        for i in range(self.N):
            for j in range(i+1,self.N):
                d = np.sqrt((self.x[i,self.t] - self.x[j,self.t])**2 + (self.y[i,self.t] - self.y[j,self.t])**2)
                if d < self.R_ex:
                    revert[i] = 1
                    revert[j] = 1

                    # calculate turning magnitude
                    # contributions from different pairwise interactions are additive!

                    # fish i
                    # calculate angle pointing from fish j to i
                    phi = np.arctan2(self.y[i,self.t] - self.y[j,self.t],self.x[i,self.t] - self.x[j,self.t]) 
                    # direction & magnitude of turn
                    turn = phi - self.theta[i,self.t]
                    if turn < -np.pi:
                        turn += 2*np.pi
                    elif turn > np.pi:
                        turn -= 2*np.pi
                    total_turn_magnitude[i] += self.omega*turn 

                    # fish j
                    # calculate angle pointing from fish i to j
                    phi = np.arctan2(self.y[j,self.t] - self.y[i,self.t],self.x[j,self.t] - self.x[i,self.t])
                    # direction & magnitude of turn
                    turn = phi - self.theta[j,self.t]
                    if turn < -np.pi:
                        turn += 2*np.pi
                    elif turn > np.pi:
                        turn -= 2*np.pi
                    total_turn_magnitude[j] += self.omega*turn
                    
                    
        self.x[revert, self.t] = self.x[revert, self.t-1]
        self.y[revert, self.t] = self.y[revert, self.t-1]
        self.theta[:, self.t] += total_turn_magnitude

        return

    
    def run_simulation(self):
        Tind_max = np.ceil(self.Tmax/self.dt).astype(int)
        self.x = np.empty((self.N, Tind_max+1 )) # N x time_points
        self.y = np.empty((self.N, Tind_max+1 ))
        self.theta = np.empty((self.N, Tind_max+1 ))
        self.x[:,0], self.y[:,0], self.theta[:,0] = self.get_initial_positions()
        diffusion_probabilities = self.s / np.sum(self.s)
        
        while self.t < Tind_max:
            pairs,copy_rates = self.get_copying_pairs()
            copy_probabilities = copy_rates / np.sum(copy_rates)
            
            # sample number of copying and diffusion events
            n_copy = self.rng.poisson(lam = np.sum(copy_rates)*self.dt)
            n_diffuse = self.rng.poisson(lam = np.sum(self.s)*self.dt)
            
            # implement random copying and diffusion events
            if n_copy:
                pair_idxs = self.rng.choice(len(pairs),size=n_copy,replace=True, p = copy_probabilities)
                for idx in pair_idxs:
                    pair = pairs[idx]
                    # fish pair[0] copies fish pair[1]'s heading
                    self.theta[pair[0], self.t] = self.theta[pair[1], self.t]
                    
            if n_diffuse:
                fish_idxs = self.rng.choice(self.N,size=n_diffuse,replace=True, p = diffusion_probabilities)
                for idx in fish_idxs:
                    self.theta[idx, self.t] += self.rng.normal(loc=0,scale=np.sqrt(self.eps[idx]))
            
        
            self.update_positions()
            self.t+=1

            self.implement_excluded_volume_interactions()

        return self.x,self.y,self.theta