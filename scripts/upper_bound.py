import scipy.integrate as integrate
from scipy.stats import norm
import numpy as np

from hausdorff_subsampling import hausdorff_distance
import matplotlib.pyplot as plt

def draw_distances(x,y):
    normal_draws = np.random.normal(0,1, x*y) 
    lognormal_draws = np.exp(normal_draws).reshape([x,y])
    return lognormal_draws

def manual_hd(X, pr=False):
    '''
    Compute HD dist manually of Distance matrix (subset n-m x m )
    '''
    mins = X.min(axis=1)
    if pr:
        print(mins)  
    maxmin = mins.max()
    return maxmin

  
n =100 #n samples

m_s = np.arange(1,n)

n_runs = 5

dist_mat = np.zeros([n_runs, n-1])
bound_mat = np.zeros([n_runs, n-1])

for i in np.arange(n_runs):
    dists = []
    bounds = []
    for m in m_s:

        #get lognormal distribution distances of (n-m) x m which are used for bound / HD dist
        D = draw_distances(n-m, m)

        #print(draw_distances(5,3))

        Bound, error = integrate.quad(lambda x: 1 - norm.cdf(np.log(x))**(m*(n-m)),0,np.inf)
        bounds.append(Bound)
        #print(f'Upper Bound: {Bound}')
        if m < 5:
            hd_true = manual_hd(D, pr=True)
        else:
            hd_true = manual_hd(D)
        #print(f'Actual HD: {hd_true}')
        dists.append(hd_true)
        
        if m < 5:
            print(f'true vs bound vs error: {hd_true}, {Bound}, {error}')
    dist_mat[i,:] = dists
    bound_mat[i,:] = bounds

plt.figure()

dist_mean = dist_mat.mean(axis=0)
dist_std = dist_mat.std(axis=0)
lower = dist_mean - dist_std
upper = dist_mean + dist_std
plt.plot(m_s, dist_mean, color='blue', label='True HD Distance', alpha=1)
plt.fill_between(m_s, lower, upper, color='blue', alpha=.25)

bound_mean = bound_mat.mean(axis=0)
bound_std = bound_mat.std(axis=0)
lower = bound_mean - bound_std
upper = bound_mean + bound_std
plt.plot(m_s, bound_mean, color='green', label='Expectation Upper Bound', alpha=1)
plt.fill_between(m_s, lower, upper, color='green', alpha=.25)


#plt.plot(m_s, dists, label='True HD Distance')
#plt.plot(m_s, bounds, label='Upper Bound')
plt.legend()
plt.ylabel('Distance')
plt.xlabel(f'Subsample Size, with total n={n}')
plt.title('Upper Bound Visualization')

plt.show()





