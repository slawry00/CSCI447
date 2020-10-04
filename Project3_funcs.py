import numpy as np
import networkx as nx
import random
import math


# two-dimensional numpy array + int + float (+ optional int)
# -> tuple (2D np array, 1D np array of sets)
# takes a data matrix, number of clusters, and convergence parameter and performs the k-means
# algorithm on that data. Max iterations can be specified optionally
# Returns (means array, array of cluster numbers)
def k_means(D, k, eps, max_iterations=1000):
    t = 0
    tot_sum = 0.0
    mean_change = eps + 1 # so it will enter the while loop the first time
    means = assign_rand_reps(D, k)
    while (mean_change >= eps and t < max_iterations):
        tot_sum = 0.0
        mean_change = 0
        t += 1
        clust_list = [] # going to be a list of cluster sets (containing the entry index)
        for i in range(k):
            clust_list.append(set())
        for i, entry in enumerate(D):
            closest = find_closest(means, entry)
            clust_list[closest].add(i)
        for clust_num, clust in enumerate(clust_list):
            if (len(clust) <= 0):
                new_rep = assign_rand_reps(D, k) # no entires in cluster, reassign randomly
                means[clust_num] = new_rep
                mean_change = eps + 1 # needs to retry until all clusters have an entry
                break
            sums = [0 for i in range(D.shape[1])] # initialze dimension sums to 0
            for ind in clust:
                for dim_num, dim_val in enumerate(D[ind]):
                    sums[dim_num] += dim_val
            new_rep = [val/len(clust) for val in sums] # divide by # of elements in cluster
            cumul_sum = 0
            for dim_num, dim_val in enumerate(new_rep): # calculate the difference in means
                diff = dim_val - means[clust_num][dim_num]
                cumul_sum += math.pow(diff,2)
            tot_sum += cumul_sum
            mean_change += math.sqrt(cumul_sum)
            means[clust_num] = new_rep
    return (np.array(means), np.array(clust_list))

# data matrix + number of clusters -> 2D list
# does the random assignment of mean representatives
def assign_rand_reps(D, k):
    means = []
    for i in range(k): # for each cluster
        represent = []
        for j in range(D.shape[1]): # for each dimension
            represent.append(random.uniform(min(D.T[j]), max(D.T[j]))) # randomly initialize means
        means.append(represent)
    return means

# two-dimensional list + one_dimensional list -> int
# takes a 2D list where each member is a representative mean, which is represented by a list of
# its coordinates in each dimension
# and an entry, which is some point in the dataset, represented by a list of coordinates
# and calculates the index of the closest representative in reps
def find_closest(reps, entry):
    min_dist = -1
    for rep_ind, rep in enumerate(reps):
        cumul_sum = 0
        for dim_num, dim_val in enumerate(rep):
            diff = dim_val - entry[dim_num]
            cumul_sum += math.pow(diff,2)
        dist = math.sqrt(cumul_sum)
        if (dist < min_dist or min_dist == -1):
            min_dist = dist
            min_ind = rep_ind
    return min_ind

# 2D np array + 2D np array + 1D np array of sets -> 1D np array
# returns the sum of squared distances to the each representative as an ordered list
def calc_square_to_reps(D, reps, clust_list):
    dists = []
    for clust_num, clust in enumerate(clust_list):
        tot_dist = 0.0
        for entry in clust:
            pt = D[entry]
            cumul_sum = 0.0
            for dim_num, dim_val in enumerate(pt):
                diff = dim_val - reps[clust_num][dim_num]
                cumul_sum += math.pow(diff,2)
            dist = math.sqrt(cumul_sum)
            tot_dist += math.pow(dist,2)
        dists.append(tot_dist)
    return np.array(dists)

# two-dimensional numpy array + int + float ->  2D numpy array
# takes a data matrix, minpts to be a "core point", and radius of neighborhood: epsilon and
# and performs the DBSCAN algorithm on that data.
# Returns the clusters as a 2D array where:
# The first column classifies each point as either noise(0), border(1), or core(2).
# The second column is the cluster number that data entry is assigned to (-1 = no cluster)
def dbscan(D, minpts, eps):
    core = set()
    ret_list = []
    unassigned = -1
    for entry_num, entry in enumerate(D):
        count = eps_neighborhood(entry, D, eps)[0]
        if (count >= minpts):
            core.add(entry_num)
            ret_list.append([2, unassigned]) # 2 means core
        else:
            ret_list.append([-1, unassigned]) # -1 means: not core (border or noise)
    k = -1 # cluster number (will start at 0 when first incremented)
    for core_pt in core:
        if (ret_list[core_pt][1] == unassigned):
            k += 1
            ret_list[core_pt][1] = k # assign it to cluster k
            ret_list = density_connected(D[core_pt], core, k, D, eps, ret_list)
    for entry in ret_list:
        if (entry[1] == unassigned): # still not set to any cluster
            entry[0] = 0 # set to noise
        elif (entry[0] != 2): # if its not noise, and not core
            entry[0] = 1 # set to border
    return np.array(ret_list)

# 1D numpy array + 2D numpy array + float -> int
# computes the epsilon neighborhood of pt by finding the amount of points within eps distance
# of pt in D and returning that count
def eps_neighborhood(pt, D, eps):
    count = 0
    EN_list = []
    for entry_num, entry in enumerate(D):
        cumul_sum = 0
        for dim_num, dim_val in enumerate(entry):
            diff = pt[dim_num] - dim_val
            cumul_sum += math.pow(diff,2)
        dist = math.sqrt(cumul_sum)
        if (dist <= eps):
            EN_list.append(entry_num)
            count += 1
    return (count, EN_list)

# 1D np array + set + int + 2D np array + float + 2D list -> 2D list
# takes a point, a core set, a cluster number, the original data matrix, an epsilon radius,
# and the currently assigned [point types, assigned cluster matrix] list and returns
# a newly formed version of the list after exploring its neighbors recursively
def density_connected(pt, core, k, D, eps, ret_list):
    EN = eps_neighborhood(pt, D, eps)[1]
    for neigh in EN: # neigh is an index of an entry in D
        if ret_list[neigh][1] != k: # if not already visited for this cluster
            ret_list[neigh][1] = k # assign to cluster k
            if (neigh in core):
                ret_list = density_connected(D[neigh], core, k, D, eps, ret_list)
    return ret_list



