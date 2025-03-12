
# impact_gap.py
# Christopher McAllester, February 2019, UW Madison Laboratory of Genetics
#
# A script for running calculations of the impact gap gerrymandering metric
#
# Current Concerns:
#   1) may be numerically unstable at large voter sizes
#   2) calculation time may not scale well to large electorate sizes,
#   and many distance calculations may be repeated if
#   voter data is only accurate to the polling location
#   3) does not deal with the corrections necessary to account for
#   the effects of the non-Euclidean curvature of the Earth on distances
#   - these are likely mild effects
#   4) a Gaussian kernel is intuitively correct, but (more difficult)
#   direct modeling of average district maps may be worthwhile

import numpy as np
import argparse
import math as m
import os
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fsolve



# Calculates the scaled bivariate gaussian kernel (non-normalized)
def biv_gauss(d,h):
    return(np.exp(-np.power(d/h,2)))

# Estimates the scaling factor necessary for constant district density D
#   at all test locations (giving an idealized average district)
# IS THERE A CLOSED FORM SOLUTION FOR THE SCALING?
def estimate_scaling(distance_matrix,pilot_h,opt_density):
    est_error = lambda h, d : np.sum(biv_gauss(d,h)) - opt_density
    estimate_h = lambda d : fsolve(est_error, pilot_h, args=(d))
    return(np.apply_along_axis(estimate_h, 0, distance_matrix))

# Calculates the euclidean distance between locations
#   based on voter coordinate data.
# CHECK - if location data is repeated (i.e. entered by polling location)
#   then structuring the data to remove duplicate calcs will save time.
# CHECK - optimize against numerical instability
def estimate_density(voter_coords,ideal_district_size,num_voters):
    # Calculate the distances between each voter
    pairwise_dists = pdist(voter_coords, 'euclidean')
    dist_matrix = squareform(pairwise_dists)

    # Guess a universal scaling parameter
    # pilot_h = 1
    pilot_h = np.max(pairwise_dists)

    # Numerically estimate optimal scaling parameters
    h = estimate_scaling(dist_matrix,pilot_h,ideal_district_size)

    # Calculate densities at the optimal scaling
    K_scaled = biv_gauss(dist_matrix,h)

    # Return the normalized densities
    return(K_scaled/ideal_district_size)

# A convenience wrapper function for taking the entropy 
#   of n probability distributions in m columns (base e)
def entropy(prob_array):
    base = m.e
    entropy = -(prob_array * np.log(prob_array)/np.log(base)).sum(axis=0)
    return(entropy)

# For calculating the entropy of an 'average district' at the
#   location of each voter based on the 'balloon estimator' approximation
def individual_entropies(K_scaled,voter_party,num_parties,num_voters):
    party_probs = np.zeros((num_parties,num_voters))
    # There has to be a better way to sum party density
    for (i1,i2),k in np.ndenumerate(K_scaled):
        party_probs[voter_party[i1],i2] += k
    H_i = entropy(party_probs)
    return H_i

# For calculating the entropy of the electorate in each district
def district_entropies(voters,num_parties,num_districts):
    # Calculate the counts of voters per party per district
    counts = np.zeros((num_parties,num_districts))
    for v in np.nditer(voters):
        counts[v['party'],v['district']] += 1
    # Could also do this with np.unique(..., return_counts = true), but
    #   that requires making a 1D array of (p,d) tuples

    # And calc the probabilities/frequencies of a party voter in a district
    freqs = counts/np.sum(counts,axis=0)

    # Calculate the total entropy of each district
    H_d = entropy(freqs)

    return(H_d)

# Calculates the impact of the districting plan on each party
#   from the difference between average and existing ditrict
#   electorate entropies
def party_impacts(H_i,H_d,voters,num_parties):
    I_p = np.zeros(num_parties)
    for i,v in np.ndenumerate(voters):
        voter_impact = H_d[v['district']] - H_i[i]
        I_p[v['party']] += voter_impact
    return(I_p)

# Only defined now for a two-party system.
# The difference between the two parties in the voter impact
#   of the party members voting in the state.
def impact(voters):
    # An alternate structure to putting everything in main(),
    #   this is to allow external calls on the impact gap function
    #   when this file isn't called as a script directly

    # Get relevant numbers from the voter data
    num_voters = voters.shape[0]
    num_parties = np.amax(voters['party']) + 1
    num_districts = np.amax(voters['district']) + 1
    ideal_district_size = num_voters / num_districts

    # Make some basic assertions about the input
    assert (num_voters>10), 'Ideally an impact gap is run on reasonable \
        electorate sizes: {} is not even > 10'.format(num_voters)
    assert (num_parties>1), 'All voters appear to be of one party, \
        for which impact is irrelevant'
    assert (num_districts>1), 'All voters appear to be in one district, \
        and without subdivision gerrymandering is irrelevant'

    # voter_coords = np.concatenate()
    voter_coords = np.vstack((voters['x1'],voters['x2'])).T

    # Calculate the Kernel densities for the individual electorates
    K_scaled = estimate_density(voter_coords,
                                ideal_district_size, num_voters)

    

    # Calculate the entropy of the approximate electorate of each individual
    H_indi = individual_entropies(K_scaled,voters['party'],
                                    num_parties,num_voters)

    # Calculate the entropy of the electorate in each district
    H_dist = district_entropies(voters, num_parties, num_districts)

    # Calculate the impact of the district on each party
    I_p = party_impacts(H_indi,H_dist,voters,num_parties)
    print(I_p)

    # Average this impact per voter
    # (This partially repeats count data from the district entropy)
    unique, counts = np.unique(voters['party'], return_counts=True)
    print(counts)
    avg_I_p = I_p/counts
    print(avg_I_p)

    # An approximate metric for two party systems - the difference in impact
    impact_gap = avg_I_p[1]-avg_I_p[0]
    print("Impact Gap between Parties 0, 1: " + str(impact_gap))
    return

# A wrapper for parsing an input voter csv file
def parse_voter_file(file_name,len_header):
    voters = np.genfromtxt(file_name, dtype=None, 
        names=['x1','x2','party','district'], delimiter=',',
        skip_header=len_header)
    return(voters)


# Handling direct calls to impact_gap.py, 
#   and processing of variable input files
def main(args):
    # Separate the arguments
    voter_file_name = args.v
    len_header = args.h
    # state_file_name = args.s
    output_directory = args.out
    
    assert (voter_file_name!=''),'voter list file path (--v) {} \
        not given'.format(voter_file_name)
    # assert (state_file_name!=''),'state data file path (--s) {} \
    #     not given'.format(state_file_name)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process the csv voter file to a numpy ndarray
    voters = parse_voter_file(voter_file_name,len_header)

    # Determine the impact gap statistics
    impact(voters)
    return


# Runs if impact_gap.py is the main script
if __name__ == "__main__":
	# Collect input arguments
	# CHECK argparse documentation for other input ideas?
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--v',
                help='the input csv list of voter data (float coords and \
                int party/district): x coord, y coord, party, and district',
                type=str,
                default='')
	parser.add_argument('--h',
                help='the number of header lines on the voter csv file',
                type=int,
                default=0)
	parser.add_argument('--s',
                help='the input csv list of state data (as integers): \
                number of voters, parties, and districts',
                type=str,
                default='')
	parser.add_argument('--out',
                help='output directory for the impact analysis',
                type=str,
                default='default_output/')
	args = parser.parse_args()
	main(args)