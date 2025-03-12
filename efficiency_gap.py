
# efficiency_gap.py
#
# A script for running calculations of the efficiency gap gerrymandering metric
#

import numpy as np
import argparse
import os


# Calculates the 'wasted' votes in the state
#   (only works for >2 parties if a party has a majority)
def wasted_votes(voters,num_parties,num_districts):
    # Calculate the counts of voters per party per district
    counts = np.zeros((num_parties,num_districts))
    dist_sizes = np.zeros(num_districts)
    for v in np.nditer(voters):
        counts[v['party'],v['district']] += 1
        dist_sizes[v['district']] += 1
    # Calculate the 'wasted' votes 
    #   (only works for >2 parties if a party has a majority)
    wasted_votes = np.remainder(counts,np.floor_divide(dist_sizes,2))
    total_wasted = np.sum(wasted_votes,axis=0)
    return(total_wasted)

# Defined for two parties, so only compares the first two entries,
#   calculates the difference in 'wasted' votes / total number of voters
def efficiency_gap(W,N):
    E_g = (W[0]-W[1])/N
    return(E_g)
    

# Only defined for a two-party system.
# The difference between the two parties in the quantity
#   of 'wasted' votes in the state.
def efficiency(voters):
    # Get relevant numbers from the voter data
    num_voters = voters.shape[0]
    num_parties = np.amax(voters['party']) + 1
    num_districts = np.amax(voters['district']) + 1

    # Make some basic assertions about the input
    assert (num_voters>10), 'Ideally an efficiency gap is run on reasonable \
        electorate sizes: {} is not even > 10'.format(num_voters)
    assert (num_parties>1), 'All voters appear to be of one party, \
        for which impact is irrelevant'
    assert (num_districts>1), 'All voters appear to be in one district, \
        and without subdivision gerrymandering is irrelevant'
    

    # Calculate the 'wasted' votes 
    W = wasted_votes(voters,num_parties,num_districts)
    # Calculate the statistic
    eff_gap = efficiency_gap(W,num_voters)
    print("Efficiency Gap between Parties 0, 1: " + str(eff_gap))
    return




# A wrapper for parsing an input voter csv file
def parse_voter_file(file_name,len_header):
    voters = np.genfromtxt(file_name, dtype=None, 
        names=['x1','x2','party','district'], delimiter=',',
        skip_header=len_header)
    return(voters)

# Handling direct calls to efficiency_gap.py, 
#   and processing of variable input files
def main(args):
    # Separate the arguments
    voter_file_name = args.v
    len_header = args.h
    output_directory = args.out
    
    assert (voter_file_name!=''),'voter list file path (--v) {} \
        not given'.format(voter_file_name)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process the csv voter file to a numpy ndarray
    voters = parse_voter_file(voter_file_name,len_header)

    # Determine the efficiency gap statistic
    efficiency(voters)
    return


# Runs if efficiency_gap.py is the main script
if __name__ == "__main__":
	# Collect input arguments
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
	parser.add_argument('--out',
                help='output directory for the impact analysis',
                type=str,
                default='default_output/')
	args = parser.parse_args()
	main(args)