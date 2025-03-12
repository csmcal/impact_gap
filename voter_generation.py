
# A script for generating simulated voter .csv files
#   for use in calculating the impact_gap and efficiency_gap
#

import numpy as np
import argparse
from scipy.stats import truncnorm
import os


# Returns position and party data for voters in a city
def sample_city(l,w,x_1,x_2,h,N,p):
    # Sampling by two truncated normals is probably wrong, but
    #   for now let's just say that's our city distribution
    #   (Even if it's right as a symmetric you can't make ellipsoids)
    coord_1 = truncnorm((0-x_1)/h,(l-x_1)/h,loc=x_1,scale=h).rvs(N)
    coord_2 = truncnorm((0-x_2)/h,(w-x_2)/h,loc=x_2,scale=h).rvs(N)
    party = np.random.binomial(1,p,N)
    return(coord_1,coord_2,party)

# Returns basic voter data for the underlying uniform distribution
def sample_uniform(l,w,N,p):
    coord_1 = np.random.uniform(0,l,N)
    coord_2 = np.random.uniform(0,w,N)
    party = np.random.binomial(1,p,N)
    return(coord_1,coord_2,party)

# Returns districts for positions (c_1, c_2)
#   in a grid of m x n districts in an l x w state
def check_district(c_1,c_2,m,n,l,w):
    district = np.floor_divide(c_1,l/m) + m*np.floor_divide(c_2,w/n)
    return(district)



# A basic simulation - a rectangular state with m x n districts
#   in a grid, l x w state dimensions, two political parties,
#   and population divided equally between a uniform distribution
#   and N_cities cities modeled as symetric Gaussians with 
#   bandwidth h_cities, randomly distributed
#   Districts are labeled from (0,0) -> (0,1) -> (1,0): right then up
def sim_square_state(m,n,l,w,N,n_cities,p_unif,p_cities,h_cities):
    # Calculate the city population size
    N_city = N//(n_cities+1)
    # Calculate the underlying remaining population
    N_unif = N_city + N%(n_cities+1)
    # Populate the state background
    (coord_1,coord_2,party) = sample_uniform(l,w,N_unif,p_unif)
    # Sample city locations
    x_1 = np.random.uniform(0,l,n_cities)
    x_2 = np.random.uniform(0,w,n_cities)
    # Populate the cities
    for i in np.arange(n_cities):
        N_unif -= N_city
        (c_1,c_2,par) = sample_city(l,w,x_1[i],x_2[i],h_cities,N_city,p_cities)
        coord_1 = np.concatenate((coord_1,c_1))
        coord_2 = np.concatenate((coord_2,c_2))
        party = np.concatenate((party,par))
    # Assign the correct district
    district = check_district(coord_1,coord_2,m,n,l,w)
    # Set up the voter array
    voters = np.empty(N,
        dtype=[('x1','f4'),('x2','f4'),('party','i4'),('district','i4')])
    voters['x1'] = coord_1
    voters['x2'] = coord_2
    voters['party'] = party
    voters['district'] = district

    return(voters)

# Writes voter data 
def write_voters(file_name,voters):
    print('writing \''+file_name+'\'')
    # print(voters)
    np.savetxt(file_name,voters,
        fmt=['%.18e','%.18e','%.0f','%.0f'],delimiter=',')
    return

# Parses the state csv file
def read_state(file_name,len_header):
    state = np.genfromtxt(file_name, dtype=None, delimiter=',',
        skip_header=len_header)
    # I still can't figure out why just '= state' doesn't work
    (m,n,l,w,N,n_cities,p_unif,p_cities,h_cities) = np.array([state])[0]
    return(m,n,l,w,N,n_cities,p_unif,p_cities,h_cities)


# Handling direct calls to impact_gap.py, 
#   and processing of variable input files
def main(args):
    # Separate the arguments
    state_f = args.s
    h_l = args.h
    num_sims = args.N
    out_dir = args.out
    out_tag = args.tag
    
    assert (state_f!=''),'state data file path (--s) {} \
        not given'.format(state_f)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Interpret the generation parameters    
    (m,n,l,w,N,n_cities,p_unif,p_cities,h_cities) = read_state(state_f,h_l)

    # Generate a voter file for each simulation
    if num_sims > 1:
        for i in np.arange(num_sims):
            output_file = out_dir + out_tag + "_" + str(i) + ".csv"
            # Generate the voters
            voters = sim_square_state(m,n,l,w,N,n_cities,p_unif,p_cities,h_cities)
            # Write the voters
            write_voters(output_file,voters)
    else:
        output_file = out_dir + out_tag + ".csv"
        # Generate the voters
        voters = sim_square_state(m,n,l,w,N,n_cities,p_unif,p_cities,h_cities)
        # Write the voters
        write_voters(output_file,voters)

    return


# Runs if voter_generation.py is the main script
if __name__ == "__main__":
	# Collect input arguments
	# CHECK argparse documentation for other input ideas?
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--s',
                help='the input csv list of state data (as csv): \
                ,n,l,w,N,number_cities,p_unif,p_cities,scale_cities',
                type=str,
                default='')
	parser.add_argument('--h',
                help='the number of header lines on the state csv file',
                type=int,
                default=1)
	parser.add_argument('--N',
                help='the number of simulated states to generate',
                type=int,
                default=1)
	parser.add_argument('--out',
                help='output directory for the voter data csv file',
                type=str,
                default='test_data/')
	parser.add_argument('--tag',
                help='output directory for the voter data csv file',
                type=str,
                default='sim_voters')
	args = parser.parse_args()
	main(args)