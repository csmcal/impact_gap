#!/usr/bin/env python3
"""
Impact Gap Calculator for Gerrymandering Analysis
Author: Christopher McAllester
Initial Version: 2019
Updated: 2024

This script calculates the impact gap gerrymandering metric by analyzing voter distributions
and district assignments. It uses a Gaussian kernel density estimation approach to model
idealized districts and compare them with actual district assignments.

Key Features:
- Calculates pairwise voter distances and density estimations
- Computes entropy-based metrics for district analysis
- Supports multi-party analysis (optimized for two-party systems)
- Supports analysis of any categorical demographic variable (party, race, education, income, etc.)
- Handles HDF5 input for voter data with demographic information

Current Concerns:
1) may be numerically unstable at large voter sizes
2) calculation time may not scale well to large electorate sizes,
and many distance calculations may be repeated if
voter data is only accurate to the polling location
3) does not deal with the corrections necessary to account for
the effects of the non-Euclidean curvature of the Earth on distances
- these are likely mild effects
4) a Gaussian kernel is intuitively correct, but (more difficult)
direct modeling of average district maps may be worthwhile

Usage:
    python impact_gap.py --input voter_data.h5 [--demographic party_id] [--out output_dir]
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import numpy.typing as npt
import argparse
import math
import os
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fsolve
import logging
from voter_data_schema import VoterData, VoterMetadata, Race, Education

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define available demographic variables and their metadata
DEMOGRAPHIC_VARIABLES = {
    'party_id': {
        'description': 'Political party affiliation',
        'name_getter': lambda vd: vd.metadata.party_names,
        'num_categories': lambda vd: vd.metadata.num_parties
    },
    'race': {
        'description': 'Racial category',
        'name_getter': lambda _: {r.value: r.name for r in Race},
        'num_categories': lambda _: len(Race)
    },
    'education': {
        'description': 'Education level',
        'name_getter': lambda _: {e.value: e.name for e in Education},
        'num_categories': lambda _: len(Education)
    },
    'income_decile': {
        'description': 'Income decile (1-10)',
        'name_getter': lambda _: {i: f"Income Decile {i+1}" for i in range(10)},
        'num_categories': lambda _: 10
    },
    'household_size': {
        'description': 'Number of people in household',
        # Assuming max household size of 8, with 8+ grouped together
        'name_getter': lambda _: {
            i: str(i+1) if i < 7 else "8+" for i in range(8)
        },
        'num_categories': lambda _: 8  # 1-7 and 8+
    }
}

def biv_gauss(d: npt.NDArray, h: float) -> npt.NDArray:
    """
    Calculate the scaled bivariate Gaussian kernel (non-normalized).
    
    Args:
        d: Distance matrix
        h: Bandwidth parameter
    
    Returns:
        Array of kernel values
    """
    return np.exp(-np.power(d/h, 2))

def estimate_scaling(
    distance_matrix: npt.NDArray,
    pilot_h: float,
    opt_density: float
) -> npt.NDArray:
    """
    Estimate the scaling factor needed at each voter location
    to achieve a constant district density.
    
    Args:
        distance_matrix: Matrix of pairwise distances
        pilot_h: Initial bandwidth guess
        opt_density: Target density
    
    Returns:
        Array of optimal bandwidth parameters
    """
    def est_error(h: float, d: npt.NDArray) -> float:
        return np.sum(biv_gauss(d, h)) - opt_density
    
    estimate_h = lambda d: fsolve(est_error, pilot_h, args=(d))[0]
    return np.apply_along_axis(estimate_h, 0, distance_matrix)

def estimate_density(voter_data: VoterData) -> npt.NDArray:
    """
    Calculate density estimates using Gaussian kernel.
    
    Args:
        voter_data: VoterData instance containing voter information
    
    Returns:
        Array of normalized density estimates
    """
    try:
        # Extract coordinates into a 2D array for distance calculation
        coords = np.column_stack((voter_data.data['longitude'], voter_data.data['latitude']))
        
        pairwise_dists = pdist(coords, metric='euclidean')
        dist_matrix = squareform(pairwise_dists)
        
        # Use maximum distance as initial bandwidth
        pilot_h = np.max(pairwise_dists)
        
        h = estimate_scaling(dist_matrix, pilot_h, voter_data.metadata.num_voters / voter_data.metadata.num_districts)
        K_scaled = biv_gauss(dist_matrix, h)
        
        return K_scaled / (voter_data.metadata.num_voters / voter_data.metadata.num_districts)
    
    except Exception as e:
        logging.error(f"Error in density estimation: {str(e)}")
        raise

def entropy(prob_array: npt.NDArray, base: float = math.e) -> npt.NDArray:
    """
    Calculate entropy of probability distributions.
    
    Args:
        prob_array: Array of probability distributions
        base: Logarithm base for entropy calculation
    
    Returns:
        Array of entropy values
    """
    # Add small epsilon to avoid log(0)
    eps = np.finfo(float).eps
    prob_array = np.clip(prob_array, eps, 1.0)
    return -(prob_array * np.log(prob_array)/np.log(base)).sum(axis=0)

def individual_entropies(
    K_scaled: npt.NDArray,
    voter_data: VoterData,
    demographic_var: str
) -> npt.NDArray:
    """
    Calculate entropy of 'average district' at each voter location for a given demographic variable.
    
    Args:
        K_scaled: Scaled kernel density matrix
        voter_data: VoterData instance containing voter information
        demographic_var: Name of demographic variable to analyze
    
    Returns:
        Array of entropy values for each voter location
    """
    num_categories = DEMOGRAPHIC_VARIABLES[demographic_var]['num_categories'](voter_data)
    category_probs = np.zeros((num_categories, voter_data.metadata.num_voters))
    
    # Vectorized operation for category density summation
    for category in range(num_categories):
        mask = voter_data.data[demographic_var] == category
        category_probs[category] = np.sum(K_scaled[mask], axis=0)
    
    return entropy(category_probs)

def district_entropies(
    voter_data: VoterData,
    demographic_var: str
) -> npt.NDArray:
    """
    Calculate entropy of the electorate in each district for a given demographic variable.
    
    Args:
        voter_data: VoterData instance containing voter information
        demographic_var: Name of demographic variable to analyze
    
    Returns:
        Array of entropy values for each district
    """
    # Calculate demographic category counts per district using numpy operations
    num_categories = DEMOGRAPHIC_VARIABLES[demographic_var]['num_categories'](voter_data)
    counts = np.zeros((num_categories, voter_data.metadata.num_districts))
    
    for category in range(num_categories):
        for district in range(voter_data.metadata.num_districts):
            counts[category, district] = np.sum(
                (voter_data.data[demographic_var] == category) & 
                (voter_data.data['district_id'] == district)
            )
    
    # Calculate frequencies and entropy
    freqs = counts / np.sum(counts, axis=0)
    return entropy(freqs)

def category_impacts(
    H_i: npt.NDArray,
    H_d: npt.NDArray,
    voter_data: VoterData,
    demographic_var: str
) -> npt.NDArray:
    """
    Calculate impact of districting plan on each category of a demographic variable.
    
    Args:
        H_i: Individual entropy values
        H_d: District entropy values
        voter_data: VoterData instance
        demographic_var: Name of demographic variable to analyze
    
    Returns:
        Array of impact values for each category
    """
    num_categories = DEMOGRAPHIC_VARIABLES[demographic_var]['num_categories'](voter_data)
    I_c = np.zeros(num_categories)
    
    for category in range(num_categories):
        category_mask = voter_data.data[demographic_var] == category
        I_c[category] = np.sum(
            H_d[voter_data.data['district_id'][category_mask]] - H_i[category_mask]
        )
    return I_c

def calculate_impact_gap(
    voter_data: VoterData,
    demographic_var: str = 'party_id'
) -> Dict[str, Any]:
    """
    Calculate the impact gap metric for a given voter distribution and demographic variable.
    
    Args:
        voter_data: VoterData instance containing voter information
        demographic_var: Name of demographic variable to analyze
    
    Returns:
        Dictionary containing impact gap results and intermediate calculations,
        including all pairwise impact gaps between categories
    """
    try:
        if demographic_var not in DEMOGRAPHIC_VARIABLES:
            raise ValueError(f"Unsupported demographic variable: {demographic_var}")
        
        # Calculate kernel densities for each voter location
        K_scaled = estimate_density(voter_data)
        
        # Calculate the entropy of the 'idealized electorate' or 
        # average district at each voter location
        H_indi = individual_entropies(K_scaled, voter_data, demographic_var)

        # Calculate the entropy of the electorate in each established district
        H_dist = district_entropies(voter_data, demographic_var)
        
        # Calculate the impact of the district on each category
        I_c = category_impacts(H_indi, H_dist, voter_data, demographic_var)
        
        # Calculate the average impact of the district on each category
        category_counts = np.bincount(voter_data.data[demographic_var])
        avg_I_c = I_c / category_counts
        
        # Get category names
        category_names = DEMOGRAPHIC_VARIABLES[demographic_var]['name_getter'](voter_data)
        
        # Calculate all pairwise impact gaps
        num_categories = DEMOGRAPHIC_VARIABLES[demographic_var]['num_categories'](voter_data)
        pairwise_gaps = []
        for i in range(num_categories):
            for j in range(i + 1, num_categories):
                gap = abs(avg_I_c[i] - avg_I_c[j])
                cat1, cat2 = category_names[i], category_names[j]
                pairwise_gaps.append({
                    'category1': cat1,
                    'category2': cat2,
                    'gap': gap,
                    'avg_impact1': avg_I_c[i],
                    'avg_impact2': avg_I_c[j]
                })
        
        # Sort gaps by magnitude
        pairwise_gaps.sort(key=lambda x: x['gap'], reverse=True)
        
        # Calculate overall impact gap (difference between max and min impacts)
        impact_gap = np.max(avg_I_c) - np.min(avg_I_c)
        max_impact_category = category_names[np.argmax(avg_I_c)]
        min_impact_category = category_names[np.argmin(avg_I_c)]
            
        return {
            'demographic_var': demographic_var,
            'impact_gap': impact_gap,  # Maximum gap (for backward compatibility)
            'category_impacts': I_c,
            'avg_category_impacts': avg_I_c,
            'category_counts': category_counts,
            'individual_entropies': H_indi,
            'district_entropies': H_dist,
            'max_impact_category': max_impact_category,
            'min_impact_category': min_impact_category,
            'category_names': category_names,
            'pairwise_gaps': pairwise_gaps
        }
        
    except Exception as e:
        logging.error(f"Error calculating impact gap: {str(e)}")
        raise

def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the impact gap calculation script.
    
    Args:
        args: Parsed command line arguments
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load voter data
        voter_data = VoterData.from_hdf5(args.input)
        
        # Calculate impact gap
        results = calculate_impact_gap(voter_data, args.demographic)
        
        # Log results
        logging.info(f"\nImpact Gap Analysis for {results['demographic_var']}:")
        logging.info(f"Maximum Impact Gap: {results['impact_gap']:.4f}")
        logging.info(f"Most Impacted: {results['max_impact_category']}")
        logging.info(f"Least Impacted: {results['min_impact_category']}")
        
        # Log all pairwise gaps
        logging.info("\nPairwise Impact Gaps (sorted by magnitude):")
        for gap_info in results['pairwise_gaps']:
            logging.info(
                f"{gap_info['category1']} vs {gap_info['category2']}: "
                f"Gap = {gap_info['gap']:.4f} "
                f"(Impacts: {gap_info['avg_impact1']:.4f} vs {gap_info['avg_impact2']:.4f})"
            )
        
        logging.info("\nDetailed Category Statistics:")
        for category, (impact, avg_impact) in enumerate(zip(
            results['category_impacts'],
            results['avg_category_impacts']
        )):
            category_name = results['category_names'][category]
            count = results['category_counts'][category]
            logging.info(
                f"{category_name}: Impact = {impact:.4f}, "
                f"Avg Impact = {avg_impact:.4f}, "
                f"Count = {count}"
            )
        
        # Save results
        np.savez(
            output_dir / f'impact_gap_{args.demographic}_results.npz',
            **results,
            metadata=voter_data.metadata.to_dict()
        )
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        help='Input HDF5 file containing voter data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--demographic',
        help='Demographic variable to analyze',
        type=str,
        choices=list(DEMOGRAPHIC_VARIABLES.keys()),
        default='party_id'
    )
    parser.add_argument(
        '--out',
        help='Output directory for impact analysis results',
        type=str,
        default='impact_gap_output'
    )
    
    args = parser.parse_args()
    main(args)