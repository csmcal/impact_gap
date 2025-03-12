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
- Handles CSV input for voter data

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
    python impact_gap.py --v voter_data.csv [--h header_lines] [--out output_dir]
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
import numpy.typing as npt
import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fsolve
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class VoterData:
    """Container for voter-related data and calculations."""
    coords: npt.NDArray
    party: npt.NDArray
    district: npt.NDArray
    num_voters: int
    num_parties: int
    num_districts: int
    ideal_district_size: float

    @classmethod
    def from_array(cls, voters: npt.NDArray) -> 'VoterData':
        """Create VoterData from a structured numpy array."""
        num_voters = voters.shape[0]
        num_parties = np.amax(voters['party']) + 1
        num_districts = np.amax(voters['district']) + 1
        
        return cls(
            coords=np.vstack((voters['x1'], voters['x2'])).T,
            party=voters['party'],
            district=voters['district'],
            num_voters=num_voters,
            num_parties=num_parties,
            num_districts=num_districts,
            ideal_district_size=num_voters / num_districts
        )

    def validate(self) -> None:
        """Validate the voter data for basic requirements."""
        if self.num_voters <= 10:
            raise ValueError(f'Too few voters: {self.num_voters} (minimum 10 required)')
        if self.num_parties <= 1:
            raise ValueError('At least two parties required for impact analysis')
        if self.num_districts <= 1:
            raise ValueError('At least two districts required for gerrymandering analysis')

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
    Estimate the scaling factor for constant district density.
    
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
        pairwise_dists = pdist(voter_data.coords, metric='euclidean')
        dist_matrix = squareform(pairwise_dists)
        
        # Use maximum distance as initial bandwidth
        pilot_h = np.max(pairwise_dists)
        
        h = estimate_scaling(dist_matrix, pilot_h, voter_data.ideal_district_size)
        K_scaled = biv_gauss(dist_matrix, h)
        
        return K_scaled / voter_data.ideal_district_size
    
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
    voter_data: VoterData
) -> npt.NDArray:
    """
    Calculate entropy of 'average district' at each voter location.
    
    Args:
        K_scaled: Scaled kernel density matrix
        voter_data: VoterData instance containing voter information
    
    Returns:
        Array of entropy values for each voter location
    """
    party_probs = np.zeros((voter_data.num_parties, voter_data.num_voters))
    
    # Vectorized operation for party density summation
    for party in range(voter_data.num_parties):
        mask = voter_data.party == party
        party_probs[party] = np.sum(K_scaled[mask], axis=0)
    
    return entropy(party_probs)

def district_entropies(voter_data: VoterData) -> npt.NDArray:
    """
    Calculate entropy of electorate in each district.
    
    Args:
        voter_data: VoterData instance containing voter information
    
    Returns:
        Array of entropy values for each district
    """
    # Calculate party counts per district using numpy operations
    counts = np.zeros((voter_data.num_parties, voter_data.num_districts))
    for party in range(voter_data.num_parties):
        for district in range(voter_data.num_districts):
            counts[party, district] = np.sum(
                (voter_data.party == party) & (voter_data.district == district)
            )
    
    # Calculate frequencies and entropy
    freqs = counts / np.sum(counts, axis=0)
    return entropy(freqs)

def party_impacts(
    H_i: npt.NDArray,
    H_d: npt.NDArray,
    voter_data: VoterData
) -> npt.NDArray:
    """
    Calculate impact of districting plan on each party.
    
    Args:
        H_i: Individual entropy values
        H_d: District entropy values
        voter_data: VoterData instance
    
    Returns:
        Array of impact values for each party
    """
    I_p = np.zeros(voter_data.num_parties)
    for party in range(voter_data.num_parties):
        party_mask = voter_data.party == party
        I_p[party] = np.sum(
            H_d[voter_data.district[party_mask]] - H_i[party_mask]
        )
    return I_p

def calculate_impact_gap(voter_data: VoterData) -> Dict[str, Any]:
    """
    Calculate the impact gap metric for a given voter distribution.
    
    Args:
        voter_data: VoterData instance containing voter information
    
    Returns:
        Dictionary containing impact gap results and intermediate calculations
    """
    try:
        # Validate input data
        voter_data.validate()
        
        # Calculate kernel densities
        K_scaled = estimate_density(voter_data)
        
        # Calculate entropies
        H_indi = individual_entropies(K_scaled, voter_data)
        H_dist = district_entropies(voter_data)
        
        # Calculate party impacts
        I_p = party_impacts(H_indi, H_dist, voter_data)
        
        # Calculate per-voter impacts
        party_counts = np.bincount(voter_data.party)
        avg_I_p = I_p / party_counts
        
        # Calculate impact gap (currently for two-party system)
        if voter_data.num_parties == 2:
            impact_gap = avg_I_p[1] - avg_I_p[0]
        else:
            logging.warning("Impact gap calculation optimized for two-party system")
            impact_gap = None
            
        return {
            'impact_gap': impact_gap,
            'party_impacts': I_p,
            'avg_party_impacts': avg_I_p,
            'party_counts': party_counts,
            'individual_entropies': H_indi,
            'district_entropies': H_dist
        }
        
    except Exception as e:
        logging.error(f"Error calculating impact gap: {str(e)}")
        raise

def parse_voter_file(file_path: str, header_lines: int = 0) -> VoterData:
    """
    Parse voter data from CSV file.
    
    Args:
        file_path: Path to CSV file
        header_lines: Number of header lines to skip
    
    Returns:
        VoterData instance containing parsed voter information
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Voter file not found: {file_path}")
            
        voters = np.genfromtxt(
            file_path,
            dtype=[('x1', 'f8'), ('x2', 'f8'), ('party', 'i4'), ('district', 'i4')],
            delimiter=',',
            skip_header=header_lines
        )
        
        return VoterData.from_array(voters)
        
    except Exception as e:
        logging.error(f"Error parsing voter file: {str(e)}")
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
        
        # Parse voter data
        voter_data = parse_voter_file(args.v, args.h)
        
        # Calculate impact gap
        results = calculate_impact_gap(voter_data)
        
        # Log results
        logging.info("Calculation Results:")
        logging.info(f"Impact Gap: {results['impact_gap']}")
        logging.info(f"Party Impacts: {results['party_impacts']}")
        logging.info(f"Average Party Impacts: {results['avg_party_impacts']}")
        
        # Save results to output directory
        np.savez(
            output_dir / 'impact_gap_results.npz',
            **results
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
        '--v',
        help='Input CSV file with voter data (x, y coords and party/district IDs)',
        type=str,
        required=True
    )
    parser.add_argument(
        '--h',
        help='Number of header lines in voter CSV file',
        type=int,
        default=0
    )
    parser.add_argument(
        '--out',
        help='Output directory for impact analysis results',
        type=str,
        default='impact_gap_output'
    )
    
    args = parser.parse_args()
    main(args)