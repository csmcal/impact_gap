#!/usr/bin/env python3
"""
Enhanced Voter Data Generation Script
Author: Christopher McAllester
Updated: 2024

This script generates synthetic voter data with simple demographic correlations
and geographic distributions, supporting:
- Arbitrary state and district geometries
- Correlated demographic variables
- Urban/rural population distributions
- HDF5 output following voter_data_schema.py structure
"""

import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point, Polygon
from scipy.stats import multivariate_normal
import h5py
from typing import Dict, List, Tuple, Optional
import argparse
from voter_data_schema import (
    VoterData,
    VoterMetadata,
    Race,
    Education
)

class DemographicCorrelations:
    """Manages correlations between demographic variables."""
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        # Correlation matrix for demographic variables
        # Order: [party, age, race, education, income, household_size]
        self.correlation_matrix = np.array([
            [1.0,  0.1,  0.3,  0.2,  0.15, -0.1],  # party
            [0.1,  1.0,  0.0,  0.1,  0.3,  -0.2],  # age
            [0.3,  0.0,  1.0,  0.1,  0.2,   0.1],  # race
            [0.2,  0.1,  0.1,  1.0,  0.4,  -0.1],  # education
            [0.15, 0.3,  0.2,  0.4,  1.0,   0.0],  # income
            [-0.1, -0.2, 0.1, -0.1,  0.0,   1.0]   # household_size
        ])
        
        # Convert correlation to covariance matrix
        self.covariance_matrix = self.correlation_matrix.copy()
        
    def sample_demographics(self, n_samples: int) -> np.ndarray:
        """Sample correlated demographic variables."""
        # Generate multivariate normal samples
        samples = multivariate_normal.rvs(
            mean=np.zeros(6),
            cov=self.covariance_matrix,
            size=n_samples,
            random_state=self.rng
        )
        
        # Convert to appropriate ranges and types
        demographics = np.zeros(n_samples, dtype=VoterData.create_dtype())
        
        # Convert continuous samples to categorical variables
        demographics['party_id'] = (self.rng.random(n_samples) < 
            1 / (1 + np.exp(-samples[:, 0]))).astype(np.int16)  # logistic transform
        
        # Age: 18-95 years
        demographics['age'] = np.clip(
            45 + samples[:, 1] * 15, 18, 95
        ).astype(np.int16)
        
        # Race: weighted sampling based on local demographics
        race_probs = np.exp(samples[:, 2:3] * 0.5)
        race_probs = race_probs / race_probs.sum(axis=1, keepdims=True)
        demographics['race'] = self.rng.choice(
            len(Race),
            size=n_samples,
            p=[0.6, 0.15, 0.1, 0.05, 0.02, 0.05, 0.03]  # example proportions
        ).astype(np.int8)
        
        # Education: ordered categorical
        edu_scores = samples[:, 3]
        demographics['education'] = np.digitize(
            edu_scores,
            bins=[-np.inf, -0.5, 0, 0.5, 1.0]
        ).astype(np.int8)
        
        # Income decile: 1-10
        income_scores = samples[:, 4]
        demographics['income_decile'] = np.clip(
            np.digitize(
                income_scores,
                bins=np.linspace(-2, 2, 10)
            ),
            1, 10
        ).astype(np.int8)
        
        # Household size: 1-8+
        household_scores = samples[:, 5]
        demographics['household_size'] = np.clip(
            np.round(3 + household_scores).astype(np.int8),
            1, 8
        ).astype(np.int8)
        
        return demographics

class UrbanCluster:
    """Represents an urban area with distinct demographic patterns."""
    
    def __init__(
        self,
        center: Tuple[float, float],
        radius: float,
        population: int,
        demographic_bias: Dict[str, float]
    ):
        self.center = center
        self.radius = radius
        self.population = population
        self.demographic_bias = demographic_bias
        
    def generate_points(self, rng: np.random.Generator) -> np.ndarray:
        """Generate points following a 2D normal distribution."""
        points = rng.multivariate_normal(
            mean=self.center,
            cov=[[self.radius/3, 0], [0, self.radius/3]],
            size=self.population
        )
        return points

class VoterGenerator:
    """Generates synthetic voter data with realistic patterns."""
    
    def __init__(
        self,
        state_boundary: gpd.GeoDataFrame,
        district_boundaries: gpd.GeoDataFrame,
        urban_clusters: List[UrbanCluster],
        total_population: int,
        seed: Optional[int] = None
    ):
        self.state_boundary = state_boundary
        self.district_boundaries = district_boundaries
        self.urban_clusters = urban_clusters
        self.total_population = total_population
        self.rng = np.random.default_rng(seed)
        
        # Initialize demographic correlations
        self.demographics = DemographicCorrelations(self.rng)
        
        # Calculate urban/rural split
        self.urban_population = sum(c.population for c in urban_clusters)
        self.rural_population = total_population - self.urban_population
    
    def generate_voter_locations(self) -> np.ndarray:
        """Generate voter locations for both urban and rural areas."""
        voter_data = np.zeros(self.total_population, dtype=VoterData.create_dtype())
        
        # Generate urban points
        current_idx = 0
        for cluster in self.urban_clusters:
            points = cluster.generate_points(self.rng)
            end_idx = current_idx + cluster.population
            voter_data['longitude'][current_idx:end_idx] = points[:, 0]
            voter_data['latitude'][current_idx:end_idx] = points[:, 1]
            current_idx = end_idx
        
        # Generate rural points using rejection sampling
        rural_points = []
        bounds = self.state_boundary.total_bounds
        while len(rural_points) < self.rural_population:
            points = self.rng.uniform(
                low=[bounds[0], bounds[1]],
                high=[bounds[2], bounds[3]],
                size=(self.rural_population * 2, 2)
            )
            for point in points:
                if len(rural_points) >= self.rural_population:
                    break
                if self.state_boundary.contains(Point(point)).any():
                    rural_points.append(point)
        
        rural_points = np.array(rural_points)
        voter_data['longitude'][current_idx:] = rural_points[:, 0]
        voter_data['latitude'][current_idx:] = rural_points[:, 1]
        
        return voter_data
    
    def assign_districts(self, voter_data: np.ndarray) -> np.ndarray:
        """Assign districts based on point locations."""
        points = gpd.GeoSeries(
            [Point(x, y) for x, y in zip(
                voter_data['longitude'],
                voter_data['latitude']
            )]
        )
        
        for idx, district in self.district_boundaries.iterrows():
            mask = points.within(district.geometry)
            voter_data['district_id'][mask] = idx
        
        return voter_data
    
    def generate_voter_data(self) -> VoterData:
        """Generate complete voter dataset."""
        # Generate locations
        voter_data = self.generate_voter_locations()
        
        # Assign districts
        voter_data = self.assign_districts(voter_data)
        
        # Generate demographics
        demographics = self.demographics.sample_demographics(self.total_population)
        for field in demographics.dtype.names:
            if field not in ['longitude', 'latitude', 'district_id']:
                voter_data[field] = demographics[field]
        
        # Generate additional fields
        voter_data['polling_id'] = self.rng.integers(
            1000, 2000, self.total_population, dtype=np.int32
        )
        voter_data['precinct_id'] = self.rng.integers(
            1, 101, self.total_population, dtype=np.int32
        )
        voter_data['reg_year'] = self.rng.integers(
            1970, 2025, self.total_population, dtype=np.int16
        )
        voter_data['vote_history'] = self.rng.integers(
            0, 2**16, self.total_population, dtype=np.uint16
        )
        voter_data['census_block'] = self.rng.integers(
            1e5, 1e6, self.total_population, dtype=np.int64
        )
        voter_data['weight'] = np.ones(self.total_population, dtype=np.float32)
        
        # Create metadata
        metadata = VoterMetadata(
            num_voters=self.total_population,
            num_districts=len(self.district_boundaries),
            num_parties=2,
            district_names={
                i: f"District {i+1}"
                for i in range(len(self.district_boundaries))
            },
            party_names={0: "Democratic", 1: "Republican"},
            geographic_bounds=tuple(self.state_boundary.total_bounds),
            year=2024,
            state=self.state_boundary.get('STATE_NAME', ['Unknown'])[0],
            election_type="general"
        )
        
        return VoterData(data=voter_data, metadata=metadata)

def main(args: argparse.Namespace) -> None:
    """Generate synthetic voter data based on input parameters."""
    # Load geographic boundaries
    state_boundary = gpd.read_file(args.state_shapefile)
    district_boundaries = gpd.read_file(args.district_shapefile)
    
    # Define urban clusters from input file
    urban_clusters_df = pd.read_csv(args.urban_clusters)
    urban_clusters = [
        UrbanCluster(
            center=(row['longitude'], row['latitude']),
            radius=row['radius'],
            population=int(row['population']),
            demographic_bias={
                'party_id': row['party_bias'],
                'education': row['education_bias'],
                'income_decile': row['income_bias']
            }
        )
        for _, row in urban_clusters_df.iterrows()
    ]
    
    # Initialize generator
    generator = VoterGenerator(
        state_boundary=state_boundary,
        district_boundaries=district_boundaries,
        urban_clusters=urban_clusters,
        total_population=args.population,
        seed=args.seed
    )
    
    # Generate voter data
    voter_data = generator.generate_voter_data()
    
    # Save to HDF5 file
    output_path = Path(args.output)
    voter_data.to_hdf5(output_path)
    print(f"Generated voter data saved to: {output_path}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(voter_data.get_demographic_summary())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--state-shapefile',
        help='Path to state boundary shapefile',
        type=str,
        required=True
    )
    parser.add_argument(
        '--district-shapefile',
        help='Path to district boundaries shapefile',
        type=str,
        required=True
    )
    parser.add_argument(
        '--urban-clusters',
        help='CSV file defining urban cluster parameters',
        type=str,
        required=True
    )
    parser.add_argument(
        '--population',
        help='Total voter population to generate',
        type=int,
        required=True
    )
    parser.add_argument(
        '--output',
        help='Output path for HDF5 file',
        type=str,
        default='voter_data.h5'
    )
    parser.add_argument(
        '--seed',
        help='Random seed for reproducibility',
        type=int,
        default=None
    )
    
    args = parser.parse_args()
    main(args)