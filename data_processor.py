#!/usr/bin/env python3
"""
Voter Data Processor for Impact Gap Analysis
Author: Christopher McAllester
Created: 2024

This script downloads and processes voter data from multiple sources:
- Voting and Election Science Team (VEST)
- Redistricting Data Hub
- Harvard Dataverse - US Elections Project
- Census Bureau Redistricting Data (PL 94-171)

The script transforms the data into the format required by the impact_gap.py analysis:
- x coordinate (longitude)
- y coordinate (latitude)
- party affiliation
- district assignment

Usage:
    python data_processor.py --state STATE_ABBREV --year YEAR --output OUTPUT_DIR
"""

import argparse
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import requests
import zipfile
import io
import json
from typing import Dict, List, Optional, Tuple, Any
import censusgeocode as cg
from dataclasses import dataclass
import warnings
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

@dataclass
class DataSource:
    """Container for data source information and authentication."""
    name: str
    base_url: str
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Set up authentication if needed."""
        if self.name == "census" and not self.api_key:
            logging.warning("Census API key not provided. Some features may be limited.")

class DataProcessor:
    """Main class for downloading and processing voter data."""
    
    def __init__(self, state: str, year: int, output_dir: Path):
        """
        Initialize the data processor.
        
        Args:
            state: Two-letter state abbreviation
            year: Election year
            output_dir: Output directory for processed data
        """
        self.state = state.upper()
        self.year = year
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data sources
        self.sources = {
            "vest": DataSource(
                "vest",
                "https://dataverse.harvard.edu/api/access/datafile/"
            ),
            "rdh": DataSource(
                "rdh",
                "https://redistrictingdatahub.org/api/v1/"
            ),
            "harvard": DataSource(
                "harvard",
                "https://dataverse.harvard.edu/api/access/datafile/"
            ),
            "census": DataSource(
                "census",
                "https://api.census.gov/data/",
                self._get_census_api_key()
            )
        }
        
        # Create cache directory for downloaded files
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(exist_ok=True)

    def _get_census_api_key(self) -> Optional[str]:
        """Get Census API key from environment or config file."""
        try:
            config_path = Path.home() / ".census_api_key"
            if config_path.exists():
                return config_path.read_text().strip()
            return None
        except Exception as e:
            logging.warning(f"Could not read Census API key: {e}")
            return None

    def download_vest_data(self) -> pd.DataFrame:
        """
        Download precinct-level election results from VEST.
        
        Returns:
            DataFrame containing election results
        """
        logging.info(f"Downloading VEST data for {self.state} {self.year}")
        try:
            # VEST data is hosted on Harvard Dataverse
            # We need to query the API to get the correct file ID
            # This is a simplified example - actual implementation would need
            # to handle the specific file IDs for each state/year
            
            # For demonstration, we'll show the process for 2020 data
            vest_data = pd.DataFrame()  # Placeholder
            
            return vest_data
            
        except Exception as e:
            logging.error(f"Error downloading VEST data: {e}")
            raise

    def download_rdh_data(self) -> gpd.GeoDataFrame:
        """
        Download shapefile data from Redistricting Data Hub.
        
        Returns:
            GeoDataFrame containing district boundaries
        """
        logging.info(f"Downloading RDH data for {self.state} {self.year}")
        try:
            # RDH provides shapefiles for districts
            # We need to download and process the relevant shapefile
            
            rdh_data = gpd.GeoDataFrame()  # Placeholder
            
            return rdh_data
            
        except Exception as e:
            logging.error(f"Error downloading RDH data: {e}")
            raise

    def download_census_data(self) -> pd.DataFrame:
        """
        Download PL 94-171 redistricting data from Census Bureau.
        
        Returns:
            DataFrame containing census block level data
        """
        logging.info(f"Downloading Census data for {self.state} {self.year}")
        try:
            if not self.sources["census"].api_key:
                raise ValueError("Census API key required")
                
            # Census API endpoint for PL 94-171 data
            # This is a simplified example - actual implementation would need
            # to handle the specific tables and variables needed
            
            census_data = pd.DataFrame()  # Placeholder
            
            return census_data
            
        except Exception as e:
            logging.error(f"Error downloading Census data: {e}")
            raise

    def process_voter_locations(self, 
                              precinct_data: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Process voter location data from precinct centroids.
        
        Args:
            precinct_data: GeoDataFrame containing precinct boundaries
            
        Returns:
            DataFrame with voter coordinates
        """
        try:
            # Calculate precinct centroids
            centroids = precinct_data.geometry.centroid
            
            # Create DataFrame with coordinates
            locations = pd.DataFrame({
                'x': centroids.x,
                'y': centroids.y
            })
            
            return locations
            
        except Exception as e:
            logging.error(f"Error processing voter locations: {e}")
            raise

    def estimate_party_affiliation(self, 
                                 election_data: pd.DataFrame,
                                 census_data: Optional[pd.DataFrame] = None,
                                 confidence_threshold: float = 0.6) -> Tuple[pd.Series, pd.Series]:
        """
        Estimate party affiliation by considering:
        1. Historical voting patterns from multiple elections
        2. Demographic correlations (if census data available)
        3. Geographic clustering effects
        4. Confidence scores for each estimation
        
        Args:
            election_data: DataFrame containing election results with columns:
                - precinct_id: Unique identifier for each precinct
                - election_type: Type of election (president, senate, etc.)
                - election_year: Year of the election
                - dem_votes: Democratic votes
                - rep_votes: Republican votes
                - other_votes: Other party votes
            census_data: Optional DataFrame with demographic information
            confidence_threshold: Minimum confidence required for party assignment
            
        Returns:
            Tuple of (party_affiliations, confidence_scores)
            where party_affiliations is 0 (Republican) or 1 (Democratic)
        """
        try:
            logging.info("Estimating party affiliations with enhanced model")
            
            # Initialize output arrays
            num_precincts = len(election_data['precinct_id'].unique())
            party_scores = np.zeros(num_precincts)
            confidence_scores = np.zeros(num_precincts)
            
            # 1. Calculate base party lean from historical voting patterns
            historical_scores = self._calculate_historical_scores(election_data)
            party_scores += historical_scores['party_lean'].values
            confidence_scores += historical_scores['vote_consistency'].values * 0.4  # Weight: 40%
            
            # 2. Incorporate demographic correlations if available
            if census_data is not None:
                demographic_scores = self._calculate_demographic_scores(census_data)
                party_scores += demographic_scores['party_lean'].values
                confidence_scores += demographic_scores['demographic_confidence'].values * 0.3  # Weight: 30%
            
            # 3. Add geographic clustering effects
            geographic_scores = self._calculate_geographic_scores(
                election_data,
                historical_scores['party_lean']
            )
            party_scores += geographic_scores['spatial_lean'].values
            confidence_scores += geographic_scores['spatial_confidence'].values * 0.3  # Weight: 30%
            
            # Normalize scores to [-1, 1] range
            party_scores = np.clip(party_scores / 3, -1, 1)
            
            # Normalize confidence scores to [0, 1] range
            confidence_scores = np.clip(confidence_scores, 0, 1)
            
            # Convert scores to binary party affiliations
            party_affiliations = pd.Series(
                np.where(party_scores > 0, 1, 0),  # 1 for Democratic, 0 for Republican
                dtype='int32'
            )
            
            # Mark low-confidence predictions as -1
            party_affiliations = pd.Series(
                np.where(confidence_scores >= confidence_threshold,
                        party_affiliations,
                        -1),
                dtype='int32'
            )
            
            # Log summary statistics
            dem_count = (party_affiliations == 1).sum()
            rep_count = (party_affiliations == 0).sum()
            uncertain_count = (party_affiliations == -1).sum()
            
            logging.info(f"Party affiliation estimates:")
            logging.info(f"  Democratic: {dem_count}")
            logging.info(f"  Republican: {rep_count}")
            logging.info(f"  Uncertain: {uncertain_count}")
            logging.info(f"  Average confidence: {confidence_scores.mean():.2f}")
            
            return party_affiliations, pd.Series(confidence_scores)
            
        except Exception as e:
            logging.error(f"Error estimating party affiliations: {e}")
            raise

    def _calculate_historical_scores(self, election_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate party lean scores based on historical voting patterns.
        
        Args:
            election_data: DataFrame containing election results
            
        Returns:
            DataFrame with columns:
            - precinct_id: Precinct identifier
            - party_lean: Score between -1 (Republican) and 1 (Democratic)
            - vote_consistency: Consistency of voting patterns
        """
        try:
            # Group by precinct and calculate metrics
            precinct_metrics = election_data.groupby('precinct_id').agg({
                'dem_votes': list,
                'rep_votes': list,
                'other_votes': list,
                'election_year': list
            })
            
            # Calculate Democratic vote share for each election
            dem_shares = []
            for i in range(len(precinct_metrics)):
                dem_votes = np.array(precinct_metrics.iloc[i]['dem_votes'])
                rep_votes = np.array(precinct_metrics.iloc[i]['rep_votes'])
                total_major_votes = dem_votes + rep_votes
                dem_shares.append(dem_votes / np.maximum(total_major_votes, 1))
            
            # Calculate mean and standard deviation of Democratic vote share
            dem_share_means = np.array([np.mean(shares) for shares in dem_shares])
            dem_share_stds = np.array([np.std(shares) if len(shares) > 1 else 1.0 
                                     for shares in dem_shares])
            
            # Convert to party lean scores (-1 to 1)
            party_lean = 2 * (dem_share_means - 0.5)
            
            # Calculate vote consistency (inverse of standard deviation)
            vote_consistency = 1 / (1 + dem_share_stds)
            
            return pd.DataFrame({
                'precinct_id': precinct_metrics.index,
                'party_lean': party_lean,
                'vote_consistency': vote_consistency
            })
            
        except Exception as e:
            logging.error(f"Error calculating historical scores: {e}")
            raise

    def _calculate_demographic_scores(self, census_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate party lean scores based on demographic correlations.
        
        Args:
            census_data: DataFrame containing demographic information
            
        Returns:
            DataFrame with columns:
            - precinct_id: Precinct identifier
            - party_lean: Score between -1 (Republican) and 1 (Democratic)
            - demographic_confidence: Confidence in demographic prediction
        """
        try:
            # Define demographic correlations with party preference
            # These weights are based on historical voting patterns
            # and should be updated with recent election data
            demographic_weights = {
                'age_18_29_pct': 0.3,        # Younger voters lean Democratic
                'age_65_plus_pct': -0.2,      # Older voters lean Republican
                'college_grad_pct': 0.25,     # College education correlates with Democratic voting
                'urban_pct': 0.3,             # Urban areas lean Democratic
                'rural_pct': -0.3,            # Rural areas lean Republican
                'median_income': 0.1,         # Income has complex relationship
                'minority_pct': 0.3           # Minorities tend to lean Democratic
            }
            
            # Normalize demographic features
            normalized_data = census_data.copy()
            for col in demographic_weights.keys():
                if col in normalized_data.columns:
                    normalized_data[col] = (normalized_data[col] - normalized_data[col].mean()) / normalized_data[col].std()
            
            # Calculate weighted sum of demographic features
            party_lean = np.zeros(len(census_data))
            feature_count = 0
            
            for feature, weight in demographic_weights.items():
                if feature in normalized_data.columns:
                    party_lean += weight * normalized_data[feature]
                    feature_count += abs(weight)
            
            # Normalize party lean scores to [-1, 1]
            party_lean = np.clip(party_lean / feature_count, -1, 1)
            
            # Calculate confidence based on data completeness and variance
            demographic_confidence = np.ones(len(census_data))
            for feature in demographic_weights.keys():
                if feature in normalized_data.columns:
                    # Reduce confidence for missing or highly variable data
                    confidence_penalty = normalized_data[feature].isna().astype(float) * 0.2
                    demographic_confidence -= confidence_penalty
            
            demographic_confidence = np.clip(demographic_confidence, 0, 1)
            
            return pd.DataFrame({
                'precinct_id': census_data.index,
                'party_lean': party_lean,
                'demographic_confidence': demographic_confidence
            })
            
        except Exception as e:
            logging.error(f"Error calculating demographic scores: {e}")
            raise

    def _calculate_geographic_scores(self, 
                                  election_data: pd.DataFrame,
                                  historical_leans: pd.Series,
                                  neighbor_weight: float = 0.3) -> pd.DataFrame:
        """
        Calculate party lean scores based on geographic clustering effects.
        
        Args:
            election_data: DataFrame containing election results and precinct geometry
            historical_leans: Series of historical party leans by precinct
            neighbor_weight: Weight given to neighboring precinct effects
            
        Returns:
            DataFrame with columns:
            - precinct_id: Precinct identifier
            - spatial_lean: Score between -1 (Republican) and 1 (Democratic)
            - spatial_confidence: Confidence in spatial prediction
        """
        try:
            # Convert election data to GeoDataFrame if not already
            if not isinstance(election_data, gpd.GeoDataFrame):
                raise ValueError("Election data must include geometry for spatial analysis")
            
            # Find neighboring precincts
            neighbors = election_data.geometry.touches.sparse.todense()
            
            # Calculate spatially weighted average of historical leans
            spatial_leans = np.zeros(len(historical_leans))
            spatial_confidence = np.zeros(len(historical_leans))
            
            for i in range(len(historical_leans)):
                # Get neighboring precinct indices
                neighbor_idx = neighbors[i].nonzero()[0]
                
                if len(neighbor_idx) > 0:
                    # Calculate weighted average of neighboring leans
                    neighbor_leans = historical_leans.iloc[neighbor_idx]
                    spatial_leans[i] = (
                        (1 - neighbor_weight) * historical_leans.iloc[i] +
                        neighbor_weight * neighbor_leans.mean()
                    )
                    
                    # Higher confidence with more neighbors and consistent voting patterns
                    neighbor_std = neighbor_leans.std()
                    spatial_confidence[i] = (
                        0.5 +  # Base confidence
                        0.25 * (len(neighbor_idx) / 8) +  # More neighbors -> higher confidence
                        0.25 * (1 / (1 + neighbor_std))  # More consistency -> higher confidence
                    )
                else:
                    # No neighbors - use historical lean with reduced confidence
                    spatial_leans[i] = historical_leans.iloc[i]
                    spatial_confidence[i] = 0.5
            
            return pd.DataFrame({
                'precinct_id': election_data.index,
                'spatial_lean': spatial_leans,
                'spatial_confidence': spatial_confidence
            })
            
        except Exception as e:
            logging.error(f"Error calculating geographic scores: {e}")
            raise

    def assign_districts(self, 
                        voter_coords: pd.DataFrame, 
                        district_shapes: gpd.GeoDataFrame) -> pd.Series:
        """
        Assign voters to districts based on their coordinates.
        
        Args:
            voter_coords: DataFrame containing voter coordinates
            district_shapes: GeoDataFrame containing district boundaries
            
        Returns:
            Series containing district assignments
        """
        try:
            # Convert voter coordinates to GeoDataFrame
            voter_points = gpd.GeoDataFrame(
                voter_coords,
                geometry=gpd.points_from_xy(voter_coords.x, voter_coords.y)
            )
            
            # Perform spatial join
            districts = gpd.sjoin(
                voter_points,
                district_shapes,
                how="left",
                predicate="within"
            )["district"]
            
            return districts
            
        except Exception as e:
            logging.error(f"Error assigning districts: {e}")
            raise

    def create_impact_gap_input(self) -> pd.DataFrame:
        """
        Create the final input file for impact_gap analysis.
        
        Returns:
            DataFrame in the format required by impact_gap.py
        """
        try:
            # Download data from all sources
            vest_data = self.download_vest_data()
            rdh_data = self.download_rdh_data()
            census_data = self.download_census_data()
            
            # Process voter locations
            locations = self.process_voter_locations(rdh_data)
            
            # Estimate party affiliations
            party, confidence = self.estimate_party_affiliation(vest_data, census_data)
            
            # Assign districts
            districts = self.assign_districts(locations, rdh_data)
            
            # Combine all data
            final_data = pd.DataFrame({
                'x1': locations['x'],
                'x2': locations['y'],
                'party': party,
                'district': districts,
                'confidence': confidence
            })
            
            # Save to CSV
            output_file = self.output_dir / f"{self.state}_{self.year}_voter_data.csv"
            final_data.to_csv(output_file, index=False)
            
            logging.info(f"Data saved to {output_file}")
            return final_data
            
        except Exception as e:
            logging.error(f"Error creating impact gap input: {e}")
            raise

def main(args: argparse.Namespace) -> None:
    """
    Main entry point for the data processor script.
    
    Args:
        args: Parsed command line arguments
    """
    try:
        processor = DataProcessor(
            state=args.state,
            year=args.year,
            output_dir=args.output
        )
        
        processor.create_impact_gap_input()
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--state',
        help='Two-letter state abbreviation',
        type=str,
        required=True
    )
    parser.add_argument(
        '--year',
        help='Election year',
        type=int,
        required=True
    )
    parser.add_argument(
        '--output',
        help='Output directory for processed data',
        type=str,
        default='processed_data'
    )
    
    args = parser.parse_args()
    main(args) 