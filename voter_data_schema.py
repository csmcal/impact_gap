"""
Voter Data Schema and Processing
Author: Christopher McAllester
Created: 2024

This module defines the data structures and processing logic for voter data
that includes demographic information and supports efficient storage and computation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from pathlib import Path
import pandas as pd
from enum import IntEnum
import h5py
import logging

class Race(IntEnum):
    """Standard race categories based on census definitions."""
    WHITE = 0
    BLACK = 1
    ASIAN = 2
    NATIVE_AMERICAN = 3
    PACIFIC_ISLANDER = 4
    OTHER = 5
    MULTIRACIAL = 6

class Education(IntEnum):
    """Education level categories."""
    LESS_THAN_HS = 0
    HIGH_SCHOOL = 1
    SOME_COLLEGE = 2
    BACHELORS = 3
    GRADUATE = 4

@dataclass
class VoterMetadata:
    """Metadata about the voter dataset."""
    num_voters: int
    num_districts: int
    num_parties: int
    district_names: Dict[int, str]
    party_names: Dict[int, str]
    geographic_bounds: Tuple[float, float, float, float]  # min_lon, max_lon, min_lat, max_lat
    year: int
    state: str
    election_type: str
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary for storage."""
        return {
            'num_voters': self.num_voters,
            'num_districts': self.num_districts,
            'num_parties': self.num_parties,
            'district_names': str(self.district_names),
            'party_names': str(self.party_names),
            'geographic_bounds': self.geographic_bounds,
            'year': self.year,
            'state': self.state,
            'election_type': self.election_type
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'VoterMetadata':
        """Create metadata from dictionary."""
        return cls(
            num_voters=data['num_voters'],
            num_districts=data['num_districts'],
            num_parties=data['num_parties'],
            district_names=eval(data['district_names']),
            party_names=eval(data['party_names']),
            geographic_bounds=data['geographic_bounds'],
            year=data['year'],
            state=data['state'],
            election_type=data['election_type']
        )

@dataclass
class VoterData:
    """Voter data structure with demographic information."""
    # Core structured array containing all voter data
    data: npt.NDArray
    # Metadata about the dataset
    metadata: VoterMetadata
    
    @classmethod
    def create_dtype(cls) -> np.dtype:
        """Create the numpy dtype for the structured array."""
        return np.dtype([
            # Geographic data (using float32 for memory efficiency)
            ('longitude', 'f4'),    # longitude in degrees
            ('latitude', 'f4'),     # latitude in degrees
            
            # Political data (using int16 for small integers)
            ('party_id', 'i2'),     # political party identifier
            ('district_id', 'i2'),  # assigned district identifier
            
            # Demographics (using appropriate sized integers)
            ('age', 'i2'),          # age in years
            ('race', 'i1'),         # race category (using Race enum)
            ('education', 'i1'),    # education level (using Education enum)
            ('income_decile', 'i1'),  # income decile (1-10)
            ('household_size', 'i1'),  # number of people in household
            
            # Electoral data
            ('polling_id', 'i4'),   # polling location identifier
            ('precinct_id', 'i4'),  # precinct identifier
            ('reg_year', 'i2'),     # registration year
            
            # Voting history (using bit flags for efficiency)
            ('vote_history', 'u2'), # bit flags for last 16 elections
            
            # Census data
            ('census_block', 'i8'), # census block identifier
            
            # Statistical weight
            ('weight', 'f4')        # statistical weight for adjustments
        ])
    
    @classmethod
    def from_hdf5(cls, file_path: Path) -> 'VoterData':
        """Load voter data from HDF5 file format."""
        with h5py.File(file_path, 'r') as f:
            # Load the structured array
            data = f['voter_data'][:]
            
            # Load metadata
            metadata_dict = {}
            for key, value in f['metadata'].attrs.items():
                metadata_dict[key] = value
            
            metadata = VoterMetadata.from_dict(metadata_dict)
            
        return cls(data=data, metadata=metadata)
    
    def to_hdf5(self, file_path: Path) -> None:
        """Save voter data to HDF5 file format."""
        with h5py.File(file_path, 'w') as f:
            # Create main dataset with compression
            f.create_dataset('voter_data', data=self.data, compression='gzip', compression_opts=9)
            
            # Store metadata as attributes
            metadata_group = f.create_group('metadata')
            for key, value in self.metadata.to_dict().items():
                metadata_group.attrs[key] = value
    
    def get_demographic_summary(self, district_id: Optional[int] = None) -> pd.DataFrame:
        """Generate demographic summary statistics, optionally for a specific district."""
        mask = slice(None) if district_id is None else (self.data['district_id'] == district_id)
        
        summary = pd.DataFrame({
            'Total Voters': len(self.data[mask]),
            'Age (Mean)': np.mean(self.data[mask]['age']),
            'Age (Median)': np.median(self.data[mask]['age']),
            'Household Size (Mean)': np.mean(self.data[mask]['household_size']),
            'Income Decile (Mean)': np.mean(self.data[mask]['income_decile']),
        })
        
        # Add race distribution
        for race in Race:
            race_pct = np.mean(self.data[mask]['race'] == race.value) * 100
            summary[f'Race {race.name} (%)'] = race_pct
        
        # Add education distribution
        for edu in Education:
            edu_pct = np.mean(self.data[mask]['education'] == edu.value) * 100
            summary[f'Education {edu.name} (%)'] = edu_pct
        
        return summary 