"""
Example script demonstrating the creation and usage of voter data format.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from voter_data_schema import (
    VoterData,
    VoterMetadata,
    Race,
    Education
)

def create_example_data(num_voters: int = 1000) -> None:
    """Create example voter data file with demographic information."""
    # Create metadata
    metadata = VoterMetadata(
        num_voters=num_voters,
        num_districts=5,
        num_parties=2,
        district_names={
            0: "District A",
            1: "District B",
            2: "District C",
            3: "District D",
            4: "District E"
        },
        party_names={
            0: "Democratic",
            1: "Republican"
        },
        geographic_bounds=(-74.5, -73.5, 40.5, 41.5),  # NYC area: min_lon, max_lon, min_lat, max_lat
        year=2024,
        state="NY",
        election_type="general"
    )
    
    # Create synthetic voter data
    rng = np.random.default_rng(42)  # for reproducibility
    
    # Initialize structured array
    voter_data = np.zeros(num_voters, dtype=VoterData.create_dtype())
    
    # Generate synthetic data
    voter_data['longitude'] = rng.uniform(-74.5, -73.5, num_voters)
    voter_data['latitude'] = rng.uniform(40.5, 41.5, num_voters)
    voter_data['party_id'] = rng.integers(0, 2, num_voters)
    voter_data['district_id'] = rng.integers(0, 5, num_voters)
    voter_data['age'] = rng.integers(18, 95, num_voters)
    voter_data['race'] = rng.integers(0, len(Race), num_voters)
    voter_data['education'] = rng.integers(0, len(Education), num_voters)
    voter_data['income_decile'] = rng.integers(1, 11, num_voters)
    voter_data['household_size'] = rng.integers(1, 8, num_voters)
    voter_data['polling_id'] = rng.integers(1000, 2000, num_voters)
    voter_data['precinct_id'] = rng.integers(1, 101, num_voters)
    voter_data['reg_year'] = rng.integers(1970, 2025, num_voters)
    voter_data['vote_history'] = rng.integers(0, 2**16, num_voters, dtype=np.uint16)
    voter_data['census_block'] = rng.integers(1e5, 1e6, num_voters)
    voter_data['weight'] = np.ones(num_voters)  # Default weight of 1.0
    
    # Create VoterData instance
    voter_dataset = VoterData(data=voter_data, metadata=metadata)
    
    # Save to HDF5 file
    output_path = Path("voter_data.h5")
    voter_dataset.to_hdf5(output_path)
    print(f"Created voter data file: {output_path}")
    
    # Demonstrate usage
    loaded_data = VoterData.from_hdf5(output_path)
    
    # Print overall demographic summary
    print("\nOverall Demographic Summary:")
    print(loaded_data.get_demographic_summary())
    
    # Print demographic summary for District A
    print("\nDemographic Summary for District A:")
    print(loaded_data.get_demographic_summary(district_id=0))
    
    # Example analysis: Calculate party distribution by education level
    print("\nParty Distribution by Education Level:")
    for edu in Education:
        mask = loaded_data.data['education'] == edu.value
        party_dist = np.bincount(
            loaded_data.data['party_id'][mask],
            minlength=metadata.num_parties
        ) / np.sum(mask)
        print(f"{edu.name}:")
        for party_id, percentage in enumerate(party_dist):
            print(f"  {metadata.party_names[party_id]}: {percentage:.1%}")

if __name__ == "__main__":
    create_example_data() 