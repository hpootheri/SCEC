"""
Ridgecrest Earthquake Observation Data Mapping Script
Maps Ridgecrest (2019) dataset to Current schema format
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RidgecrestDataMapper:
    """Class to handle mapping of Ridgecrest earthquake observation data to current schema"""
    
    def __init__(self):
        # Define field mappings based on semantic analysis
        self.ridgecrest_mapping = {
            'intid': None, 
            'origid': 'Station_ID',
            'observer': 'Creator',
            'obs_date': 'Date_of_Movement',
            'origin': 'Feature_Origin',
            'obs_affiliation': None,  # Store in Notes
            'team_id': None,  # Store in Notes  
            'team': None,  # Store in Notes
            'obs_position': None,  # Store in Notes
            'source': None,  # Earthquake source - store in Notes
            'citation': None,  # Store in Notes
            'description': 'Notes',
            'note': 'Vector_Offset_Feature_Notes',
            'fault_az_pref': 'Local_Fault_Azimuth_Degrees',
            'fault_dip_pref': 'Local_Fault_Dip',
            'sense': 'Slip_Sense',
            'rupture_width_pref': 'Rupture_Width_m',
            'rup_width_min': 'Rupture_Width_Min_m',
            'rup_width_max': 'Rupture_Width_Max_m',
            'fault_expression': 'Rupture_Expression',
            'scarp_facing_direction': 'Scarp_Facing_Direction',
            'vector_length_pref': 'Net_Slip_Preferred_cm',
            'vector_length_min': 'Net_Slip_Min_cm', 
            'vector_length_max': 'Net_Slip_Max_cm',
            'vect_az_pref': 'VM_Slip_Azimuth',
            'vect_plunge_pref': 'Plunge',
            'horiz_offset_pref': 'Horizontal_Separation_cm',
            'horiz_offset_min': 'Horizontal_Separation_Min_cm',
            'horiz_offset_max': 'Horizontal_Separation_Max_cm',
            'horiz_slip_type': 'Fault_Slip_Measurement_Type',
            'horiz_az_pref': 'Slip_Azimuth',
            'vert_offset_pref': 'Vertical_Separation_cm',
            'vert_offset_min': 'Vertical_Separation_Min_cm',
            'vert_offset_max': 'Vertical_Separation_Max_cm',
            'heave_pref': 'Heave_cm',
            'heave_min': 'Heave_min_cm',
            'heave_max': 'Heave_max_cm',
            'latitude': None,  # NEED TO ADD TO CURRENT SCHEMA
            'longitude': None,  # NEED TO ADD TO CURRENT SCHEMA
            'orig_lat': None,
            'orig_lon': None,
            'observed_feature': None,
            'feature_type': None,
            'striations_observed': None,
            'gouge_observed': None,
        }
        
        # Define current schema structure
        self.current_schema_fields = [
            'OBJECTID', 'Station_ID', 'Feature_Origin', 'Notes', 'Confidence_Feature_ID',
            'Mode_Observation', 'Slip_Sense', 'Scarp_Facing_Direction', 'Local_Fault_Dip',
            'Local_Fault_Azimuth_Degrees', 'Slip_Azimuth', 'Fault_Slip_Measurement_Type',
            'Heave_cm', 'Heave_min_cm', 'Heave_max_cm', 'Rupture_Expression',
            'Rupture_Width_m', 'Rupture_Width_Min_m', 'Rupture_Width_Max_m',
            'VM_Slip_Azimuth', 'Plunge', 'Net_Slip_Preferred_cm', 'Net_Slip_Min_cm',
            'Net_Slip_Max_cm', 'Vector_Measurement_Confidence', 'Vector_Offset_Feature_Notes',
            'Horizontal_Separation_cm', 'Horizontal_Separation_Min_cm', 'Horizontal_Separation_Max_cm',
            'Vertical_Separation_cm', 'Vertical_Separation_Min_cm', 'Vertical_Separation_Max_cm',
            'Slip_Measurement_Confidence', 'Slip_Offset_Feature_Notes', 'Diameter_m',
            'Height_of_Material_m', 'Estimated_Max_VertMov_m', 'LQ_Area_Affected_sqm',
            'Date_of_Movement', 'Displacement_Amount', 'Est_Direction_SM', 'Landslide_Feature',
            'Slide_Type', 'Material_Type', 'Depth', 'SM_Area_Affected_sqm',
            'Est_Max_Drop_Elev_ft', 'Length_Exposed_Downslope', 'Cause_of_Damage',
            'Facility_Affected', 'Utility_Affected', 'Damage_Severity', 'GlobalID',
            'CreationDate', 'Creator', 'EditDate', 'Editor'
        ]

    def load_ridgecrest_data(self, file_path):
        """Load Ridgecrest dataset"""
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            logger.info(f"Loaded Ridgecrest data: {len(df)} records, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading Ridgecrest data: {e}")
            return None

    def create_notes_field(self, row, unmapped_fields):
        """Create consolidated Notes field from multiple unmapped fields"""
        notes_parts = []
        
        for field in unmapped_fields:
            if field in row and pd.notna(row[field]) and row[field] != '':
                field_name = field.replace('_', ' ').title()
                notes_parts.append(f"{field_name}: {row[field]}")
        
        return "; ".join(notes_parts) if notes_parts else ""

    def map_ridgecrest_to_current(self, ridgecrest_df):
        """Map Ridgecrest dataset to current schema"""
        logger.info("Mapping Ridgecrest data to current schema...")
        
        # Initialize current schema dataframe  
        current_df = pd.DataFrame(columns=self.current_schema_fields)
        
        # Unmapped fields that should go in Notes
        ridgecrest_unmapped = [
            'obs_affiliation', 'team_id', 'team', 'obs_position', 'source', 'citation',
            'observed_feature', 'feature_type', 'striations_observed', 'gouge_observed',
            'fault_az_min', 'fault_az_max', 'fault_dip_min', 'fault_dip_max',
            'local_frac_az_min', 'local_frac_az_pref', 'local_frac_az_max',
            'vect_plunge_min', 'vect_plunge_max', 'vect_az_min', 'vect_az_max',
            'aperture_min', 'aperture_pref', 'aperture_max', 'horiz_az_min', 'horiz_az_max',
            'vert_slip_type', 'heave_type'
        ]
        
        for idx, row in ridgecrest_df.iterrows():
            mapped_row = {}
            
            # Generate OBJECTID
            mapped_row['OBJECTID'] = idx + 1
            
            # Apply direct mappings
            for ridgecrest_field, current_field in self.ridgecrest_mapping.items():
                if current_field and ridgecrest_field in row:
                    mapped_row[current_field] = row[ridgecrest_field]
            
            # Handle Notes field - combine description with unmapped fields
            notes_parts = []
            if 'description' in row and pd.notna(row['description']):
                notes_parts.append(str(row['description']))
            
            # Add unmapped field info to notes
            unmapped_notes = self.create_notes_field(row, ridgecrest_unmapped)
            if unmapped_notes:
                notes_parts.append(f"Additional: {unmapped_notes}")
            
            # CRITICAL: Add location data to notes since it's missing from current schema
            location_parts = []
            for coord_field in ['latitude', 'longitude', 'orig_lat', 'orig_lon']:
                if coord_field in row and pd.notna(row[coord_field]):
                    location_parts.append(f"{coord_field}: {row[coord_field]}")
            
            if location_parts:
                notes_parts.append(f"Coordinates: {'; '.join(location_parts)}")
            
            mapped_row['Notes'] = "; ".join(notes_parts) if notes_parts else ""
            
            # Set default values for system fields
            mapped_row['CreationDate'] = datetime.now()
            mapped_row['EditDate'] = datetime.now() 
            mapped_row['Editor'] = 'Ridgecrest_Migration'
            
            # Add dataset source identifier
            if 'Notes' in mapped_row:
                mapped_row['Notes'] = f"Source: Ridgecrest 2019; {mapped_row['Notes']}"
            else:
                mapped_row['Notes'] = "Source: Ridgecrest 2019"
            
            # Append to dataframe
            current_df = pd.concat([current_df, pd.DataFrame([mapped_row])], ignore_index=True)
        
        logger.info(f"Mapped {len(current_df)} Ridgecrest records to current schema")
        return current_df

    def finalize_dataset(self, current_df):
        """Finalize the mapped dataset"""
        logger.info("Finalizing dataset...")
        
        # Fill NaN values appropriately
        for col in current_df.columns:
            if col in ['Notes', 'Vector_Offset_Feature_Notes', 'Slip_Offset_Feature_Notes']:
                current_df[col] = current_df[col].fillna('')
            elif current_df[col].dtype == 'object':
                current_df[col] = current_df[col].fillna('')
            else:
                current_df[col] = current_df[col].fillna(np.nan)
        
        logger.info(f"Finalized dataset: {len(current_df)} total records")
        return current_df

def main():
    """Main execution function"""
    # Initialize mapper
    mapper = RidgecrestDataMapper()
    
    # File path (update this to your actual file location)
    ridgecrest_file = "ridgecrest_observations.csv"  # or .xlsx
    
    try:
        # Load dataset
        logger.info("Starting Ridgecrest data migration...")
        ridgecrest_df = mapper.load_ridgecrest_data(ridgecrest_file)
        
        if ridgecrest_df is None:
            logger.error("Failed to load Ridgecrest dataset")
            return
        
        # Map to current schema
        ridgecrest_current = mapper.map_ridgecrest_to_current(ridgecrest_df)
        
        # Finalize
        final_dataset = mapper.finalize_dataset(ridgecrest_current)
        
        # Save dataset
        output_file = f"ridgecrest_current_schema_{datetime.now().strftime('%Y%m%d')}.csv"
        final_dataset.to_csv(output_file, index=False)
        logger.info(f"Mapped dataset saved to: {output_file}")
        
        print("\nRidgecrest migration completed successfully!")
        print(f"Output file: {output_file}")
        print(f"Records processed: {len(final_dataset)}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")

if __name__ == "__main__":
    main()