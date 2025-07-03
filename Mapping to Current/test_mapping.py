"""
Earthquake Observation Data Mapping Script
Maps Napa (2014) and Ridgecrest (2019) datasets to Current schema format
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarthquakeDataMapper:
    """Class to handle mapping of earthquake observation data to current schema"""
    
    def __init__(self):
        # Define field mappings based on semantic analysis
        self.napa_mapping = {
            # Core identification
            'stnid': 'Station_ID',  # Station identifier
            'intid': None,  # Internal ID - not needed in current
            'observer': 'Creator',  # Person who made observation
            'obs_date': 'Date_of_Movement',  # When earthquake occurred
            'origin': 'Feature_Origin',  # Tectonic vs uncertain
            
            # Location data (CRITICAL - missing in current schema)
            'latitude': None,  # Need to add to current schema
            'longitude': None,  # Need to add to current schema
            'orig_lat': None,  # Original coordinates
            'orig_lon': None,  # Original coordinates
            
            # Descriptive fields
            'description': 'Notes',
            'citation': None,  # Could go in notes if needed
            'photo': None,  # Photo availability - could note in Notes
            
            # Fault geometry
            'fault_azimuth': 'Local_Fault_Azimuth_Degrees',
            
            # Measurements
            'ss_displacement': 'Horizontal_Separation_cm',  # Strike-slip displacement
            'ss_sense': 'Slip_Sense',
            'ext_offset': 'Heave_cm',  # Extension offset
            'comp_offset': None,  # Compression - could combine with heave
            'vert_offset': 'Vertical_Separation_cm',
            'upthrown_side': 'Scarp_Facing_Direction',
            
            # Feature classification
            'observed_feature': None,  # Store in Notes
            'trace': None,  # Trace identifier - store in Notes
        }
        
        self.ridgecrest_mapping = {
            # Core identification
            'intid': None,  # Will use origid for Station_ID instead
            'origid': 'Station_ID',
            'observer': 'Creator',
            'obs_date': 'Date_of_Movement',
            'origin': 'Feature_Origin',
            
            # Team/affiliation info
            'obs_affiliation': None,  # Store in Notes
            'team_id': None,  # Store in Notes  
            'team': None,  # Store in Notes
            'obs_position': None,  # Store in Notes
            'source': None,  # Earthquake source - store in Notes
            'citation': None,  # Store in Notes
            
            # Descriptive
            'description': 'Notes',
            'note': 'Vector_Offset_Feature_Notes',
            
            # Fault geometry - preferred values
            'fault_az_pref': 'Local_Fault_Azimuth_Degrees',
            'fault_dip_pref': 'Local_Fault_Dip',
            'sense': 'Slip_Sense',
            
            # Rupture characteristics
            'rupture_width_pref': 'Rupture_Width_m',
            'rup_width_min': 'Rupture_Width_Min_m',
            'rup_width_max': 'Rupture_Width_Max_m',
            'fault_expression': 'Rupture_Expression',
            'scarp_facing_direction': 'Scarp_Facing_Direction',
            
            # Slip vector measurements
            'vector_length_pref': 'Net_Slip_Preferred_cm',
            'vector_length_min': 'Net_Slip_Min_cm', 
            'vector_length_max': 'Net_Slip_Max_cm',
            'vect_az_pref': 'VM_Slip_Azimuth',
            'vect_plunge_pref': 'Plunge',
            
            # Horizontal offset
            'horiz_offset_pref': 'Horizontal_Separation_cm',
            'horiz_offset_min': 'Horizontal_Separation_Min_cm',
            'horiz_offset_max': 'Horizontal_Separation_Max_cm',
            'horiz_slip_type': 'Fault_Slip_Measurement_Type',
            'horiz_az_pref': 'Slip_Azimuth',
            
            # Vertical offset  
            'vert_offset_pref': 'Vertical_Separation_cm',
            'vert_offset_min': 'Vertical_Separation_Min_cm',
            'vert_offset_max': 'Vertical_Separation_Max_cm',
            
            # Heave (extension/compression)
            'heave_pref': 'Heave_cm',
            'heave_min': 'Heave_min_cm',
            'heave_max': 'Heave_max_cm',
            
            # Location (CRITICAL ISSUE - these fields exist in Ridgecrest but not Current)
            'latitude': None,  # NEED TO ADD TO CURRENT SCHEMA
            'longitude': None,  # NEED TO ADD TO CURRENT SCHEMA
            'orig_lat': None,
            'orig_lon': None,
            
            # Observational details - store in Notes
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

    def load_napa_data(self, file_path):
        """Load Napa dataset"""
        try:
            if file_path.endswith('.csv'):
                # Try comma-separated first
                df = pd.read_csv(file_path)
                
                # If only 1 column, try tab-separated
                if len(df.columns) == 1:
                    df = pd.read_csv(file_path, sep='\t')
                    
                # If still only 1 column, try other separators
                if len(df.columns) == 1:
                    df = pd.read_csv(file_path, sep='|')  # Try pipe separator
                    
            else:
                df = pd.read_excel(file_path)
                
            logger.info(f"Loaded Napa data: {len(df)} records, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Error loading Napa data: {e}")
            return None

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

    def create_notes_field(self, row, unmapped_fields, source_mapping):
        """Create consolidated Notes field from multiple unmapped fields"""
        notes_parts = []
        
        for field in unmapped_fields:
            if field in row and pd.notna(row[field]) and row[field] != '':
                field_name = field.replace('_', ' ').title()
                notes_parts.append(f"{field_name}: {row[field]}")
        
        return "; ".join(notes_parts) if notes_parts else ""

    def map_napa_to_current(self, napa_df):
        """Map Napa dataset to current schema"""
        logger.info("Mapping Napa data to current schema...")
        
        # Initialize current schema dataframe
        current_df = pd.DataFrame(columns=self.current_schema_fields)
        
        # Unmapped fields that should go in Notes
        napa_unmapped = ['citation', 'photo', 'observed_feature', 'trace', 'comp_offset']
        
        for idx, row in napa_df.iterrows():
            mapped_row = {}
            
            # Generate OBJECTID (auto-incrementing)
            mapped_row['OBJECTID'] = idx + 1
            
            # Apply direct mappings
            for napa_field, current_field in self.napa_mapping.items():
                if current_field and napa_field in row:
                    mapped_row[current_field] = row[napa_field]
            
            # Handle Notes field - combine description with unmapped fields
            notes_parts = []
            if 'description' in row and pd.notna(row['description']):
                notes_parts.append(str(row['description']))
            
            # Add unmapped field info to notes
            unmapped_notes = self.create_notes_field(row, napa_unmapped, self.napa_mapping)
            if unmapped_notes:
                notes_parts.append(f"Additional: {unmapped_notes}")
            
            mapped_row['Notes'] = "; ".join(notes_parts) if notes_parts else ""
            
            # Set default values for system fields
            mapped_row['CreationDate'] = datetime.now()
            mapped_row['EditDate'] = datetime.now()
            mapped_row['Editor'] = 'Napa_Migration'
            
            # Add dataset source identifier
            if 'Notes' in mapped_row:
                mapped_row['Notes'] = f"Source: Napa 2014; {mapped_row['Notes']}"
            else:
                mapped_row['Notes'] = "Source: Napa 2014"
            
            # Append to dataframe
            current_df = pd.concat([current_df, pd.DataFrame([mapped_row])], ignore_index=True)
        
        logger.info(f"Mapped {len(current_df)} Napa records to current schema")
        return current_df

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
            mapped_row['OBJECTID'] = idx + 1000  # Offset to avoid conflicts with Napa
            
            # Apply direct mappings
            for ridgecrest_field, current_field in self.ridgecrest_mapping.items():
                if current_field and ridgecrest_field in row:
                    mapped_row[current_field] = row[ridgecrest_field]
            
            # Handle Notes field - combine description with unmapped fields
            notes_parts = []
            if 'description' in row and pd.notna(row['description']):
                notes_parts.append(str(row['description']))
            
            # Add unmapped field info to notes
            unmapped_notes = self.create_notes_field(row, ridgecrest_unmapped, self.ridgecrest_mapping)
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

    def consolidate_datasets(self, napa_current_df, ridgecrest_current_df):
        """Consolidate mapped datasets into single current schema dataset"""
        logger.info("Consolidating datasets...")
        
        # Combine datasets
        consolidated_df = pd.concat([napa_current_df, ridgecrest_current_df], ignore_index=True)
        
        # Reset OBJECTID to be sequential
        consolidated_df['OBJECTID'] = range(1, len(consolidated_df) + 1)
        
        # Fill NaN values appropriately
        for col in consolidated_df.columns:
            if col in ['Notes', 'Vector_Offset_Feature_Notes', 'Slip_Offset_Feature_Notes']:
                consolidated_df[col] = consolidated_df[col].fillna('')
            elif consolidated_df[col].dtype == 'object':
                consolidated_df[col] = consolidated_df[col].fillna('')
            else:
                consolidated_df[col] = consolidated_df[col].fillna(np.nan)
        
        logger.info(f"Consolidated dataset: {len(consolidated_df)} total records")
        return consolidated_df

    def generate_migration_report(self, napa_df, ridgecrest_df, consolidated_df):
        """Generate report on data migration"""
        report = []
        report.append("EARTHQUAKE DATA MIGRATION REPORT")
        report.append("=" * 50)
        report.append(f"Migration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("INPUT DATASETS:")
        report.append(f"  Napa (2014): {len(napa_df)} records, {len(napa_df.columns)} fields")
        report.append(f"  Ridgecrest (2019): {len(ridgecrest_df)} records, {len(ridgecrest_df.columns)} fields")
        report.append("")
        
        report.append("OUTPUT DATASET:")
        report.append(f"  Consolidated: {len(consolidated_df)} records, {len(consolidated_df.columns)} fields")
        report.append("")
        
        report.append("CRITICAL ISSUES IDENTIFIED:")
        report.append("  1. LOCATION DATA MISSING: Current schema lacks latitude/longitude fields")
        report.append("     - Ridgecrest coordinates stored in Notes field")
        report.append("     - Recommend adding coordinate fields to Current schema")
        report.append("")
        report.append("  2. DATA LOSS: Some detailed measurements lost due to schema differences")
        report.append("     - Min/max ranges for some measurements")
        report.append("     - Local fracture orientation data")
        report.append("     - Observational details (striations, gouge)")
        report.append("")
        
        report.append("FIELD MAPPING SUMMARY:")
        report.append(f"  Napa direct mappings: {sum(1 for v in self.napa_mapping.values() if v is not None)}")
        report.append(f"  Ridgecrest direct mappings: {sum(1 for v in self.ridgecrest_mapping.values() if v is not None)}")
        report.append("")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    # Initialize mapper
    mapper = EarthquakeDataMapper()
    
    # File paths (update these to your actual file locations)
    # napa_file = "napa_observations.csv" 
    napa_file = "C:/Users/rajuv/OneDrive/Desktop/Work/SCEC SOURCES Internship/SCEC/Mapping to Current/napa_observations.csv" 
    ridgecrest_file = "C:/Users/rajuv/OneDrive/Desktop/Work/SCEC SOURCES Internship/SCEC/Mapping to Current/ridgecrest_observations.csv"  
    
    try:
        # Load datasets
        logger.info("Starting earthquake data migration...")
        napa_df = mapper.load_napa_data(napa_file)
        ridgecrest_df = mapper.load_ridgecrest_data(ridgecrest_file)

        # DEBUG: Check what columns we actually have
        print("\nNAPA COLUMNS:")
        print(list(napa_df.columns)[:])  # First 10 columns
        
        print("\nRIDGECREST COLUMNS:")
        print(list(ridgecrest_df.columns)[:])  # First 10 columns
    
        # Check if they're identical
        if list(napa_df.columns) == list(ridgecrest_df.columns):
            print("\n⚠️  WARNING: Napa and Ridgecrest have identical column structures!")
            print("This suggests you might have the same dataset loaded twice.")
            
            if napa_df is None or ridgecrest_df is None:
                logger.error("Failed to load input datasets")
                return
            
        # Map to current schema
        napa_current = mapper.map_napa_to_current(napa_df)
        ridgecrest_current = mapper.map_ridgecrest_to_current(ridgecrest_df)
        
        # Consolidate
        consolidated = mapper.consolidate_datasets(napa_current, ridgecrest_current)
        
        # Save consolidated dataset
        output_file = f"consolidated_earthquake_observations_{datetime.now().strftime('%Y%m%d')}.csv"
        consolidated.to_csv(output_file, index=False)
        logger.info(f"Consolidated dataset saved to: {output_file}")
        
        # Generate and save report
        report = mapper.generate_migration_report(napa_df, ridgecrest_df, consolidated)
        report_file = f"migration_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Migration report saved to: {report_file}")
        
        print("\nMigration completed successfully!")
        print(f"Output files: {output_file}, {report_file}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")

if __name__ == "__main__":
    main()