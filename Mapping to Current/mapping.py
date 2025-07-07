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
        self.napa_mapping = {
            'stnid': 'Station_ID', 
            'intid': None, 
            'observer': 'Creator',  
            'obs_date': 'Date_of_Movement',  
            'origin': 'Feature_Origin',  
            'latitude': None,  
            'longitude': None,  
            'orig_lat': None, 
            'orig_lon': None, 
            'description': 'Notes',
            'citation': None, 
            'photo': None, 
            'fault_azimuth': 'Local_Fault_Azimuth_Degrees',
            'ss_displacement': 'Horizontal_Separation_cm',
            'ss_sense': 'Slip_Sense',
            'ext_offset': 'Heave_cm', 
            'comp_offset': None, 
            'vert_offset': 'Vertical_Separation_cm',
            'upthrown_side': 'Scarp_Facing_Direction',
            'observed_feature': None, 
            'trace': None, 
        }
        
        self.ridgecrest_mapping = {
            'intid' : None,  # Let OBJECTID auto-generate
            'origid' : 'Station_ID',
            'observer' : 'Creator',
            'obs_affiliation' : None,
            'team_id' : None,
            'team' : None,
            'obs_position' : None,
            'obs_date' : 'CreationDate',
            'origin' : 'Feature_Origin',
            'source' : None,
            'citation' : None, 
            'description' : 'Notes',  # FIXED: Was None, now maps to Notes
            'fault_az_min' : None,
            'fault_az_pref' : 'Local_Fault_Azimuth_Degrees',
            'fault_az_max' : None,
            'fault_dip_min' : None,
            'fault_dip_pref' : 'Local_Fault_Dip',
            'fault_dip_max' : None,
            'local_frac_az_min' : None,
            'local_frac_az_pref' : None,
            'local_frac_az_max' : None,
            'rup_width_min' : 'Rupture_Width_Min_m',
            'rupture_width_pref' : 'Rupture_Width_m',
            'rup_width_max' : 'Rupture_Width_Max_m',
            'fault_expression' : 'Rupture_Expression',
            'scarp_facing_direction' : 'Scarp_Facing_Direction',
            'striations_observed' : None,
            'gouge_observed' : None,
            'sense' : 'Slip_Sense',
            'observed_feature' : None,
            'feature_type' : None,
            'vector_length_min' : 'Net_Slip_Min_cm',
            'vector_length_pref' : 'Net_Slip_Preferred_cm',
            'vector_length_max' : 'Net_Slip_Max_cm',
            'vect_plunge_min' : None,
            'vect_plunge_pref' : 'Plunge',
            'vect_plunge_max' : None,
            'vect_az_min' : None,
            'vect_az_pref' : 'VM_Slip_Azimuth',
            'vect_az_max' : None,
            'aperture_min' : None,
            'aperture_pref' : None,
            'aperture_max' : None,
            'horiz_offset_min' : 'Horizontal_Separation_Min_cm',
            'horiz_offset_pref' : 'Horizontal_Separation_cm',
            'horiz_offset_max' : 'Horizontal_Separation_Max_cm',
            'horiz_slip_type' : 'Fault_Slip_Measurement_Type',
            'horiz_az_min' : None,
            'horiz_az_pref' : 'Slip_Azimuth',
            'horiz_az_max' : None,
            'vert_offset_min' : 'Vertical_Separation_Min_cm',
            'vert_offset_pref' : 'Vertical_Separation_cm',
            'vert_offset_max' : 'Vertical_Separation_Max_cm',
            'vert_slip_type' : None,
            'heave_type' : None,
            'heave_min' : 'Heave_min_cm',
            'heave_pref' : 'Heave_cm',
            'heave_max' : 'Heave_max_cm',
            'latitude' : None,
            'longitude' : None,
            'orig_lat' : None,
            'orig_lon' : None,
            'note' : 'Vector_Offset_Feature_Notes',  # FIXED: Was 'Notes', now correct
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
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        
                        # If only 1 column, try tab-separated
                        if len(df.columns) == 1:
                            df = pd.read_csv(file_path, sep='\t', encoding=encoding)
                        
                        logger.info(f"Successfully loaded Napa data with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        logger.warning(f"Failed with {encoding}: {e}")
                        continue
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
            
            # Generate OBJECTID (auto-incrementing, offset to avoid conflicts with Napa)
            mapped_row['OBJECTID'] = idx + 1000
            
            # Apply direct mappings
            for ridgecrest_field, current_field in self.ridgecrest_mapping.items():
                if current_field and ridgecrest_field in row:
                    mapped_row[current_field] = row[ridgecrest_field]
            
            # Handle Notes field - combine description with unmapped fields
            notes_parts = []
            
            # Get the description from direct mapping (now that description maps to Notes)
            existing_notes = mapped_row.get('Notes', '')
            if existing_notes:
                notes_parts.append(str(existing_notes))
            
            # Add unmapped field info to notes
            unmapped_notes = self.create_notes_field(row, ridgecrest_unmapped, self.ridgecrest_mapping)
            if unmapped_notes:
                notes_parts.append(f"Additional: {unmapped_notes}")
            
            # Add location data to notes since it's missing from current schema
            location_parts = []
            for coord_field in ['latitude', 'longitude', 'orig_lat', 'orig_lon']:
                if coord_field in row and pd.notna(row[coord_field]):
                    location_parts.append(f"{coord_field}: {row[coord_field]}")
            
            if location_parts:
                notes_parts.append(f"Coordinates: {'; '.join(location_parts)}")
            
            # Update the Notes field with combined content
            mapped_row['Notes'] = "; ".join(notes_parts) if notes_parts else ""
            
            # Set default values for system fields
            mapped_row['CreationDate'] = datetime.now()
            mapped_row['EditDate'] = datetime.now() 
            mapped_row['Editor'] = 'Ridgecrest_Migration'
            
            # Add dataset source identifier
            if mapped_row['Notes']:
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
        report.append("     - All coordinates stored in Notes field")
        report.append("     - Recommend adding coordinate fields to Current schema")
        report.append("")
        report.append("  2. DATA PRESERVATION: Unmapped fields stored in Notes")
        report.append("     - Observer affiliations, team info, observational details")
        report.append("     - Min/max measurement ranges")
        report.append("     - Local fracture orientation data")
        report.append("")
        
        report.append("FIELD MAPPING SUMMARY:")
        report.append(f"  Napa direct mappings: {sum(1 for v in self.napa_mapping.values() if v is not None)}")
        report.append(f"  Ridgecrest direct mappings: {sum(1 for v in self.ridgecrest_mapping.values() if v is not None)}")
        report.append("")
        
        report.append("MAPPING CORRECTIONS APPLIED:")
        report.append("  - Fixed: Ridgecrest description → Notes (was unmapped)")
        report.append("  - Fixed: Ridgecrest note → Vector_Offset_Feature_Notes (was Notes)")
        report.append("  - Fixed: Ridgecrest OBJECTID auto-generation (removed intid mapping)")
        report.append("  - Enhanced: Ridgecrest location data preservation in Notes")
        report.append("")
        report.append("NAPA MAPPING: No changes - kept original mapping intact")
        report.append("")
        
        return "\n".join(report)

def main():
    """Main execution function"""
    # Initialize mapper
    mapper = EarthquakeDataMapper()
    
    # File paths (update these to your actual file locations)
    napa_file = "C:/Users/rajuv/OneDrive/Desktop/Work/SCEC SOURCES Internship/SCEC/Mapping to Current/napa_observations.csv" 
    ridgecrest_file = "C:/Users/rajuv/OneDrive/Desktop/Work/SCEC SOURCES Internship/SCEC/Mapping to Current/ridgecrest_observations.csv"  
    
    try:
        # Load datasets
        logger.info("Starting earthquake data migration...")
        napa_df = mapper.load_napa_data(napa_file)
        ridgecrest_df = mapper.load_ridgecrest_data(ridgecrest_file)
        
        # Check if loading failed
        if napa_df is None and ridgecrest_df is None:
            logger.error("Failed to load both datasets")
            return
        elif napa_df is None:
            logger.warning("Failed to load Napa data - proceeding with Ridgecrest only")
        elif ridgecrest_df is None:
            logger.warning("Failed to load Ridgecrest data - proceeding with Napa only")
        
        # Map to current schema
        datasets_to_consolidate = []
        
        if napa_df is not None:
            napa_current = mapper.map_napa_to_current(napa_df)
            datasets_to_consolidate.append(napa_current)
        
        if ridgecrest_df is not None:
            ridgecrest_current = mapper.map_ridgecrest_to_current(ridgecrest_df)
            datasets_to_consolidate.append(ridgecrest_current)
        
        # Consolidate available datasets
        if len(datasets_to_consolidate) == 2:
            consolidated = mapper.consolidate_datasets(datasets_to_consolidate[0], datasets_to_consolidate[1])
        elif len(datasets_to_consolidate) == 1:
            consolidated = datasets_to_consolidate[0]
            # Still need to finalize single dataset
            for col in consolidated.columns:
                if col in ['Notes', 'Vector_Offset_Feature_Notes', 'Slip_Offset_Feature_Notes']:
                    consolidated[col] = consolidated[col].fillna('')
                elif consolidated[col].dtype == 'object':
                    consolidated[col] = consolidated[col].fillna('')
                else:
                    consolidated[col] = consolidated[col].fillna(np.nan)
        else:
            logger.error("No datasets available for processing")
            return
        
        # Save consolidated dataset
        output_file = f"consolidated_earthquake_observations_{datetime.now().strftime('%Y%m%d')}.csv"
        consolidated.to_csv(output_file, index=False)
        logger.info(f"Consolidated dataset saved to: {output_file}")
        
        # Generate and save report
        if napa_df is not None and ridgecrest_df is not None:
            report = mapper.generate_migration_report(napa_df, ridgecrest_df, consolidated)
        else:
            # Generate simplified report for single dataset
            dataset_name = "Napa" if napa_df is not None else "Ridgecrest"
            dataset_df = napa_df if napa_df is not None else ridgecrest_df
            report = f"EARTHQUAKE DATA MIGRATION REPORT\n"
            report += f"{'=' * 50}\n"
            report += f"Migration Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"INPUT DATASET:\n"
            report += f"  {dataset_name}: {len(dataset_df)} records, {len(dataset_df.columns)} fields\n\n"
            report += f"OUTPUT DATASET:\n"
            report += f"  Mapped: {len(consolidated)} records, {len(consolidated.columns)} fields\n"
        
        report_file = f"migration_report_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Migration report saved to: {report_file}")
        
        print("\nMigration completed successfully!")
        print(f"Output files: {output_file}, {report_file}")
        print(f"Records processed: {len(consolidated)}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()