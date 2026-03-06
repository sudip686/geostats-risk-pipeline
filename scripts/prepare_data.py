import pandas as pd
import os
import numpy as np

# Script to convert the provided specific files to the standard format required by the package

def prepare():
    # Paths (adjust as needed based on workspace)
    # The workspace seems to be the root.
    # Provided files are in the root directory.
    
    root_dir = os.getcwd()
    
    # Mapping for Collar
    # SLNO,BHID ,XCOLLAR (m),YCOLLAR (m),ZCOLLAR (m),END DEPTH (m),...
    try:
        collar_raw = pd.read_csv('collar.csv') # Assuming it's in current dir as per context, or adjust path
    except:
        # Try to read from known workspace path if running locally
        collar_raw = pd.read_csv(os.path.join(root_dir, 'collar.csv'))

    collar = pd.DataFrame()
    collar['hole_id'] = collar_raw['BHID ']
    collar['x'] = collar_raw['XCOLLAR (m)']
    collar['y'] = collar_raw['YCOLLAR (m)']
    collar['z'] = collar_raw['ZCOLLAR (m)']
    
    # Mapping for Survey
    survey = pd.DataFrame()
    survey_loaded = False
    
    try:
        survey_path = os.path.join(root_dir, 'survey.xlsx')
        if os.path.exists(survey_path):
            survey_raw = pd.read_excel(survey_path)
            if not survey_raw.empty and len(survey_raw.columns) >= 4:
                print(f"Loaded survey.xlsx: {survey_raw.shape}")
                survey['hole_id'] = survey_raw.iloc[:, 0]
                survey['depth'] = survey_raw.iloc[:, 1]
                survey['azimuth_deg'] = survey_raw.iloc[:, 2]
                survey['dip_deg'] = survey_raw.iloc[:, 3]
                survey_loaded = True
            else:
                print("Survey file is empty or invalid structure.")
        else:
            print("survey.xlsx not found.")
            
    except Exception as e:
        print(f"Could not read survey.xlsx: {e}")

    if not survey_loaded:
        print("Falling back to vertical holes based on collar data.")
        if 'END DEPTH (m)' in collar_raw.columns:
            # Create top (0) and bottom (end_depth) survey points
            hole_ids = []
            depths = []
            azimuths = []
            dips = []
            
            for _, row in collar_raw.iterrows():
                hid = row['BHID ']
                ed = row['END DEPTH (m)']
                if pd.isna(ed): ed = 100.0 # Default if missing
                
                # Collar point
                hole_ids.append(hid)
                depths.append(0.0)
                azimuths.append(0.0)
                dips.append(-90.0)
                
                # End point
                hole_ids.append(hid)
                depths.append(ed)
                azimuths.append(0.0)
                dips.append(-90.0)
                
            survey['hole_id'] = hole_ids
            survey['depth'] = depths
            survey['azimuth_deg'] = azimuths
            survey['dip_deg'] = dips
        else:
            print("Cannot generate survey: 'END DEPTH (m)' not in collar.")
            return
    
    # Mapping for Assay
    # TANGA_COMBINE_ASSAY_UPDATED.csv
    # BHID ,SAMPLE NO,FROM,TO,LENGTH OF SAMPLE (M),LITHO CODE ,... GRAPHITIC CARBON ...
    try:
        assay_raw = pd.read_csv('TANGA_COMBINE_ASSAY_UPDATED.csv')
    except:
        assay_raw = pd.read_csv(os.path.join(root_dir, 'TANGA_COMBINE_ASSAY_UPDATED.csv'))
        
    assay = pd.DataFrame()
    assay['hole_id'] = assay_raw['BHID ']
    assay['from_m'] = assay_raw['FROM']
    assay['to_m'] = assay_raw['TO']
    # Handle TGC column name. Usually 'GRAPHITIC CARBON'
    if 'GRAPHITIC CARBON' in assay_raw.columns:
        assay['tgc_pct'] = assay_raw['GRAPHITIC CARBON']
    else:
        # Try finding it
        col = [c for c in assay_raw.columns if 'GRAPHITIC' in c or 'TGC' in c][0]
        assay['tgc_pct'] = assay_raw[col]
        
    # Clean non-numeric TGC
    assay['tgc_pct'] = pd.to_numeric(assay['tgc_pct'], errors='coerce')
    assay = assay.dropna(subset=['tgc_pct'])

    # Mapping for Litho
    # litho.csv
    # BHID,FROM,TO,LITHO
    try:
        litho_raw = pd.read_csv('litho.csv')
    except:
        litho_raw = pd.read_csv(os.path.join(root_dir, 'litho.csv'))
        
    litho = pd.DataFrame()
    litho['hole_id'] = litho_raw['BHID']
    litho['from_m'] = litho_raw['FROM']
    litho['to_m'] = litho_raw['TO']
    litho['lith_code'] = litho_raw['LITHO']
    
    # Save to data/ directory
    if not os.path.exists('data'):
        os.makedirs('data')
        
    collar.to_csv('data/collar.csv', index=False)
    survey.to_csv('data/survey.csv', index=False)
    assay.to_csv('data/assay.csv', index=False)
    litho.to_csv('data/lithology.csv', index=False)
    
    print("Data prepared in data/ directory.")

if __name__ == "__main__":
    prepare()
