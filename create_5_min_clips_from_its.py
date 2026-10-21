import os
import glob
import math
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# --- 1. Constants & Configuration ---
CLIP_DURATION_SEC = 300  # 5 minutes
TARGET_CLIPS = 60
ITS_DIRECTORY = "../../data/blena/lena_raw_its"
EXCEL_FILE = "../../data/blena/Bilingual_LENA.xlsx" 
MAP = {'FAN': 'femaleAdultWordCnt',
       'MAN': 'maleAdultWordCnt',
       '1st': 'Rec1Age',
       '2nd': 'Rec2Age'}

def parse_participant_info(filename):
    info = filename.split('_')
    return info[0].strip(), info[1].strip(), info[2].strip()

# --- 2. ITS Parsing Logic ---
def parse_duration(time_str):
    """
    LENA ITS times are typically in ISO 8601 duration format (e.g., 'PT1H2M3.5S' or 'PT15.2S').
    This is a simplified parser for standard LENA PT format.
    """
    time_str = time_str.replace("PT", "")
    hours = minutes = seconds = 0.0
    
    if "H" in time_str:
        h_split = time_str.split("H")
        hours = float(h_split[0])
        time_str = h_split[1]
    if "M" in time_str:
        m_split = time_str.split("M")
        minutes = float(m_split[0])
        time_str = m_split[1]
    if "S" in time_str:
        seconds = float(time_str.replace("S", ""))
        
    return (hours * 3600) + (minutes * 60) + seconds

def parse_its_files(directory):
    """
    Iterates through all .its files and chunks them into 5-minute clips.
    Extracts AWC, CVC, and CTC for each clip.
    """
    clip_data = []
    df_master = pd.read_excel(EXCEL_FILE)
    
    for filepath in glob.glob(os.path.join(directory, "*.its")):
        filename = os.path.basename(filepath)
        tree = ET.parse(filepath)
        root = tree.getroot()
        try:
            participant_id, lang, rec_id = parse_participant_info(filename)
            #print(f"Filename: {filename}, language: {lang}, recording: {rec_id}, participant: {participant_id}")
            age_str = df_master.loc[df_master['ExperimentCode'] == participant_id, MAP[rec_id]].values[0]
            if pd.isna(age_str):
                age_str = df_master.loc[df_master['ExperimentCode'] == participant_id, 'CurrentAge'].values[0]
           
            age = int(age_str.split(';')[0])
        except KeyError:
            print(f"Filename: {filename}, language: {lang}, recording: {rec_id}, participant: {participant_id}")

        

        # Dictionary to hold running totals for 5-minute bins
        # Key: bin_index (int), Value: {'awc': 0, 'cvc': 0, 'ctc': 0}
        bins = {}

        def get_bin(start_sec):
            bin_idx = int(start_sec // CLIP_DURATION_SEC)
            if bin_idx not in bins:
                bins[bin_idx] = {'AWC': 0, 'CVC': 0, 'CTC': 0}
            return bin_idx
        
        # Navigate standard LENA .its XML structure
        # ProcessingUnit contains Segments
        for pu in root.findall('.//ProcessingUnit'):
            for segment in pu.findall('.//Segment'):
                start_sec = parse_duration(segment.get('startTime', 'PT0S'))
                bin_idx = get_bin(start_sec)
                
                spkr = segment.get('spkr', '')
                
                # 1. Adult Word Count (AWC)
                if spkr in ['FAN', 'MAN']: # Female or Male Adult Near
                    awc_val = float(segment.get(MAP[spkr], 0))
                    bins[bin_idx]['AWC'] += awc_val
                
                # 2. Child Vocalization Count (CVC)
                if spkr == 'CHN': # Key Child
                    # Sometimes represented by childUttCnt or simply segment counts
                    cvc_val = int(segment.get('childUttCnt', 0)) 
                    bins[bin_idx]['CVC'] += cvc_val
                

            # 2. Parse Conversations for CTC (turn taking)
            for conv in pu.findall('.//Conversation'):
                start_sec = parse_duration(conv.get('startTime', 'PT0S'))
                bin_idx = get_bin(start_sec)
                
                # Check for 'turnTaking' or fallback to 'turnCount' just in case
                turn_val = conv.get('turnTaking', conv.get('turnCount', '0'))
                try:
                    bins[bin_idx]['CTC'] += int(turn_val)
                except ValueError:
                    pass
                    
        # Append all bins from this file to the master dataset
        for bin_idx, counts in bins.items():
            start_time = bin_idx * CLIP_DURATION_SEC
            end_time = start_time + CLIP_DURATION_SEC
            clip_data.append({
                'file': filename,
                'participant_id': participant_id,
                'start_sec': start_time,
                'end_sec': end_time,
                'language': lang,
                'age': age,
                'AWC': counts['AWC'],
                'CVC': counts['CVC'],
                'CTC': counts['CTC']
            })
            
    return pd.DataFrame(clip_data)


def select_diverse_lena_clips(df, total_samples=60, samples_per_lang=30, max_per_participant=3, target_per_bin=3, random_seed=42):
    """
    Bins LENA dataset features and selects a diverse subset of clips with:
        - Balanced language representation (30 Chinese, 30 Vietnamese)
        - 100% bin coverage for all unique AWC, CVC, CTC, and Age bins
        - A strict cap of max_per_participant clips per unique participant ID
    """
    df = df.copy()
    np.random.seed(random_seed)

    # --- Step 1: Binning the continuous feats ---
    df['awc_bin'] = (df['AWC'] // 50) * 50
    df['cvc_bin'] = (df['CVC'] // 10) * 10
    df['ctc_bin'] = (df['CTC'] // 4) * 4
    df['age_bin'] = (df['age'] // 2) * 2

    unique_participants = df['participant_id'].unique()
    num_participants = len(unique_participants)

    # --- Step 2: Compute max possible counts per bin in data ---
    # prevents code from chasing a target for bins that have < 3 total rows in dataset
    max_awc_possible = df['awc_bin'].value_counts().to_dict()
    max_cvc_possible = df['cvc_bin'].value_counts().to_dict()
    max_ctc_possible = df['ctc_bin'].value_counts().to_dict()
    max_age_possible = df['age_bin'].value_counts().to_dict()
    
    # Track selected counts per bin
    awc_bin_counts = {b: 0 for b in max_awc_possible}
    cvc_bin_counts = {b: 0 for b in max_cvc_possible}
    ctc_bin_counts = {b: 0 for b in max_ctc_possible}
    age_bin_counts = {b: 0 for b in max_age_possible}

    uncovered_participants = set(unique_participants)
    selected_indices = []
    participant_counts = {}
    lang_counts = {'CHN': 0, 'VNM': 0}

    # Working dataframe with index tracking
    df['orig_idx'] = df.index
    remaining_pool = df.copy()

    # --- Step 3: Greedy frequency targeted selection loop ---
    for _ in range(total_samples):
        if remaining_pool.empty:
            break
            
        best_row_idx = None
        best_score = -float('inf')
        
        # Evaluate each candidate in the remaining pool
        for idx, row in remaining_pool.iterrows():
            lang = row['language']
            pid = row['participant_id']
            
            # Constraint Check 1: Language quota (max 30 per language)
            if lang_counts.get(lang, 0) >= samples_per_lang:
                continue
                
            # Constraint Check 2: Participant clip cap (max 3 per participant)
            if participant_counts.get(pid, 0) >= max_per_participant:
                continue
                
            # Calculate priority score based on how far each bin is from target_per_bin
            score = 0

            # Priority 1: Heavy bonus for selecting a participant who hasn't been picked yet
            if pid in uncovered_participants:
                score += 1000.0


            # Priority 2: bin frequency target (min 3 per bin if possible)
            a_bin, c_bin, t_bin, ag_bin = row['awc_bin'], row['cvc_bin'], row['ctc_bin'], row['age_bin']

            # For each feature, reward selecting a row if its bin hasn't reached the target 
            # capped by what's actually available in the dataset
            target_a = min(target_per_bin, max_awc_possible[a_bin])
            if awc_bin_counts[a_bin] < target_a:
                score += (target_a - awc_bin_counts[a_bin]) * 2.0
                
            target_c = min(target_per_bin, max_cvc_possible[c_bin])
            if cvc_bin_counts[c_bin] < target_c:
                score += (target_c - cvc_bin_counts[c_bin]) * 2.0
                
            target_t = min(target_per_bin, max_ctc_possible[t_bin])
            if ctc_bin_counts[t_bin] < target_t:
                score += (target_t - ctc_bin_counts[t_bin]) * 2.0
                
            target_ag = min(target_per_bin, max_age_possible[ag_bin])
            if age_bin_counts[ag_bin] < target_ag:
                score += (target_ag - age_bin_counts[ag_bin]) * 2.0
            
            # Tie-breaker: if scores are equal, add a small random factor to ensure variety
            score += np.random.uniform(0, 0.1)
            
            if score > best_score:
                best_score = score
                best_row_idx = idx
                
        # Fallback if no score-positive rows are found (e.g. all target bins already have >= 3 samples)
        if best_row_idx is None or best_score <= 0.0:
            valid_pool = remaining_pool[
                (remaining_pool['language'].map(lang_counts) < samples_per_lang) &
                (remaining_pool['participant_id'].map(lambda p: participant_counts.get(p, 0)) < max_per_participant)
            ]
            if valid_pool.empty:
                break
            best_row_idx = valid_pool.index[0]
            
        # Select the best row
        chosen_row = remaining_pool.loc[best_row_idx]
        orig_idx = chosen_row['orig_idx']
        
        selected_indices.append(orig_idx)
        
        # Update trackers
        lang = chosen_row['language']
        lang_counts[lang] += 1
        pid = chosen_row['participant_id']
        participant_counts[pid] = participant_counts.get(pid, 0) + 1

        if pid in uncovered_participants:
            uncovered_participants.remove(pid)
        
        awc_bin_counts[chosen_row['awc_bin']] += 1
        cvc_bin_counts[chosen_row['cvc_bin']] += 1
        ctc_bin_counts[chosen_row['ctc_bin']] += 1
        age_bin_counts[chosen_row['age_bin']] += 1
        
        # Remove selected row from pool
        remaining_pool = remaining_pool.drop(best_row_idx)

    # --- Step 4: Finalize and validate ---
    final_df = df.loc[selected_indices].reset_index(drop=True)
    
    # Drop helper columns
    helper_cols = ['orig_idx']
    final_df = final_df.drop(columns=helper_cols)
    
    # Print Coverage Diagnostics
    print(f"Total Selected Clips: {len(final_df)}")
    print(f"Language Breakdown: {final_df['language'].value_counts().to_dict()}")
    print(f"Max clips per participant in selection: {max(participant_counts.values()) if participant_counts else 0}")
    
    return final_df


 

if __name__ == "__main__":
    print("Parsing .its files...")
    all_clips_df = parse_its_files(ITS_DIRECTORY)
    #all_clips_df.to_csv("all_5_min_lena_clips.csv", index=False)

    print("\nRange Coverage Summary:")
    print(all_clips_df[['AWC', 'CVC', 'CTC', 'age']].agg(['min', 'median', 'max', 'mean']).round(1))

    final_selection = select_diverse_lena_clips(all_clips_df)
    final_selection.to_csv("60_5_min_lena_clips.csv", index=False)
    print(f"AWC bin Counts: {final_selection['awc_bin'].value_counts()}")
    print(f"CVC bin Counts: {final_selection['cvc_bin'].value_counts()}")
    print(f"CTC bin Counts: {final_selection['ctc_bin'].value_counts()}")
    print(f"Age bin Counts: {final_selection['age_bin'].value_counts()}")
    print("Finished!")
    
    