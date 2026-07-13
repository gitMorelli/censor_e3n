#!/usr/bin/env python3
import argparse
import logging
import os
from time import perf_counter
from pathlib import Path
import sys
import shutil
import numpy as np 
import gc
import pandas as pd
import warnings
import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta


logging.basicConfig( 
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REF_PDF_PATH = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\ref_pdf_Qx"

OUTPUT_PATH = "//vms-e34n-databr/2025-handwriting\\data\\csv_files"

E3N_PATH = "//vms-e34n-databr/2025-handwriting\\E3N\\TABLES" #d01_20260216_handwriting
E3N_COVARIATES_PATH = os.path.join(E3N_PATH, "d01_20260216_handwriting.csv")
E3N_REMPLI_PATH = os.path.join(E3N_PATH, "indic_questq1aq13_20260305.csv")

CENSORED_PATH = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\censored_files"

#EXTRACTED_PATH = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\extracted_data\\grids\\analysis\\final_aggregated_data.csv"
EXTRACTED_PATH = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\extracted_data\\final_aggregated_data.csv"


def update_progress_file():
    source = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\extracted_data\\grids\\ref_pdf"
    progress = pd.read_csv(os.path.join(source,"progress.csv"),encoding='cp1252')
    #print the number of unique ids in the progress file
    unique_ids = progress['id'].unique()
    #print(progress.head(10))
    questionnaires = [str(i) for i in range(1,14)]
    source_path = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\censored_files"
    all_unique_ids = set()
    for q in questionnaires:
        ids_path = os.path.join(source_path, f"Q{q}\\run_data_q{q}\\csv_results_aggregated\\combined_success_ids.csv")
        df = pd.read_csv(ids_path)
        #get unique ids from the column e3n_id_hand and select 10 randomly
        unique_entries = pd.Series(df["e3n_id_hand"].unique())
        all_unique_ids.update(unique_entries.tolist())
        del df, unique_entries
    #check which ids in all_unique_ids are not in the progress file unique ids
    missing_ids = all_unique_ids.difference(unique_ids)
    print(f"Number of unique ids in the progress file: {len(unique_ids)}")
    print(f"Number of unique ids in the combined success ids files: {len(all_unique_ids)}")
    print(f"Number of ids in the combined success ids files that are missing in the progress file: {len(missing_ids)}")
    #add the missing ids to the progress file in the id column with the status processed=False of the processed column
    if len(missing_ids) > 0:
        new_rows = pd.DataFrame({"id": list(missing_ids), "processed": False})
        progress = pd.concat([progress, new_rows], ignore_index=True)
        progress.to_csv(os.path.join(source,"progress.csv"), index=False, encoding='cp1252')
        print(f"Added {len(missing_ids)} missing ids to the progress file.")
    #print the number of unique ids in the progress file after the update
    unique_ids = progress['id'].unique()
    print(f"Number of unique ids in the progress file after the update: {len(unique_ids)}")
    #print the number of ids with processed=True and processed=False
    print(f"Number of ids with processed=True: {progress[progress['processed']==True].shape[0]}")
    print(f"Number of ids with processed=False: {progress[progress['processed']==False].shape[0]}")
    #save the updated progress file
    progress.to_csv(os.path.join("//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$","progress.csv"), index=False, encoding='cp1252')

def check_combined_success_ids():
    print("Check progress file ..")
    source = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\\extracted_data\\grids\\ref_pdf"
    source = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\test_combined_success"
    progress = pd.read_csv(os.path.join(source,"progress.csv"),encoding='cp1252')
    ids_to_check = ['A7X4D3P1']
    #print the number of unique ids in the progress file
    unique_ids = progress['id'].unique()
    print(f"Number of unique ids in the progress file: {len(unique_ids)}")
    for id in ids_to_check:
        id_progress = progress[progress['id']==id]
        if id_progress.empty:
            print(f"No progress found for id {id}")
        else:
            print(id_progress)
    
    print("Now checking the combined success ids files..")
    questionnaires = [str(i) for i in range(1,14)]
    source_path = "//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\censored_files"
    all_unique_ids = set()
    for q in questionnaires:
        ids_path = os.path.join(source_path, f"Q{q}\\run_data_q{q}\\csv_results_aggregated\\combined_success_ids.csv")
        df = pd.read_csv(ids_path)
        #get unique ids from the column e3n_id_hand and select 10 randomly
        unique_entries = pd.Series(df["e3n_id_hand"].unique())
        all_unique_ids.update(unique_entries.tolist())
        del df, unique_entries
    #print the number of unique ids in the combined success ids files
    print(f"Number of unique ids in the combined success ids files: {len(all_unique_ids)}")
    for id in ids_to_check:
        if id in all_unique_ids:
            print(f"Id {id} is present in the combined success ids.")
        else:
            print(f"Id {id} is NOT present in the combined success ids.")
    
    print("Now checking from individual files..")
    source_path = f"//smb-recherche-s1.prod-powerscale.intra.igr.fr/E34N_HANDWRITING$\\test_combined_success"
    all_unique_ids = set()
    for q in questionnaires:
        ids_path = os.path.join(source_path, f"combined_success_ids_q{q}.csv")
        df = pd.read_csv(ids_path)
        #get unique ids from the column id and select 10 randomly
        unique_entries = pd.Series(df["e3n_id_hand"].unique())
        all_unique_ids.update(unique_entries.tolist())
        del df, unique_entries
    #print the number of unique ids in the combined success ids files
    print(f"Number of unique ids in the combined success ids files: {len(all_unique_ids)}")
    for id in ids_to_check:
        if id in all_unique_ids:
            print(f"Id {id} is present in the individual success ids.")
        else:
            print(f"Id {id} is NOT present in the individual success ids.")
    
    return

def clean_extracted_df(df):
    ''' extracted df ha un valore per ogni coppia id,qx anche se qx non esiste per quell'id -> tutte le colonne sono nan ->
    questa funzione rimuove queste righe inutili'''
    extracted_cols = ['X',  'X_random',   'hand', 'hand_partial_full',  'hand_sentences_full',  'number',  'number_random']
    df_clean = df.dropna(subset=extracted_cols, how='all')
    return df_clean

def explore_covariates():
    final_table = pd.read_csv(E3N_COVARIATES_PATH,encoding='cp1252')
    columns_to_explore = ['an_park_dg_final1','diag_park_final1_quest','age_suivi_diag1_quest','date_suivi_diag1_quest','date_lastquest','dernier_q_rempli','dateq1']
    filtered_final_table = final_table[columns_to_explore]
    print(filtered_final_table.head(10))
    print(filtered_final_table.describe())
    return 
def explore_extraction_data():
    extraction_data = pd.read_csv(EXTRACTED_PATH)
    print(extraction_data.head(10))
    #print(extraction_data.describe())

def assign_splits(df, n_train, n_val, n_test, split_col="split", 
                  shuffle=True, seed=42):
    """
    Assign train/val/test labels to dataframe rows in a `split` column.

    Parameters
    ----------
    df : pd.DataFrame
    n_train, n_val, n_test : int
        Number of rows for each split. Must sum to <= len(df);
        leftover rows get NaN.
    split_col : str
        Name of the column to write the split labels into.
    shuffle : bool
        Whether to shuffle rows before assigning splits.
    seed : int or None
        Random seed for reproducibility.
    """
    total = n_train + n_val + n_test
    if total > len(df):
        raise ValueError(f"Requested {total} rows but dataframe has {len(df)}")

    df = df.copy()

    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(df))
    else:
        order = np.arange(len(df))

    labels = np.full(len(df), None, dtype=object)
    labels[order[:n_train]] = "train"
    labels[order[n_train:n_train + n_val]] = "val"
    labels[order[n_train + n_val:total]] = "test"

    df[split_col] = labels
    return df

def prepare_for_matching(df,split_before = False, n_test=200, f_val=0.1):
    '''n_test is the number of PD subjects to put in the test set, f_val the fraction of remaining subjects to put in validation'''
    from pandas.api.types import is_datetime64_any_dtype as is_date

    #columns necessary for matching
    columns_of_interest = ['ident_projet','diag_park_final1_quest', 'date_suivi_diag1_quest','birth_month', 'birth_year', 'aexclure']
    columns_of_interest.extend([f'dateq{i}' for i in range(1,14)])
    columns_of_interest.extend([f'q_{i}_avail' for i in range(1,14)])
    columns_of_interest.extend([f'q_{i}_grid_file_avail' for i in range(1,14)]) #sono i dati che effettivamente ho -> devo considerare questi
    #other columns I want to keep in the data
    columns_of_interest.extend(['age_suivi_diag1_quest','cas_prev_dg1','lateralite','etudegp','actprofq2',
                                'profq2','rempli_seulq11','rempli_seulq12'])
    columns_of_interest.extend([f'q_{q}_num_X' for q in range(1,14)]) 
    columns_of_interest.extend([f'q_{q}_num_text' for q in range(1,14)])
    columns_of_interest.extend([f'q_{q}_num_digit' for q in range(1,14)])

    df_filtered = df[columns_of_interest]

    #apply the following transformation to 'profq2': if starts with "ENE"->1 elif ""->9 else 0
    df_filtered['profq2'] = df_filtered['profq2'].apply(lambda x: 1 if isinstance(x, str) and x.startswith("ENE") else (9 if pd.isna(x) else 0)).astype('int8')
    #print the number of values for each category in profq2
    print(f"Value counts for profq2: {df_filtered['profq2'].value_counts()}")
    #if etudegp is nan substitute with 9
    df_filtered['etudegp'] = df_filtered['etudegp'].fillna(9).astype('int8')
    print(f"Value counts for etudegp: {df_filtered['etudegp'].value_counts()}")
    #if lateralite is nan substitute with 9
    df_filtered['lateralite'] = df_filtered['lateralite'].fillna(9).astype('int8')
    print(f"Value counts for lateralite: {df_filtered['lateralite'].value_counts()}")
    #if rempli_seulq11 is nan substitute with 9
    df_filtered['rempli_seulq11'] = df_filtered['rempli_seulq11'].fillna(9).astype('int8')
    print(f"Value counts for rempli_seulq11: {df_filtered['rempli_seulq11'].value_counts()}")
    #select the ids for which rempli_seulq11 is not 9 and q_11_avail is 1 and check the value counts for rempli_seulq11 in this subset
    subset = df_filtered[(df_filtered['rempli_seulq11'] != 9) & (df_filtered['q_11_avail'] == 1)]
    print(f"Number of rows in the subset with q_11_avail==1 and rempli_seulq11!=9: {len(subset)}")
    print(f"Fractions for rempli_seulq11 in the subset with q_11_avail==1 and rempli_seulq11!=9: {subset['rempli_seulq11'].value_counts()/len(subset)}")
    #filter for the ids for which diag_park_final1_quest==1 and check the value counts for rempli_seulq11 in this subset
    subset_diag = df_filtered[df_filtered['diag_park_final1_quest']==1]
    print(f"Number of rows in the subset with diag_park_final1_quest==1: {len(subset_diag)}")
    print(f"Fractions for rempli_seulq11 in the subset with diag_park_final1_quest==1: {subset_diag['rempli_seulq11'].value_counts()/len(subset_diag)}")
    #if actprofq2 is nan substitute with 9
    df_filtered['actprofq2'] = df_filtered['actprofq2'].fillna(9).astype('int8')
    print(f"Value counts for actprofq2: {df_filtered['actprofq2'].value_counts()}")


    #count the number of values with cas_prev_dg1==1
    print(f"Number of values with cas_prev_dg1==1: {df_filtered[df_filtered['cas_prev_dg1']==1].shape[0]}") #cas_prev_dg1 is not in the columns of interest, I assume you meant diag_park_final1_quest
    #count the number of values with aexclure==1
    print(f"Number of values with aexclure==1: {df_filtered[df_filtered['aexclure']==1].shape[0]}")
    # and the number of cases with both the aexclure==1 and diag_park_final1_quest==1
    print(f"Number of values with aexclure==1 and diag_park_final1_quest==1: {df_filtered[(df_filtered['aexclure']==1) & (df_filtered['diag_park_final1_quest']==1)].shape[0]}")

    
    #select only the lines with aexclure!=1
    df_filtered = df_filtered[df_filtered['aexclure']!=1]

    #count nan values for each column
    print(df_filtered.isna().sum())
    #show the lines with nan values in the 'date_suivi_diag1_quest' column
    print(df_filtered[df_filtered['date_suivi_diag1_quest'].isna()])

    #df_filtered['date_suivi_diag1_quest_convert'] = pd.to_datetime(df['date_suivi_diag1_quest'], format='%d%b%Y') #if you want to keep the original colun and check
    df_filtered['date_suivi_diag1_quest'] = pd.to_datetime(df['date_suivi_diag1_quest'], format='%d%b%Y')
    for col in [f'dateq{i}' for i in range(1,14)]:
        df_filtered[col] = pd.to_datetime(df[col], format='%d%b%Y')
    df_filtered['birth_date'] = pd.to_datetime(pd.DataFrame({
        'year': df_filtered['birth_year'],
        'month': df_filtered['birth_month'],
        'day': 1  # We hardcode the 1st of the month here
    }))
    df_filtered['years_diff'] = (df_filtered['dateq1'] - df_filtered['birth_date']).dt.days / 365.25 #if one is NaN also the diff is NaN
    df_filtered['years_int'] = df_filtered['years_diff'].round(0).astype(int)
    #if you want to compute the actual years
    '''def calculate_years(row):
        start = row['start_date']
        end = row['end_date']
        
        # Simple subtraction of years
        years = end.year - start.year
        
        # Adjust if the month/day of the end date is before the start date
        if (end.month, end.day) < (start.month, start.day):
            years -= 1
        return years
    df['years_diff'] = df.apply(calculate_years, axis=1)'''


    print(df_filtered.head(10))

    print("dtypes: ", df_filtered.dtypes)

    if split_before:
        print("###### Splitting the data into train, val and test sets before matching ######")
        #select the rows with diag_park_final1_quest==1 and split them according to parameters
        df_cases = df_filtered[df_filtered['diag_park_final1_quest']==1]
        n_tot_cases = df_cases.shape[0]
        n_train_val_cases = n_tot_cases - n_test 
        n_val_cases = int(f_val * n_train_val_cases)
        n_train_cases = n_train_val_cases - n_val_cases
        df_cases = assign_splits(df_cases, n_train_cases, n_val_cases, n_test, split_col="split", shuffle=True, seed=42)

        #select the rows with diag_park_final1_quest==0 and split them according to parameters
        df_controls = df_filtered[df_filtered['diag_park_final1_quest']==0]
        n_tot_controls = df_controls.shape[0]
        n_test_controls = int(n_test/n_tot_cases * n_tot_controls) #i take the same fraction of controls as i did for cases for the test set (and other splits)
        n_train_val_controls = n_tot_controls - n_test_controls
        n_val_controls = int(f_val * n_train_val_controls)
        n_train_controls = n_train_val_controls - n_val_controls
        df_controls = assign_splits(df_controls, n_train_controls, n_val_controls, n_test_controls, split_col="split", shuffle=True, seed=42)

        df_final = pd.concat([df_cases, df_controls], ignore_index=True)

        df_final.to_csv(os.path.join(OUTPUT_PATH, "final_table_for_matching_splitted.csv"), index=False)

        return df_final
    else:
        #save the filtered dataframe with the new columns for the matching phase
        df_filtered.to_csv(os.path.join(OUTPUT_PATH, "final_table_for_matching.csv"), index=False)
        #df_filtered.to_parquet(os.path.join(OUTPUT_PATH, "final_table_for_matching.parquet"))

    return df_filtered


def matching(df_source, n_matches, time_of_entry_col="dateq1", time_of_exit_col="date_suivi_diag1_quest", id_col="ident_projet", 
             diag_col="diag_park_final1_quest", matching_cols=[], type_of_matching="exact", age_mode=0,filter_rempliseul=False,exclude_cases_rempliseul=False):
    
    # Work on a copy to avoid mutating the original dataframe
    df = df_source.copy()

    if filter_rempliseul:
        #remove controls that may have filled the questionnaires with the help of someone
        df[f'rempli_seulq11'] = df[f'rempli_seulq11'].fillna(9).astype('int8')
        df[f'rempli_seulq12'] = df[f'rempli_seulq12'].fillna(9).astype('int8')
        if exclude_cases_rempliseul:
            cond1 = (df["rempli_seulq11"] == 0)
            cond2 = (
                (df["rempli_seulq11"] == 9)
                & (df["rempli_seulq12"] == 0)
            )
        else:
            cond1 = (df["diag_park_final1_quest"] == 0) & (df["rempli_seulq11"] == 0)
            cond2 = (
                (df["diag_park_final1_quest"] == 0)
                & (df["rempli_seulq11"] == 9)
                & (df["rempli_seulq12"] == 0)
            )
        # 2. Count the number of rows removed by each condition separately
        removed_by_cond1 = cond1.sum()
        removed_by_cond2 = cond2.sum()
        exclude_mask = cond1 | cond2
        total_removed = exclude_mask.sum()
        # 4. Drop the rows by inverting the combined mask
        df = df[~exclude_mask]

        print(f"Rows excluded by Condition 1 on rempliseul: {removed_by_cond1}")
        print(f"Rows excluded by Condition 2 on rempliseul: {removed_by_cond2}")
        print(f"Total rows removed based on rempliseul: {total_removed}")
    
    # 1. Vectorized Datetime Conversions (Done once for the whole dataframe)
    df['birth_date'] = pd.to_datetime(df['birth_date'])
    df[time_of_exit_col] = pd.to_datetime(df[time_of_exit_col])
    df[time_of_entry_col] = pd.to_datetime(df[time_of_entry_col])
    
    q_date_cols = [f"dateq{i}" for i in range(1, 14)]
    for col in q_date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
    # 2. Vectorized Feature Calculations
    df['follow_up_time'] = (df[time_of_exit_col] - df[time_of_entry_col]) / pd.Timedelta(days=365.25)
    birth_date_oldest = df['birth_date'].min()
    print("Birth date of the oldest person in the cohort: ", birth_date_oldest)
    df['relative_age'] = ((df['birth_date'] - birth_date_oldest) / pd.Timedelta(days=365.25)).round().astype(int)

    # 3. Set Index for O(1) Lookups
    df_indexed = df.set_index(id_col, drop=False)
    
    # Isolate cases
    cases_df = df_indexed[df_indexed[diag_col] == 1].copy()
    case_ids = cases_df.index.tolist()
    
    # 4. Matrix Vectorization to pre-calculate 'last_q_filled' for all cases
    q_values = cases_df[q_date_cols].values
    exit_dates = cases_df[time_of_exit_col].values[:, None]
    is_nat = pd.isna(q_values)
    
    # A questionnaire date violates the window if it's NOT NaT and is >= exit date
    is_after_exit = (q_values >= exit_dates) & (~is_nat)
    has_violation = is_after_exit.any(axis=1)
    first_violation = is_after_exit.argmax(axis=1)
    cutoff = np.where(has_violation, first_violation, 13)
    
    # Create mask of allowed columns (index < cutoff and not NaT)
    col_indices = np.arange(13)
    allowed_mask = (col_indices[None, :] < cutoff[:, None]) & (~is_nat)
    
    # Identify the last True value from the right
    has_any_allowed = allowed_mask.any(axis=1)
    last_allowed_idx = 12 - np.argmax(allowed_mask[:, ::-1], axis=1)
    cases_df['last_q_filled'] = np.where(has_any_allowed, last_allowed_idx + 1, 0)

    # Trackers
    cases_with_no_questionnaires = []
    cases_with_no_matches = []
    matched_till_now = 0
    matched_results = []
    statistics_on_matches = {'id': [], 'remaining_subjects': [], 'remaining_cases': [],
    'remaining_after_age': [], 'remaining_after_matching_cols': [], 'remaining_exact': []}

    # 5. Main Matching Loop
    for ind, case in enumerate(case_ids):
        if ind % 100 == 0:
            print(f"Processing case {ind+1}/{len(case_ids)} with id {case}..")
            
        case_row = cases_df.loc[case]
        last_q = case_row['last_q_filled']
        
        if last_q == 0:
            cases_with_no_questionnaires.append(case)
            continue
            
        fu_time = case_row['follow_up_time']
        rel_age = case_row['relative_age']

        # Vectorized base cohort filtering
        mask = (df_indexed['follow_up_time'] >= fu_time) & (df_indexed.index != case)
        
        if age_mode == 0:
            mask &= (df_indexed['relative_age'] == rel_age)
        elif age_mode > 0:
            mask &= (df_indexed['relative_age'] >= rel_age - age_mode) & (df_indexed['relative_age'] <= rel_age + age_mode)
            
        df_filtered = df_indexed[mask]
        remaining_after_age = len(df_filtered)
            
        # Vectorized additional column matching
        if matching_cols:
            case_cols_values = case_row[matching_cols].values
            match_cols_mask = (df_filtered[matching_cols] == case_cols_values).all(axis=1)
            df_filtered = df_filtered[match_cols_mask]
        remaining_after_matching_cols = len(df_filtered)
        
        # Vectorized questionnaire pattern matching (removes inner column loop)
        questionnaires_to_match = [f"q_{i}_avail" for i in range(1, last_q + 1)]
        case_q_avail = case_row[questionnaires_to_match].values
        
        match_mask_exact = (df_filtered[questionnaires_to_match] == case_q_avail).all(axis=1)
        match_mask_at_least = (df_filtered[questionnaires_to_match] >= case_q_avail).all(axis=1)
        #remove from the match_mask_at_least the ones that are in match_mask_exact
        match_mask_at_least_only = match_mask_at_least & ~match_mask_exact
        n_remaining_exact = match_mask_exact.sum()
        if "exact" in type_of_matching:
            if type_of_matching == "exact":
                df_filtered = df_filtered[match_mask_exact]
            elif type_of_matching == "exact_w_fallback":
                #select all the ones that are exact match and if there are not enough matches select n-len(df_filtered_exact) 
                #at random from the ones that are at least match
                df_filtered_exact = df_filtered[match_mask_exact]
                if len(df_filtered_exact) >= n_matches:
                    df_filtered = df_filtered_exact
                else:
                    df_filtered_at_least_only = df_filtered[match_mask_at_least_only]
                    needed_from_at_least = n_matches - len(df_filtered_exact)
                    df_filtered_at_least_only_sampled = df_filtered_at_least_only.sample(n=min(needed_from_at_least, len(df_filtered_at_least_only)), replace=False)
                    df_filtered = pd.concat([df_filtered_exact, df_filtered_at_least_only_sampled])
        elif type_of_matching == "at_least":
            df_filtered = df_filtered[match_mask_at_least]

        # Statistics Gathering
        n_remaining_subj = len(df_filtered)
        n_remaining_cases = (df_filtered[diag_col] == 1).sum()
        
        statistics_on_matches['id'].append(case)
        statistics_on_matches['remaining_after_age'].append(remaining_after_age)
        statistics_on_matches['remaining_after_matching_cols'].append(remaining_after_matching_cols)
        statistics_on_matches['remaining_exact'].append(n_remaining_exact)
        statistics_on_matches['remaining_subjects'].append(n_remaining_subj)
        statistics_on_matches['remaining_cases'].append(n_remaining_cases)
        

        if n_remaining_subj > 0:
            matched_controls = df_filtered.index.to_series().sample(n=min(n_matches, n_remaining_subj), replace=False).tolist()
        else:
            cases_with_no_matches.append(case)
            matched_controls = []
            
        # Append Match Group Results
        all_ids = [case] + matched_controls
        selected = df_indexed.loc[all_ids].copy()
        selected['match_group'] = matched_till_now + 1
        matched_till_now += 1
        selected['case_control'] = (selected.index == case).astype(int)
        selected['last_avail_q'] = last_q
        selected['remaining_after_age'] = remaining_after_age
        selected['remaining_after_matching_cols'] = remaining_after_matching_cols
        selected['remaining_exact'] = n_remaining_exact
        selected['remaining_subjects'] = n_remaining_subj
        matched_results.append(selected)

    # 6. Final Data Assembly
    if len(matched_results) > 0:
        final_matched_df = pd.concat(matched_results).reset_index(drop=True)
    else:
        final_matched_df = pd.DataFrame()
        print("No matches found for any case.")
        
    print(f"Number of cases with no questionnaires: {len(cases_with_no_questionnaires)}")
    print(f"Number of cases with no matches: {len(cases_with_no_matches)}")
    
    statistics_on_matches_df = pd.DataFrame(statistics_on_matches)
    failed_cases_df = pd.DataFrame({
        'id': cases_with_no_questionnaires + cases_with_no_matches,
        'reason': ['no_questionnaires'] * len(cases_with_no_questionnaires) + ['no_matches'] * len(cases_with_no_matches)
    })

    return final_matched_df, statistics_on_matches_df, failed_cases_df


def select_matched_ids():
    #load the file with the matched ids and the distance computed for each of them
    matched_ids_path = os.path.join(OUTPUT_PATH, "matched_ids_R.csv")
    matched_ids_df = pd.read_csv(matched_ids_path)
    print(matched_ids_df.head(10))
    
    return 

def check_validity_final_table(final_table):

    questionnairres = [str(i) for i in range(1,14)]
    avail_ref_cols = [f"q_{q}_avail_ref" for q in questionnairres]
    avail_cols = [f"q_{q}_avail" for q in questionnairres]
    avail_grid_cols = [f"q_{q}_grid_file_avail" for q in questionnairres]
    avail_censor_cols = [f"q_{q}_avail_cens" for q in questionnairres]
    #for each avail_col compute the number of 1 and 0 values
    print("Avail cols")
    for col in avail_cols:
        print(f"Column {col} value counts:")
        print(final_table[col].value_counts())
    print("-----------------")
    #count the number of rows where avail_col is 1 and avail_ref_col is 0
    for avail_col, avail_ref_col in zip(avail_cols, avail_ref_cols):
        count = final_table[(final_table[avail_col] == 1) & (final_table[avail_ref_col] == 0)].shape[0]
        print(f"Number of rows where {avail_col} is 1 and {avail_ref_col} is 0: {count}")
    print("-----------------")
    print("Avail_censor cols")
    for avail_ref_col, avail_censor_col in zip(avail_ref_cols, avail_censor_cols):
        count = final_table[(final_table[avail_ref_col] == 1) & (final_table[avail_censor_col] == 0)].shape[0]
        print(f"Number of rows where {avail_ref_col} is 1 and {avail_censor_col} is 0: {count}")
    print("-----------------")
    print("Avail_grid cols")
    #count the number of rows where avail_col is 1 and avail_ref_col is 0
    for avail_ref_col, avail_grid_col in zip(avail_ref_cols, avail_grid_cols):
        count = final_table[(final_table[avail_ref_col] == 1) & (final_table[avail_grid_col] == 0)].shape[0]
        print(f"Number of rows where {avail_ref_col} is 1 and {avail_grid_col} is 0: {count}")
    print("-----------------")
    print("Avail_grid cols")
    #count the number of rows where avail_col is 1 and avail_ref_col is 0
    for avail_censor_col, avail_grid_col in zip(avail_censor_cols, avail_grid_cols):
        count = final_table[(final_table[avail_censor_col] == 1) & (final_table[avail_grid_col] == 0)].shape[0]
        print(f"Number of rows where {avail_censor_col} is 1 and {avail_grid_col} is 0: {count}")
        #print one example of an id that is 1 in avail_censor_col and 0 in avail_grid_col
        example_id = final_table[(final_table[avail_censor_col] == 1) & (final_table[avail_grid_col] == 0)]['ident_projet'].values
        if len(example_id) > 0:
            print(f"Example id where {avail_censor_col} is 1 and {avail_grid_col} is 0: {example_id[0]}")

def select_ids_for_handedness_model_single(df,seed=42):
    handedness_col = "lateralite"
    questionnaire_to_use = "5"
    availability_col = f"q_{questionnaire_to_use}_grid_file_avail"
    extraction_statistics_col = [f"q_{questionnaire_to_use}_num_X", f"q_{questionnaire_to_use}_num_text", f"q_{questionnaire_to_use}_num_digit"]
    test_pctg = 0.2
    val_pctg = 0.1
    print(f"Total number of rows in the dataframe: {df.shape[0]}")
    #print(df[handedness_col].head(20)) -> missing values are loaded as Nan
    #select only the rows with availability_col==1 and laterality not null
    df_filtered = df[(df[availability_col] == 1) & (df[handedness_col].notna())].copy()
    print(f"Number of rows with {availability_col}==1 and {handedness_col} not null: {df_filtered.shape[0]}")
    #keep only the columns ident_projet, laterality 
    df_filtered = df_filtered[['ident_projet', handedness_col] + extraction_statistics_col]
    #shuffle randomly the rows of the dataframe and reset the index
    df_filtered = df_filtered.sample(frac=1, random_state=seed).reset_index(drop=True)
    #split the dataframe into train, val and test
    n = df_filtered.shape[0]
    n_test = int(n * test_pctg)
    n_val = int((n-n_test) * val_pctg)
    print(f"Number of rows in the test set: {n_test}")
    print(f"Number of rows in the validation set: {n_val}")
    print(f"Number of rows in the train set: {n - n_test - n_val}")
    #create a column with the split (train, val, test)
    df_filtered['split'] = 'train'
    df_filtered.loc[:n_val, 'split'] = 'val'
    df_filtered.loc[n_val:n_val+n_test, 'split'] = 'test'
    return df_filtered

def select_ids_for_handedness_model(df,n_controls,seed=42):
    handedness_col = "lateralite"
    all_questionnaires_to_use = [str(i) for i in range(1,14)]
    availability_columns = [f"q_{q}_grid_file_avail" for q in all_questionnaires_to_use]
    df_filtered = df.copy()
    #print the value counts of the handedness column
    value_counts = df_filtered[handedness_col].value_counts(dropna=False)
    print(f"Value counts for {handedness_col} before filtering: {value_counts}")
    df_filtered = df_filtered[df_filtered[handedness_col].notna()]
    df_filtered['total_grid_file_avail'] = df_filtered[availability_columns].sum(axis=1)
    # Splits the data into 3 groups with roughly the same number of rows in each
    df_filtered['grid_file_category'] = pd.qcut(
        df_filtered['total_grid_file_avail'], 
        q=3, 
        labels=[1, 2, 3]
    )
    #keep only selected columns
    statistics_cols=[]
    for q in all_questionnaires_to_use:
        statistics_cols.extend([f"q_{q}_num_X", f"q_{q}_num_text", f"q_{q}_num_digit", f"q_{q}_num_sent"])
    df_filtered = df_filtered[['ident_projet', handedness_col, 'grid_file_category'] + statistics_cols + availability_columns]

    df_to_concat = []
    test_pctg = 0.1
    val_pctg = 0.1
    for category in [1, 2, 3]:
        print("Processing category ", category)
        #count the number of rows with handedness==1 in the category
        count_handedness_1 = df_filtered[(df_filtered['grid_file_category'] == category) & (df_filtered[handedness_col] == 1)].shape[0]
        #keep in the dataset n_controls*count_handedness_1 rows with handedness==0 in the same category, remove the others
        df_category = df_filtered[df_filtered['grid_file_category'] == category].copy()
        df_handedness_1 = df_category[df_category[handedness_col] == 1]
        df_handedness_0 = df_category[df_category[handedness_col] == 0]
        df_handedness_0_sampled = df_handedness_0.sample(n=min(n_controls*count_handedness_1, len(df_handedness_0)), replace=False, random_state=seed)
        df_category_filtered = pd.concat([df_handedness_1, df_handedness_0_sampled])
        #shuffle randomly the rows of the dataframe and reset the index
        df_category_filtered = df_category_filtered.sample(frac=1, random_state=seed).reset_index(drop=True)
        #split the dataframe into train, val and test
        n = df_category_filtered.shape[0]
        n_test = int(n * test_pctg)
        n_val = int((n-n_test) * val_pctg)
        print(f"Number of rows in the test set: {n_test}")
        print(f"Number of rows in the validation set: {n_val}")
        print(f"Number of rows in the train set: {n - n_test - n_val}")
        #create a column with the split (train, val, test)
        df_category_filtered['split'] = 'train'
        df_category_filtered.loc[:n_val, 'split'] = 'val'
        df_category_filtered.loc[n_val:n_val+n_test, 'split'] = 'test'
        df_to_concat.append(df_category_filtered)
        print("-"*50)
    df_filtered = pd.concat(df_to_concat).reset_index(drop=True)
    return df_filtered


def explore_rempliquest(df,folder_path):
    # Define your log file path (you can adjust this path as needed)
    log_file_path = os.path.join(folder_path, "rempli_seul_analysis.log")
    # A quick helper function to write messages to the specified log file
    def log_message(message):
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")
    with open(log_file_path, "w", encoding="utf-8") as f:
            f.write('Starting: \n')
    
    log_message(f"Initial number of rows in the dataframe: {df.shape[0]}")
    df = df[df['aexclure']!=1]
    log_message(f"Number of rows remained after excluding subj with aexclure!=1: {df.shape[0]}")
    log_message("-"*50)

    for q in range(11,13):
        log_message(f"Exploring questionnaire {q}..")
        df_filtered = df.copy()
        log_message("Removing the rows for which the questionnaire is not present..")
        df_filtered = df_filtered[df_filtered[f'q_{q}_avail'] == 1]

        log_message("Analysis on all ->")
        #if rempli_seulq11 is nan substitute with 9
        df_filtered[f'rempli_seulq{q}'] = df_filtered[f'rempli_seulq{q}'].fillna(9).astype('int8')
        value_counts = df_filtered[f'rempli_seulq{q}'].value_counts()
        log_message(f"Value counts for rempli_seulq11: {value_counts}")
        print(f"Value counts for rempli_seulq{q}: {value_counts}")
        missing_count = value_counts[9] if 9 in value_counts else 0
        available_count = value_counts.sum() - missing_count
        filled_alone_count = value_counts[1] 
        aided_count = value_counts[0]
        log_message(f"Total number of rows with q_{q}_avail==1: {len(df_filtered)}")
        log_message(f"Number of rows with missing value (9): {missing_count}")
        log_message(f"Number of rows with available value (0 and 1): {available_count}")
        log_message(f"Subjects that filled alone the Q: {filled_alone_count}; Fraction: {filled_alone_count/available_count:.2f}")
        log_message(f"Subjects that needed aid to fill the Q: {aided_count}; Fraction: {aided_count/available_count:.2f}")
        log_message("-"*25)

        log_message(f"Considering only PD cases ->")
        df_filtered = df_filtered[df_filtered['diag_park_final1_quest']==1]
        value_counts = df_filtered[f'rempli_seulq{q}'].value_counts()
        log_message(f"Value counts for rempli_seulq11: {value_counts}")
        print(f"Value counts for rempli_seulq{q}: {value_counts}")
        missing_count = value_counts[9] if 9 in value_counts else 0
        available_count = value_counts.sum() - missing_count
        filled_alone_count = value_counts[1] 
        aided_count = value_counts[0]
        log_message(f"Total number of rows with q_{q}_avail==1: {len(df_filtered)}")
        log_message(f"Number of rows with missing value (9): {missing_count}")
        log_message(f"Number of rows with available value (0 and 1): {available_count}")
        log_message(f"Subjects that filled alone the Q: {filled_alone_count}; Fraction: {filled_alone_count/available_count:.2f}")
        log_message(f"Subjects that needed aid to fill the Q: {aided_count}; Fraction: {aided_count/available_count:.2f}")
        log_message("="*50)


def explore_q_patterns(df,folder_path):
    # Define your log file path (you can adjust this path as needed)
    log_file_path = os.path.join(folder_path, "q_patterns_analysis.log")
    # A quick helper function to write messages to the specified log file
    def log_message(message):
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")
    with open(log_file_path, "w", encoding="utf-8") as f:
            f.write('Starting: \n')

    questionnairres = [str(i) for i in range(1,14)]
    for q in questionnairres:
        log_message(f"Exploring patterns for questionnaire {q}..")
        df_filtered = df.copy()
        #get the max-min for the dateq{q} column (first convert to datetime)
        df_filtered[f'dateq{q}'] = pd.to_datetime(df_filtered[f'dateq{q}'], format='%d%b%Y', errors='coerce')
        min_date = df_filtered[f'dateq{q}'].min()
        max_date = df_filtered[f'dateq{q}'].max()
        log_message(f"Date range for dateq{q}: {min_date} - {max_date}")
        log_message("-"*50)
    date_columns = [f'dateq{i}' for i in range(1,14)]
    #convert all dateq columns to datetime
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], format='%d%b%Y', errors='coerce')
    #sort the dataframe by dateq1 
    df_sorted = df.sort_values(by=f'dateq1')
    #select 5 random rows from the rows with dateq1 in the first quartile of the dateq1 distribution and 5 random rows from the rows with dateq1 in the last quartile of the dateq1 distribution
    q_first_date = df_sorted[f'dateq1'].quantile(0.25)
    q_last_date = df_sorted[f'dateq1'].quantile(0.75)
    df_first_quartile = df_sorted[df_sorted[f'dateq1'] <= q_first_date]
    df_last_quartile = df_sorted[df_sorted[f'dateq1'] >= q_last_date]
    random_row_first = df_first_quartile.sample(n=5, random_state=42)
    random_row_last = df_last_quartile.sample(n=5, random_state=42)
    first_date = df_sorted[f'dateq1'].min()
    random_row_first=random_row_first[date_columns]
    random_row_last=random_row_last[date_columns]
    #subtract the first date from the dateq columns to see how much time has passed since the first date for each of the random rows
    random_row_first[date_columns] = random_row_first[date_columns].subtract(first_date)/pd.Timedelta(days=365.25)
    random_row_last[date_columns] = random_row_last[date_columns].subtract(first_date)/pd.Timedelta(days=365.25)
    log_message("Random rows from the first quartile of dateq1:")
    log_message(random_row_first)
    log_message("Random rows from the last quartile of dateq1:")
    log_message(random_row_last)
        
def main():
 
    #output_dir = args.output_path 
    #os.makedirs(output_dir, exist_ok=True) # Create folder if it doesn't exist

    questionnairres = [str(i) for i in range(1,14)]
    variables_of_interest = ['diag_park_final1_quest','aexclure','cas_prev_dg1']
    
    if os.path.exists(E3N_COVARIATES_PATH):
        final_table = pd.read_csv(E3N_COVARIATES_PATH,encoding='cp1252')
        all_ids = set(final_table['ident_projet'].to_list())
        print(f"Number of unique ids in the final table: {len(all_ids)}")

        #load ids present in the file with the information on who filled which questionnaires
        rempli_data = pd.read_csv(os.path.join(E3N_REMPLI_PATH),encoding='cp1252')

        #load ids present in the file that collects all the succesfully processed ids for the extraction phase
        extraction_data = pd.read_csv(EXTRACTED_PATH)
        extraction_data = clean_extracted_df(extraction_data)
        #print(list(extraction_data.columns))
        #print(extraction_data.head(15))

        batch_new_cols = {}
        
        for q in questionnairres:
            print(f"Processing questionnaire Q{q}...")
            #load ids present in the ref_file
            ref_file_path = os.path.join(REF_PDF_PATH, f"ref_pdf_Q{q}.csv")
            raw_ref_data = pd.read_csv(ref_file_path)
            ref_ids = set(raw_ref_data['e3n_id_hand'].to_list())
            #set q_i_avail_ref to 1 if present, 0 if not
            if ref_ids.issubset(all_ids):
                batch_new_cols[f"q_{q}_avail_ref"] = np.where(
                    final_table['ident_projet'].isin(ref_ids), 1, 0
                ).astype('int8')
            else:
                print(f"Warning: Some ids in the ref file for questionnaire Q{q} are not present in the final table. This may lead to missing values in the q_{q}_avail_ref column.")

            #select ids from the rempli_data df for this questionnaire
            ids_with_q_filled = set(rempli_data[rempli_data[f'rempli_q{q}'] == 1]['ident_projet'].to_list())
            #set q_i_avail to 1 if present, 0 if not
            if ids_with_q_filled.issubset(all_ids):
                batch_new_cols[f"q_{q}_avail"] = np.where(
                    final_table['ident_projet'].isin(ids_with_q_filled), 1, 0
                ).astype('int8')
            else:
                print(f"Warning: Some ids in the rempli file for questionnaire Q{q} are not present in the final table. This may lead to missing values in the q_{q}_avail column.")


            #load ids present in the file that collects all the succesfully processed ids for the censoring
            censoring_file_path = os.path.join(CENSORED_PATH, f"Q{q}", f"run_data_q{q}", "csv_results_aggregated", "combined_success_ids.csv")
            censoring_data = pd.read_csv(censoring_file_path)
            ids_in_censored = set(censoring_data['e3n_id_hand'].to_list())
            #set q_i_avail_cens to 1 if present, 0 if not
            if ids_in_censored.issubset(all_ids):
                batch_new_cols[f"q_{q}_avail_cens"] = final_table['ident_projet'].isin(ids_in_censored).astype(int)
            else:
                print(f"Warning: Some ids in the censoring file for questionnaire Q{q} are not present in the final table. This may lead to missing values in the q_{q}_avail_censored column.")
            #set the column censoring warnings
            first_stage_ids = set(censoring_data[censoring_data["Warning_ordering"]=="First stage"]['e3n_id_hand'].to_list())
            col_data = np.where(
                final_table['ident_projet'].isin(first_stage_ids), 0, -1
            ).astype('int8')
            # 1
            second_stage_ids = set(censoring_data[censoring_data["Warning_ordering"]=="Second stage"]['e3n_id_hand'].to_list())
            mask = final_table['ident_projet'].isin(second_stage_ids)
            col_data[mask] = 1
            #2
            ocr_stage_ids = set(censoring_data[~censoring_data["Warning_ordering"].isin(['First stage', 'Second stage'])]['e3n_id_hand'])
            mask = final_table['ident_projet'].isin(ocr_stage_ids)
            col_data[mask] = 2
            #3
            ocr_with_warning_ids = set(censoring_data[censoring_data["Warning_ordering"].str.contains("OCR", na=False)]['e3n_id_hand'])
            mask = final_table['ident_projet'].isin(ocr_with_warning_ids)
            col_data[mask] = 3
            batch_new_cols[f"q_{q}_censoring_ord"] = col_data
            #set the column censoring warnings
            id_with_large_censoring = set(censoring_data[censoring_data["Warning_censoring"].str.contains("large", na=False)]['e3n_id_hand'])
            batch_new_cols[f"q_{q}_censoring_warn"] = np.where(
                final_table['ident_projet'].isin(id_with_large_censoring), 1, 0
            ).astype('int8')

            #select the ids that were successfully extracted for this questionnaire
            filtered_extraction_data = extraction_data[extraction_data['questionnaire']==f'q{q}']
            ids_with_q_extracted = set(filtered_extraction_data['id'].to_list())
            #print(len(ids_with_q_extracted))
            #ogni id presente nella tabella finale è anche in h5 -> ha un file grid associato
            batch_new_cols[f"q_{q}_grid_file_avail"] = np.where(
                final_table['ident_projet'].isin(ids_with_q_extracted), 1, 0
            ).astype('int8')

            #aggiungo i valori delle variabili scalari
            # 1. Select only the columns you need from the source
            #cols_to_pull = ['id','X', 'hand', 'hand_partial_full',  'hand_sentences_full',  'number']
            #new_cols = [f"q_{q}_num_X", f"q_{q}_num_text", f"q_{q}_num_part", f"q_{q}_num_sent", f"q_{q}_num_digit"]
            lookup = filtered_extraction_data.drop_duplicates('id').set_index('id')
            ident_series = final_table['ident_projet']
            # 2. Rename columns to include your "q" prefix before merging
            batch_new_cols[f"q_{q}_num_X"] = ident_series.map(lookup['X']).fillna(-1).astype('float32') #will be ordered like ident_projet
            batch_new_cols[f"q_{q}_num_text"] = ident_series.map(lookup['hand']).fillna(-1).astype('float32')
            batch_new_cols[f"q_{q}_num_part"] = ident_series.map(lookup['hand_partial_full']).fillna(-1).astype('float32')
            batch_new_cols[f"q_{q}_num_sent"] = ident_series.map(lookup['hand_sentences_full']).fillna(-1).astype('float32')
            batch_new_cols[f"q_{q}_num_digit"] = ident_series.map(lookup['number']).fillna(-1).astype('float32')

            #return 
    final_table = pd.concat([final_table, pd.DataFrame(batch_new_cols)], axis=1)
    print(final_table.head(10))
    del batch_new_cols
    gc.collect()

    return final_table

def prepare_table_for_training(filename,output_path,seed=42, split_before = False):
    val_pctg = 0.1
    test_pctg = 0.1

    df = pd.read_csv(filename, encoding='cp1252')
    group_col = 'match_group'
    warning_col = "rempliseul_warning"
    questionnairres = [str(i) for i in range(1,14)]

    #adding warning for rempliseul
    warn_cond1 = df["rempli_seulq11"] == 0
    warn_cond2 = (df["rempli_seulq11"] == 9) & (df["rempli_seulq12"] == 0)
    # 2. Combine the conditions
    warning_mask = warn_cond1 | warn_cond2
    # 3. Create the new column (True becomes 1, False becomes 0)
    df[warning_col] = warning_mask.astype(int)

    #get the value of the group_col for the rows with rempliseul_warning == 1 and case_control == 1
    groups_with_warning = df[(df[warning_col] == 1) & (df['case_control'] == 1)][group_col].unique()
    #remove all the rows with rempliseul_warning == 1
    print(f"Length of the dataframe before filtering for rempliseul_warning: {len(df)}") 
    df_filtered = df[df[warning_col] == 0].copy()
    print(f"Length of the dataframe after filtering for rempliseul_warning: {len(df_filtered)}")
    #remove rows with group_col in groups_with_warning
    df_filtered = df_filtered[~df_filtered[group_col].isin(groups_with_warning)].copy()
    print(f"Length of the dataframe after removing groups with warning: {len(df_filtered)}")

    #i have to split taking in account the case-control matching if i didn't split before
    if not split_before:
        #get the unique values of the group_col 
        unique_groups = df_filtered[group_col].unique()
        n = len(unique_groups)
        n_test = int(n * test_pctg)
        n_val = int((n-n_test) * val_pctg)
        n_train = n - n_test - n_val
        #shuffle the unique groups and split them into train, val and test
        np.random.seed(seed)
        np.random.shuffle(unique_groups)
        train_groups = unique_groups[:n_train]
        val_groups = unique_groups[n_train:n_train+n_val]
        test_groups = unique_groups[n_train+n_val:]
        #create a new column split and set it to train, val or test according to the group_col
        df_filtered['split'] = 'train'
        df_filtered.loc[df_filtered[group_col].isin(val_groups), 'split'] = 'val'
        df_filtered.loc[df_filtered[group_col].isin(test_groups), 'split'] = 'test'
        print(f"Number of ids in the train set: {(df_filtered['split'] == 'train').sum()}")
        print(f"Number of ids in the val set: {(df_filtered['split'] == 'val').sum()}")
        print(f"Number of ids in the test set: {(df_filtered['split'] == 'test').sum()}")

        #Forcing lines with the same id to be in the same split (if a subject is a case in one of the group that case should be in the same split of the group in which 
        #it is the case, else i can choose wathever split)
        # 1. Sort the DataFrame to bring the priority row to the top of each project group.
        # - case_control descending (False) puts 1 before 0.
        # - match_group ascending (True) puts the lowest number first.
        df_sorted = df_filtered.sort_values(
            by=['ident_projet', 'case_control', 'match_group'], 
            ascending=[True, False, True]
        )
        # 2. Extract the winning split for each id (the first row of each group)
        project_target_splits = df_sorted.groupby('ident_projet')['split'].first()
        # 3. Map these correct splits back to your original DataFrame
        df_filtered['split'] = df_filtered['ident_projet'].map(project_target_splits)
        print("After forcing lines with the same id to be in the same split:")
        print(f"Number of ids in the train set: {(df_filtered['split'] == 'train').sum()}")
        print(f"Number of ids in the val set: {(df_filtered['split'] == 'val').sum()}")
        print(f"Number of ids in the test set: {(df_filtered['split'] == 'test').sum()}")

    # 1. Get the maximum match_group 
    max_match_group = df_filtered['match_group'].max()
    # 2. Count the number of digits of the maximum match_group
    n_digits = len(str(max_match_group))
    # 3. Create the new id using fast, vectorized string operations
    df_filtered['unique_id'] = df_filtered['ident_projet'] + "_" + df_filtered['match_group'].astype(str).str.zfill(n_digits)

    print(df_filtered.columns)

    ######## ADD at_least LABEL + Prepare dateq data #########
    #chck if it was added with at_least or exact by comparint the pattern of rempli_questionnaire
    avail_cols=[f'q_{q}_avail' for q in questionnairres]
    grid_avail_cols = [f'q_{q}_grid_file_avail' for q in questionnairres]
    df_filtered['rempli_pattern'] = df_filtered[avail_cols].astype(str).agg(''.join, axis=1)
    df_filtered['grid_pattern'] = df_filtered[grid_avail_cols].astype(str).agg(''.join, axis=1)

    #Add case information to controls in the same group
    # 1. Extract the pattern of the case (case_control == 1) for each group
    date_columns = [f'dateq{i}' for i in range(1,14)]
    #convert the date_suivi_diag1_quest column to datetime
    df_filtered['date_suivi_diag1_quest'] = pd.to_datetime(df_filtered['date_suivi_diag1_quest'])
    #convert the dateq columns to datetime
    df_filtered[date_columns] = df_filtered[date_columns].apply(lambda col: pd.to_datetime(col))
    #print(df_filtered[['unique_id','date_suivi_diag1_quest']+date_columns].head(10))
    case_patterns = df_filtered[df_filtered['case_control'] == 1][['match_group', 'rempli_pattern',
    'unique_id','grid_pattern','date_suivi_diag1_quest']+date_columns].rename(
        columns={'rempli_pattern': 'case_pattern',
                 'unique_id': 'case_unique_id',
                 'grid_pattern': 'case_grid_pattern',
                 'date_suivi_diag1_quest': 'case_date_suivi_diag1_quest'}
    )
    #for each dateq column compute the difference in years between the dateq of the case and the date_suivi_diag1_quest
    for col in date_columns:
        case_patterns[col] = (case_patterns[col] - case_patterns['case_date_suivi_diag1_quest']).dt.days / 365.25
    #rename the dateq columns to have the prefix case_
    case_patterns = case_patterns.rename(columns={col: f"case_dt_{col}" for col in date_columns})

    # 2. Merge the case pattern back onto the main DataFrame
    df_merged = df_filtered.merge(case_patterns, on='match_group', how='left')

    # 3. Create a boolean mask to track mismatches
    mismatch_mask = pd.Series(False, index=df_merged.index)

    # 4. Loop over the unique values of last_avail_q (max 13 iterations)
    # This allows us to use fast vectorized slicing for each specific length
    for q in df_merged['last_avail_q'].unique():
        # Identify rows that have this specific q value
        q_mask = df_merged['last_avail_q'] == q
        
        # Compare the slices up to length 'q' vectorially
        mismatch = (
            df_merged.loc[q_mask, 'rempli_pattern'].str[:q] 
            != df_merged.loc[q_mask, 'case_pattern'].str[:q]
        )
        
        # Save the results into our master mask
        mismatch_mask.loc[q_mask] = mismatch

    # Create a column to indicate if there is a mismatch or not
    df_merged['at_least_warning'] = mismatch_mask.astype(int)
    df_filtered = df_merged.copy()
    del df_merged
    ####################################################
    case_dt_columns = [f"case_dt_dateq{i}" for i in range(1,14)]
    #print(df_filtered[['unique_id','date_suivi_diag1_quest']+case_dt_columns[:3]+date_columns[:3]].head(10))

    #keep these columns
    columns_to_keep = ['unique_id','split','case_control','last_avail_q','rempli_pattern','case_pattern',
    'grid_pattern','case_grid_pattern', 'diag_park_final1_quest','at_least_warning']+case_dt_columns
    columns_to_keep.extend([f'q_{q}_num_X' for q in range(1,14)]) 
    columns_to_keep.extend([f'q_{q}_num_text' for q in range(1,14)])
    columns_to_keep.extend([f'q_{q}_num_digit' for q in range(1,14)])
    columns_to_keep.extend(['etudegp','profq2','lateralite','relative_age','birth_date','follow_up_time']) #matching variables
    columns_to_keep.extend([]) #other variables of interest for the training ande evaluation of the model

    df_filtered = df_filtered[columns_to_keep]
    #save as parquet file
    #df_filtered.to_parquet(os.path.join(OUTPUT_PATH, "final_table_for_training.parquet"), index=False)

    # 2. Convert the pandas DataFrame to a PyArrow Table
    table = pa.Table.from_pandas(df_filtered)

    # 3. Define your custom metadata (Keys and values must be bytes)
    # 1. Define your dynamic variables
    current_user = "Andrea Morelli"
    generated_at = datetime.now().isoformat()
    file_comment = f"Generated from file {filename} with seed {seed} for reproducibility."
    custom_metadata = {
        b"comment": file_comment.encode("utf-8"),
        b"created_by": current_user.encode("utf-8"),
        b"generated_at": generated_at.encode("utf-8"),
    }

    # 4. Merge your custom metadata with any existing table metadata
    existing_metadata = table.schema.metadata or {}
    combined_metadata = {**existing_metadata, **custom_metadata}

    # 5. Replace the table's schema with the new metadata-enriched schema
    table = table.replace_schema_metadata(combined_metadata)

    # 6. Write to Parquet
    pq.write_table(table, os.path.join(output_path,'final_data_for_training.parquet'))

    return 

PREPARE_FOR_TRAINING = True
file_to_load = os.path.join(OUTPUT_PATH, "10_exact_w_fallback_age_mode_1_with_etudegp_profq2_split_filter_rempliseul\\final_matched_df.csv")
CREATE_DF = False
PREPARE_FOR_MATCHING = False
GENERATE_HANDEDNESS_IDS = False
MATCH = False
TYPE_OF_MATCHING = "exact_w_fallback" #exact or exact_w_fallback or at_least
SPLIT_BEFORE_MATCHING = True #if True, the split will be done before the matching, if False, the split will be done after the matching
AGE_MODE = 1 #0 means exact matching on age, 1 means matching with age within 1 year, 2 means matching with age within 2 years, etc.
N_MATCHES = 10 
SHOW_MATCHED = False
OVERRIDE = False
FILTER_REMPLI_SEUL = True #filter_rempliseul
EXCLUDE_CASES_REMPLISEUL = False
all_matching_cols = ['etudegp','profq2','lateralite','split']
matching_cols=['etudegp','profq2']#['etudegp','profq2','lateralite'] #the columns on which to perform the matching in addition to age and questionnaires. The values of these columns should be present before the matching phase (for example in the covariates file) and should not have missing values for the matched subjects. If empty, no matching will be done on these columns.
if SPLIT_BEFORE_MATCHING: 
    matching_cols+= ['split']
matching_cols_string = "_".join(matching_cols) if len(matching_cols) > 0 else ""
matching_name = f"{N_MATCHES}_{TYPE_OF_MATCHING}_age_mode_{AGE_MODE}"+(f"_with_{matching_cols_string}" if len(matching_cols) > 0 else "")
if FILTER_REMPLI_SEUL:
    matching_name += "_filter_rempliseul"
    if EXCLUDE_CASES_REMPLISEUL:
        matching_name += "_exclude_cases"

if __name__ == "__main__":
    #explore_covariates()
    #explore_extraction_data()

    if OVERRIDE:
        final_table = pd.read_csv(os.path.join(OUTPUT_PATH, "final_table_with_all_info.csv"), encoding='cp1252')
        #explore_rempliquest(final_table,OUTPUT_PATH)
        explore_q_patterns(final_table,OUTPUT_PATH)
    else:
        if CREATE_DF:
            final_table = main()
            check_validity_final_table(final_table)
            final_table.to_csv(os.path.join(OUTPUT_PATH, "final_table_with_all_info.csv"), index=False, encoding='cp1252')
        
        if PREPARE_FOR_MATCHING:
            final_table = pd.read_csv(os.path.join(OUTPUT_PATH, "final_table_with_all_info.csv"), encoding='cp1252')
            final_table = prepare_for_matching(final_table, split_before=SPLIT_BEFORE_MATCHING, n_test=150,f_val=0.1) #automatically save the results to OUTPUT_PATH
        
        if GENERATE_HANDEDNESS_IDS:
            final_table = pd.read_csv(os.path.join(OUTPUT_PATH, "final_table_with_all_info.csv"), encoding='cp1252')
            #select_ids_for_handedness_model(final_table)
            df_splitted = select_ids_for_handedness_model(final_table,n_controls=5,seed=42)
            df_splitted.to_csv(os.path.join(OUTPUT_PATH, "handedness_model_ids_all_qs_w_sentences.csv"), index=False, encoding='cp1252')
        
        if MATCH:
            #set seed for reproducibility
            seed = 42
            np.random.seed(seed)
            if SPLIT_BEFORE_MATCHING:
                final_table = pd.read_csv(os.path.join(OUTPUT_PATH, "final_table_for_matching_splitted.csv"), encoding='cp1252')
            else:
                final_table = pd.read_csv(os.path.join(OUTPUT_PATH, "final_table_for_matching.csv"), encoding='cp1252')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_matched_df, statistics_on_matches, failed_cases = matching(final_table,n_matches=N_MATCHES,time_of_entry_col="dateq1",time_of_exit_col="date_suivi_diag1_quest",
                    id_col="ident_projet", diag_col="diag_park_final1_quest",matching_cols=matching_cols,type_of_matching=TYPE_OF_MATCHING,
                    age_mode=AGE_MODE, filter_rempliseul=FILTER_REMPLI_SEUL,exclude_cases_rempliseul=EXCLUDE_CASES_REMPLISEUL)
            if not os.path.exists(os.path.join(OUTPUT_PATH, matching_name)):
                os.makedirs(os.path.join(OUTPUT_PATH, matching_name))
            final_matched_df.to_csv(os.path.join(OUTPUT_PATH, matching_name,"final_matched_df.csv"), index=False, encoding='cp1252')
            statistics_on_matches.to_csv(os.path.join(OUTPUT_PATH, matching_name,"matching_statistics.csv"), index=False, encoding='cp1252')
            failed_cases.to_csv(os.path.join(OUTPUT_PATH, matching_name,"failed_cases.csv"), index=False, encoding='cp1252')

        if SHOW_MATCHED:

            load_path = os.path.join(OUTPUT_PATH, matching_name)

            # Define your log file path (you can adjust this path as needed)
            log_file_path = os.path.join(load_path, "matching_evaluation.log")

            # A quick helper function to write messages to the specified log file
            def log_message(message):
                with open(log_file_path, "a", encoding="utf-8") as f:
                    f.write(str(message) + "\n")
            with open(log_file_path, "w", encoding="utf-8") as f:
                    f.write('Starting: \n')


            questionnairres = [str(i) for i in range(1,14)]

            final_matched_df = pd.read_csv(os.path.join(load_path, "final_matched_df.csv"), encoding='cp1252')
            grid_cols=[f'q_{q}_avail' for q in questionnairres]
            final_matched_df['rempli_pattern'] = final_matched_df[grid_cols].astype(str).agg(''.join, axis=1)

            #adding warning for rempliseul
            warn_cond1 = final_matched_df["rempli_seulq11"] == 0
            warn_cond2 = (final_matched_df["rempli_seulq11"] == 9) & (final_matched_df["rempli_seulq12"] == 0)
            # 2. Combine the conditions
            warning_mask = warn_cond1 | warn_cond2
            # 3. Create the new column (True becomes 1, False becomes 0)
            final_matched_df["rempliseul_warning"] = warning_mask.astype(int)

            filter_cols = ['ident_projet', 'match_group','case_control','diag_park_final1_quest',
            'remaining_after_age','remaining_after_matching_cols','remaining_exact','remaining_subjects',
            'follow_up_time','age_suivi_diag1_quest','date_suivi_diag1_quest',
            'relative_age','birth_date','last_avail_q','rempli_pattern']
            filter_cols.extend(['lateralite','etudegp','profq2','rempliseul_warning','rempli_seulq11','rempli_seulq12'])
            filter_cols.extend([f"dateq{i}" for i in range(1,14)])

            final_matched_df = final_matched_df[filter_cols]
            statistics_on_matches = pd.read_csv(os.path.join(load_path, "matching_statistics.csv"), encoding='cp1252')
            failed_cases = pd.read_csv(os.path.join(load_path, "failed_cases.csv"), encoding='cp1252')

            log_message("Evaluate case-control statistics..")
            #consider all the rows with case_control==0 and count how many of them are unique
            unique_controls = final_matched_df[final_matched_df['case_control']==0]['ident_projet'].unique()
            total_controls = final_matched_df[final_matched_df['case_control']==0].shape[0]
            total_cases = final_matched_df[final_matched_df['case_control']==1].shape[0]
            unique_cases = final_matched_df[final_matched_df['case_control']==1]['ident_projet'].unique()
            #consider all unique controls with case_control==0 and diag_park_final1_quest==1
            unique_controls_diag = final_matched_df[(final_matched_df['case_control']==0) & (final_matched_df['diag_park_final1_quest']==1)]['ident_projet'].unique()
            # check how many times a control is selected 
            counts = final_matched_df[final_matched_df['case_control']==0]['ident_projet'].value_counts()
            log_message(f"Total controls: {total_controls} of which {len(unique_controls)} are unique")
            log_message(f"Total cases: {total_cases} of which {len(unique_cases)} are unique (unique and toatal should be same)")
            log_message(f"Unique controls with diag_park_final1_quest==1: {len(unique_controls_diag)}")
            log_message(f"Fraction of unique controls: {len(unique_controls)/total_controls:.4f}")
            log_message("Control selection counts:")
            log_message(counts.describe())
            #log_message the number of ids that are repeated more than 3 times
            log_message(f"Number of controls that are repeated more than 3 times: {sum(counts > 3)}")
            log_message(f"Number of controls that are repeated more than 5 times: {sum(counts > 5)}")
            log_message("-"*50)

            log_message("Count cases and controls with rempliseul_warning...")
            cases_with_warning = final_matched_df[(final_matched_df['case_control']==1) & (final_matched_df['rempliseul_warning']==1)]['ident_projet'].nunique()
            controls_with_warning = final_matched_df[(final_matched_df['case_control']==0) & (final_matched_df['rempliseul_warning']==1)]['ident_projet'].nunique()
            log_message(f"Number of cases with rempliseul warning: {cases_with_warning}")
            log_message(f"Number of controls with rempliseul warning: {controls_with_warning}")
            log_message("-"*50)

            log_message("Overview of the matching results..")
            #log_message(final_matched_df.head(20))
            log_message(statistics_on_matches.describe())
            log_message("-"*50)
            log_message("Failed cases overview..")
            log_message(failed_cases['reason'].value_counts())

            log_message("-"*50)
            log_message("Count groups for which you have used the fallback and how many rows were selected with fallback: ")
            #Count the ids with 'case_control'==1 and with remaining_exact<N_matches
            fallback_cases = final_matched_df[(final_matched_df['case_control']==1) & (final_matched_df['remaining_exact']<N_MATCHES)]
            if TYPE_OF_MATCHING == "exact_w_fallback":
                n_used_fallback=fallback_cases['ident_projet'].nunique()
                n_selected_with_fallback = (N_MATCHES - fallback_cases['remaining_exact']).sum()
                unique_controls_with_fallback = final_matched_df[(final_matched_df['case_control']==0) & (final_matched_df['remaining_exact']<N_MATCHES)]['ident_projet'].nunique() 
                n_unique_selected_with_fallback = unique_controls_with_fallback
            else:
                n_used_fallback=0
                n_selected_with_fallback=0
                n_unique_selected_with_fallback=0
            log_message(f"Number of cases where fallback was used: {n_used_fallback}")
            log_message(f"Number of controls selected with fallback: {n_selected_with_fallback}.")
            log_message(f"Number of unique controls selected with fallback: {n_unique_selected_with_fallback} (actually this is includes all the controls in the cases for which fallback was used )")

            log_message("-"*50)
            log_message("Count suboptimal matches (cases with less than n_matches matches):")
            #group by match_group and count the number of controls for each case, then count how many cases have less than n_matches controls
            suboptimal_matches = final_matched_df.groupby('match_group').size()
            counts_suboptimal = suboptimal_matches.value_counts()
            log_message(counts_suboptimal)
            log_message(f"Number of cases with less than {N_MATCHES} matches: {sum(suboptimal_matches <= N_MATCHES)}")

            #save summary results across experiments
            if not os.path.exists(os.path.join(OUTPUT_PATH, "summary_table.xlsx")):
                #create a new dataframe with columns strategy, age_mode, matching_cols, total_controls, unique_controls, total_cases, unique_cases, repeated_controls, cases_with_fallback, selected_with_fallback, unique_selected_with_fallback, less_than_n_matches, 1_match, 2_matches, 3_matches
                df_summary = pd.DataFrame(columns=['strategy','n_controls', 'age_mode'] + [f'{col}' for col in all_matching_cols] + ['filter_rempliseul','exclude_cases_rempli',
                'total_controls', 'unique_controls', 
                'total_cases', 'unique_cases', 
                'repeated_controls', 'n_cases_on_which_fallback_is_used', 'n_controls_selected_w_fallback'
                , 'less_than_n_matches', 
                '1_match', '2_matches', '3_matches'])
                df_summary.to_excel(os.path.join(OUTPUT_PATH, "summary_table.xlsx"), index=False, sheet_name="Sheet1")
            file_path = os.path.join(OUTPUT_PATH, "summary_table.xlsx")
            df_excel = pd.read_excel(file_path, sheet_name="Sheet1")
            new_row = {
                'strategy': TYPE_OF_MATCHING,
                'age_mode': AGE_MODE,
                
            }
            #add all_matching_cols keys with value 1 if they are in matching_cols 0 otherwise
            for col in all_matching_cols:
                new_row[f'{col}'] = 1 if col in matching_cols else 0
            new_row['filter_rempliseul'] = 1 if FILTER_REMPLI_SEUL else 0
            new_row['n_controls'] = N_MATCHES
            new_row['exclude_cases_rempli'] = 1 if EXCLUDE_CASES_REMPLISEUL else 0
            
            #if this combination of strategy, age_mode and matching_cols already exists in the summary table -> skip
            duplicate_mask = (df_excel['strategy'] == new_row['strategy']) & (df_excel['age_mode'] == new_row['age_mode'])
            for col in all_matching_cols:
                duplicate_mask &= (df_excel[col] == new_row[col])
            duplicate_mask &= (df_excel['filter_rempliseul'] == new_row['filter_rempliseul'])
            duplicate_mask &= (df_excel['exclude_cases_rempli'] == new_row['exclude_cases_rempli'])
            duplicate_mask &= (df_excel['n_controls'] == new_row['n_controls'])
            
            if duplicate_mask.any():
                #substitute the values of the existing row with the new values
                df_excel.loc[duplicate_mask, 'total_controls'] = total_controls
                df_excel.loc[duplicate_mask, 'unique_controls'] = len(unique_controls)
                df_excel.loc[duplicate_mask, 'total_cases'] = total_cases
                df_excel.loc[duplicate_mask, 'unique_cases'] = len(unique_cases)
                df_excel.loc[duplicate_mask, 'repeated_controls'] = sum(counts > 1)
                df_excel.loc[duplicate_mask, 'n_cases_on_which_fallback_is_used'] = n_used_fallback
                df_excel.loc[duplicate_mask, 'n_controls_selected_w_fallback'] = n_selected_with_fallback
                df_excel.loc[duplicate_mask, 'less_than_n_matches'] = sum(suboptimal_matches <= N_MATCHES)
                df_excel.loc[duplicate_mask, '1_match'] = counts_suboptimal.get(1, 0)
                df_excel.loc[duplicate_mask, '2_matches'] = counts_suboptimal.get(2, 0)
                df_excel.loc[duplicate_mask, '3_matches'] = counts_suboptimal.get(3, 0)
                # Re-save the updated summary table back to Excel
                df_excel.to_excel(file_path, index=False, sheet_name="Sheet1")
            else:
                # Append the new row using pd.concat (safer for modern Pandas versions where .append() is deprecated)
                new_row['total_controls'] = total_controls
                new_row['unique_controls'] = len(unique_controls)
                new_row['total_cases'] = total_cases
                new_row['unique_cases'] = len(unique_cases)
                new_row['repeated_controls'] = sum(counts > 1)
                new_row['n_cases_on_which_fallback_is_used'] = n_used_fallback
                new_row['n_controls_selected_w_fallback'] = n_selected_with_fallback
                new_row['less_than_n_matches'] = sum(suboptimal_matches <= N_MATCHES)
                new_row['1_match'] = counts_suboptimal.get(1, 0)
                new_row['2_matches'] = counts_suboptimal.get(2, 0)
                new_row['3_matches'] = counts_suboptimal.get(3, 0)
                df_new_row = pd.DataFrame([new_row])
                df_excel = pd.concat([df_excel, df_new_row], ignore_index=True)
                
                # Re-save the updated summary table back to Excel
                df_excel.to_excel(file_path, index=False, sheet_name="Sheet1")
                print(f"Successfully appended and saved results for {TYPE_OF_MATCHING} (Age Mode: {AGE_MODE}).")

            final_matched_df.to_excel(os.path.join(load_path,'explore_matched_data.xlsx'), index=False, sheet_name='Results')

        if PREPARE_FOR_TRAINING:
             prepare_table_for_training(file_to_load,OUTPUT_PATH,seed=42, split_before=SPLIT_BEFORE_MATCHING)
        #print(1)
        #check_validity_final_table()
        #final_table = pd.read_csv(E3N_COVARIATES_PATH,encoding='cp1252')
        #prepare_for_matching(final_table)
        #check_progress_csv()
        #check_combined_success_ids()
        #update_progress_file()