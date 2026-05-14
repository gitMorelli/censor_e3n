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

def prepare_for_matching(df):
    from pandas.api.types import is_datetime64_any_dtype as is_date

    #columns necessary for matching
    columns_of_interest = ['ident_projet','diag_park_final1_quest', 'date_suivi_diag1_quest','birth_month', 'birth_year', 'aexclure']
    columns_of_interest.extend([f'dateq{i}' for i in range(1,14)])
    columns_of_interest.extend([f'q_{i}_grid_file_avail' for i in range(1,14)]) #sono i dati che effettivamente ho -> devo considerare questi
    #other columns I want to keep in the data
    columns_of_interest.extend(['cas_prev_dg1'])

    df_filtered = df[columns_of_interest]

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

    #print(is_date(df_filtered['date_suivi_diag1_quest']))    # Returns: True
    #print(is_date(df_filtered['dateq1']))  # Returns: False (it's just text)

    #save the filtered dataframe with the new columns for the matching phase
    df_filtered.to_csv(os.path.join(OUTPUT_PATH, "final_table_for_matching.csv"), index=False)
    #df_filtered.to_parquet(os.path.join(OUTPUT_PATH, "final_table_for_matching.parquet"))

    return df_filtered

def matching(df,n_matches,age_at_entry_col="years_int",time_of_exit_col="date_suivi_diag1_quest",id_col="ident_projet", 
    diag_col="diag_park_final1_quest",matching_cols=[],type_of_matching="exact",age_mode=0):

    #get the unique ids of the cases (diag_col==1)
    case_ids = df.loc[df[diag_col] == 1, id_col].unique().tolist()
    cases_with_no_questionnaires = []
    cases_with_no_matches = []
    matched_till_now = 0

    matched_results = []
    statistics_on_matches = {'id': [], 'remaining_subjects': [], 'remaining_cases': []}
    for ind,case in enumerate(case_ids):
        if ind % 100 == 0:
            print(f"Processing case {ind+1}/{len(case_ids)} with id {case}..")
        #get the date of exit for this case
        case_exit_date = df.loc[df[id_col] == case, time_of_exit_col].values[0]
        #get all rows in the dataframe with date of exit before the case exit date
        df_filtered = df.loc[df[time_of_exit_col] < case_exit_date] 

        # get the age of entry for this case
        case_entry_date = df.loc[df[id_col] == case, age_at_entry_col].values[0]
        if age_mode == 0:
            #keep only the rows with age of entry equal to the case entry age 
            df_filtered = df_filtered.loc[df[age_at_entry_col] == case_entry_date]
        elif age_mode > 0:
            #keep only the rows with age of entry within 1 years of the case entry age
            df_filtered = df_filtered.loc[df[age_at_entry_col].between(case_entry_date-age_mode, case_entry_date+age_mode)]

        #get the date at which each questionnaire was filled for this case
        case_q_dates = df.loc[df[id_col] == case, [f"dateq{i}" for i in range(1,14)]].values[0]
        #get the last questionnaire filled for this case before the exit date
        last_q_filled = None
        for i,q_date in enumerate(case_q_dates):
            if pd.isna(q_date):
                continue
            if q_date < case_exit_date:
                last_q_filled = i+1
            else:
                break
        if last_q_filled is None:
            cases_with_no_questionnaires.append(case)
            continue
        questionnaires_to_match = [f"q_{i}_grid_file_avail" for i in range(1, last_q_filled+1)]
        #get the value of the columns q_i_avail for this case
        case_q_avail = df.loc[df[id_col] == case, questionnaires_to_match].values[0]
        if  "exact" in type_of_matching:
            #keep only the rows with the same value for the columns q_i_avail as the case
            if type_of_matching == "exact":
                for i, q in enumerate(questionnaires_to_match):
                    df_filtered = df_filtered.loc[df_filtered[q] == case_q_avail[i]]
            elif type_of_matching == "exact_w_fallback":
                for i, q in enumerate(questionnaires_to_match):
                    df_filtered_temp = df_filtered.loc[df_filtered[q] == case_q_avail[i]]
                if df_filtered_temp.shape[0] > int(n_matches/2):
                    df_filtered = df_filtered_temp
                else:
                    #if the number of rows is too low, keep the rows with at least the same questionnaires filled as the case
                    for i, q in enumerate(questionnaires_to_match):
                        df_filtered = df_filtered.loc[df_filtered[q] >= case_q_avail[i]]
        elif type_of_matching == "at_least":
            #keep the rows with at least the same questionnaires filled as the case (if case has q1 and q2 filled, keep the rows with at least q1 and q2 filled, but they can have more)
            for i, q in enumerate(questionnaires_to_match):
                df_filtered = df_filtered.loc[df_filtered[q] >= case_q_avail[i]]
        
        #match on the other columns of interest (if specified)
        for col in matching_cols:
            case_col_value = df.loc[df[id_col] == case, col].values[0]
            df_filtered = df_filtered.loc[df_filtered[col] == case_col_value]
        
        #get the length of the filtered dataframe
        n_remaining_subj = df_filtered.shape[0]
        #get the rows that are cases in the filtered dataframe
        remaining_cases = df_filtered.loc[df_filtered[diag_col] == 1]
        n_remaining_cases = remaining_cases.shape[0]
        statistics_on_matches['id'].append(case)
        statistics_on_matches['remaining_subjects'].append(n_remaining_subj)
        statistics_on_matches['remaining_cases'].append(n_remaining_cases)

        #select n_matches random controls with repetition from the filtered dataframe
        if df_filtered.shape[0] > 0:
            matched_controls = df_filtered[id_col].sample(n=n_matches, replace=True).tolist()
        else:
            cases_with_no_matches.append(case)
            continue

        #select the case and the matched controls from the original dataframe 
        selected = df.loc[df[id_col].isin([case] + matched_controls)].copy()
        selected['match_group'] = matched_till_now+1
        matched_till_now += 1
        #create a column that indicates if the row is a case or a control
        selected['case_control'] = (selected[id_col] == case).astype(int)
        selected['last_avail_q'] = last_q_filled
        matched_results.append(selected.copy())
    #concatenate the matched results for all cases
    if len(matched_results) > 0:
        final_matched_df = pd.concat(matched_results, ignore_index=True)
    else:
        print("No matches found for any case.")
    print(f"Number of cases with no questionnaires: {len(cases_with_no_questionnaires)}")
    print(f"Number of cases with no matches: {len(cases_with_no_matches)}")
    statistics_on_matches_df = pd.DataFrame(statistics_on_matches)
    #create a df with the cases that had no questionnaires and the cases that had no matches indicating which is which
    failed_cases_df = pd.DataFrame({
        'id': cases_with_no_questionnaires + cases_with_no_matches,
        'reason': ['no_questionnaires'] * len(cases_with_no_questionnaires) + ['no_matches'] * len(cases_with_no_matches)
    })

    return final_matched_df, statistics_on_matches_df,failed_cases_df


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


CREATE_DF = False
MATCH = True
TYPE_OF_MATCHING = "at_least" #exact or exact_w_fallback or at_least
AGE_MODE=1 #0 means exact matching on age, 1 means matching with age within 1 year, 2 means matching with age within 2 years, etc.
SHOW = True
if __name__ == "__main__":
    #explore_covariates()
    #explore_extraction_data()
    if CREATE_DF:
        final_table = main()
        check_validity_final_table(final_table)
        final_table = prepare_for_matching(final_table)
    
    if MATCH:
        final_table = pd.read_csv(os.path.join(OUTPUT_PATH, "final_table_for_matching.csv"), encoding='cp1252')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_matched_df, statistics_on_matches, failed_cases = matching(final_table,n_matches=5,age_at_entry_col="years_int",time_of_exit_col="date_suivi_diag1_quest",
                id_col="ident_projet", diag_col="diag_park_final1_quest",matching_cols=[],type_of_matching=TYPE_OF_MATCHING,age_mode=AGE_MODE)
        final_matched_df.to_csv(os.path.join(OUTPUT_PATH, "final_matched_df.csv"), index=False, encoding='cp1252')
        statistics_on_matches.to_csv(os.path.join(OUTPUT_PATH, "matching_statistics.csv"), index=False, encoding='cp1252')
        failed_cases.to_csv(os.path.join(OUTPUT_PATH, "failed_cases.csv"), index=False, encoding='cp1252')

    if SHOW:
        questionnairres = [str(i) for i in range(1,14)]
        filter_cols = ['ident_projet', 'match_group','case_control','diag_park_final1_quest','years_int','date_suivi_diag1_quest','last_avail_q']
        filter_cols.extend([f'q_{q}_grid_file_avail' for q in questionnairres])
        filter_cols.extend([f"dateq{i}" for i in range(1,14)])
        final_matched_df = pd.read_csv(os.path.join(OUTPUT_PATH, "final_matched_df.csv"), encoding='cp1252')
        final_matched_df = final_matched_df[filter_cols]
        statistics_on_matches = pd.read_csv(os.path.join(OUTPUT_PATH, "matching_statistics.csv"), encoding='cp1252')
        failed_cases = pd.read_csv(os.path.join(OUTPUT_PATH, "failed_cases.csv"), encoding='cp1252')

        print("Evaluate case-control statistics..")
        #consider all the rows with case_control==0 and count how many of them are unique
        unique_controls = final_matched_df[final_matched_df['case_control']==0]['ident_projet'].unique()
        total_controls = final_matched_df[final_matched_df['case_control']==0].shape[0]
        total_cases = final_matched_df[final_matched_df['case_control']==1].shape[0]
        unique_cases = final_matched_df[final_matched_df['case_control']==1]['ident_projet'].unique()
        #consider all unique controls with case_control==0 and diag_park_final1_quest==1
        unique_controls_diag = final_matched_df[(final_matched_df['case_control']==0) & (final_matched_df['diag_park_final1_quest']==1)]['ident_projet'].unique()
        # check how many times a control is selected 
        counts = final_matched_df[final_matched_df['case_control']==0]['ident_projet'].value_counts()
        print(f"Total controls: {total_controls} of which {len(unique_controls)} are unique")
        print(f"Total cases: {total_cases} of which {len(unique_cases)} are unique (unique and toatal should be same)")
        print(f"Unique controls with diag_park_final1_quest==1: {len(unique_controls_diag)}")
        print(f"Fraction of unique controls: {len(unique_controls)/total_controls:.4f}")
        print("Control selection counts:")
        print(counts.describe())
        #print the number of ids that are repeated more than 3 times
        print(f"Number of controls that are repeated more than 3 times: {sum(counts > 3)}")
        print(f"Number of controls that are repeated more than 5 times: {sum(counts > 5)}")
        print("-"*50)

        print("Overview of the matching results..")
        print(final_matched_df.head(20))
        print(statistics_on_matches.describe())
        print(failed_cases['reason'].value_counts())

        final_matched_df.to_excel(os.path.join(OUTPUT_PATH,'explore','matched_data.xlsx'), index=False, sheet_name='Results')

    #print(1)
    #check_validity_final_table()
    #final_table = pd.read_csv(E3N_COVARIATES_PATH,encoding='cp1252')
    #prepare_for_matching(final_table)
    #check_progress_csv()
    #check_combined_success_ids()
    #update_progress_file()