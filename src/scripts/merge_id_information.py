#!/usr/bin/env python3
import argparse
import logging
import os
from time import perf_counter
from pathlib import Path
import sys
import shutil

import pandas as pd


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
    columns_of_interest = ['ident_projet','diag_park_final1_quest', 'date_suivi_diag1_quest','birth_month', 'birth_year', 'dateq1','aexclure','cas_prev_dg1']
    df_filtered = df[columns_of_interest]

    #count the number of values with cas_prev_dg1==1
    print(f"Number of values with cas_prev_dg1==1: {df_filtered[df_filtered['cas_prev_dg1']==1].shape[0]}") #cas_prev_dg1 is not in the columns of interest, I assume you meant diag_park_final1_quest
    #count the number of values with aexclure==1
    print(f"Number of values with aexclure==1: {df_filtered[df_filtered['aexclure']==1].shape[0]}")
    # and the number of cases with both the aexclure==1 and diag_park_final1_quest==1
    print(f"Number of values with aexclure==1 and diag_park_final1_quest==1: {df_filtered[(df_filtered['aexclure']==1) & (df_filtered['diag_park_final1_quest']==1)].shape[0]}")

    
    #select only the lines with aexclure==0
    df_filtered = df_filtered[df_filtered['aexclure']!=1]

    #count nan values for each column
    print(df_filtered.isna().sum())
    #show the lines with nan values in the 'date_suivi_diag1_quest' column
    print(df_filtered[df_filtered['date_suivi_diag1_quest'].isna()])

    #df_filtered['date_suivi_diag1_quest_convert'] = pd.to_datetime(df['date_suivi_diag1_quest'], format='%d%b%Y') #if you want to keep the original colun and check
    df_filtered['date_suivi_diag1_quest'] = pd.to_datetime(df['date_suivi_diag1_quest'], format='%d%b%Y')
    df_filtered['dateq1'] = pd.to_datetime(df['dateq1'], format='%d%b%Y')
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
    #df_filtered.to_csv(os.path.join(OUTPUT_PATH, "final_table_for_matching.csv"), index=False)
    df_filtered.to_parquet(os.path.join(OUTPUT_PATH, "final_table_for_matching.parquet"))

    pass

def select_matched_ids():
    #load the file with the matched ids and the distance computed for each of them
    matched_ids_path = os.path.join(OUTPUT_PATH, "matched_ids_R.csv")
    matched_ids_df = pd.read_csv(matched_ids_path)
    print(matched_ids_df.head(10))
    
    return 

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

        
        for q in questionnairres:
            #load ids present in the ref_file
            ref_file_path = os.path.join(REF_PDF_PATH, f"ref_pdf_Q{q}.csv")
            raw_ref_data = pd.read_csv(ref_file_path)
            ref_ids = set(raw_ref_data['e3n_id_hand'].to_list())
            #set q_i_avail_ref to 1 if present, 0 if not
            if set(ref_ids).issubset(all_ids):
                final_table[f"q_{q}_avail_ref"] = final_table['ident_projet'].apply(lambda x: 1 if x in ref_ids else 0)
            else:
                print(f"Warning: Some ids in the ref file for questionnaire Q{q} are not present in the final table. This may lead to missing values in the q_{q}_avail_ref column.")

            #select ids from the rempli_data df for this questionnaire
            ids_with_q_filled = set(rempli_data[rempli_data[f'rempli_q{q}'] == 1]['ident_projet'].to_list())
            #set q_i_avail to 1 if present, 0 if not
            if ids_with_q_filled.issubset(all_ids):
                final_table[f"q_{q}_avail"] = final_table['ident_projet'].apply(lambda x: 1 if x in ids_with_q_filled else 0)
            else:
                print(f"Warning: Some ids in the rempli file for questionnaire Q{q} are not present in the final table. This may lead to missing values in the q_{q}_avail column.")


            #load ids present in the file that collects all the succesfully processed ids for the censoring
            censoring_file_path = os.path.join(CENSORED_PATH, f"Q{q}", f"run_data_q{q}", "csv_results_aggregated", "combined_success_ids.csv")
            censoring_data = pd.read_csv(censoring_file_path)
            ids_in_censored = set(censoring_data['e3n_id_hand'].to_list())
            #set q_i_avail_cens to 1 if present, 0 if not
            if ids_in_censored.issubset(all_ids):
                final_table[f"q_{q}_avail_cens"] = final_table['ident_projet'].apply(lambda x: 1 if x in ids_in_censored else 0)
            else:
                print(f"Warning: Some ids in the censoring file for questionnaire Q{q} are not present in the final table. This may lead to missing values in the q_{q}_avail_censored column.")
            #set the column censoring warnings
            first_stage_ids = set(censoring_data[censoring_data["Warning_ordering"]=="First stage"]['e3n_id_hand'].to_list())
            final_table[f"q_{q}_censoring_ord"] = final_table['ident_projet'].apply(lambda x: 0 if x in first_stage_ids else -1)
            second_stage_ids = set(censoring_data[censoring_data["Warning_ordering"]=="Second stage"]['e3n_id_hand'].to_list())
            final_table[f"q_{q}_censoring_ord"] = final_table.apply(lambda row: 1 if row['ident_projet'] in second_stage_ids else row[f"q_{q}_censoring_ord"], axis=1)
            ocr_stage_ids = set(censoring_data[~censoring_data["Warning_ordering"].isin(['First stage', 'Second stage'])]['e3n_id_hand'])
            final_table[f"q_{q}_censoring_ord"] = final_table.apply(lambda row: 2 if row['ident_projet'] in ocr_stage_ids else row[f"q_{q}_censoring_ord"], axis=1)
            ocr_with_warning_ids = set(censoring_data[censoring_data["Warning_ordering"].str.contains("OCR", na=False)]['e3n_id_hand'])
            final_table[f"q_{q}_censoring_ord"] = final_table.apply(lambda row: 3 if row['ident_projet'] in ocr_with_warning_ids else row[f"q_{q}_censoring_ord"], axis=1)
            #set the column censoring warnings
            id_with_large_censoring = set(censoring_data[censoring_data["Warning_censoring"].str.contains("large", na=False)]['e3n_id_hand'])
            final_table[f"q_{q}_censoring_warn"] = final_table.apply(lambda row: 1 if row['ident_projet'] in id_with_large_censoring else 0, axis=1)

            #select the ids that were successfully extracted for this questionnaire
            filtered_extraction_data = extraction_data[extraction_data['questionnaire']==f'q{q}']
            ids_with_q_extracted = set(filtered_extraction_data['id'].to_list())
            #print(len(ids_with_q_extracted))
            #ogni id presente nella tabella finale è anche in h5 -> ha un file grid associato
            final_table[f"q_{q}_grid_file_avail"] = final_table['ident_projet'].apply(lambda x: 1 if x in ids_with_q_extracted else 0)
            #aggiungo i valori delle variabili scalari
            # 1. Select only the columns you need from the source
            cols_to_pull = ['id','X', 'hand', 'hand_partial_full',  'hand_sentences_full',  'number']
            new_cols = [f"q_{q}_num_X", f"q_{q}_num_text", f"q_{q}_num_part", f"q_{q}_num_sent", f"q_{q}_num_digit"]
            subset_data = filtered_extraction_data[cols_to_pull]
            # 2. Rename columns to include your "q" prefix before merging
            subset_data = subset_data.rename(columns={
                'X': new_cols[0],
                'hand': new_cols[1],
                'hand_partial_full': new_cols[2],
                'hand_sentences_full': new_cols[3],
                'number': new_cols[4]
            })
            # 3. Merge onto final_table
            final_table = final_table.merge(
                subset_data, 
                left_on='ident_projet', 
                right_on='id', 
                how='left'
            ).drop(columns=['id']) # Drop the extra ID column created by the merge
            # 4. Fill the NaNs with -1 globally for those two columns
            final_table[new_cols] = final_table[new_cols].fillna(-1)

            #return 
    print(final_table.head(10))

    return 0

if __name__ == "__main__":
    #explore_covariates()
    #explore_extraction_data()
    #main()
    final_table = pd.read_csv(E3N_COVARIATES_PATH,encoding='cp1252')
    prepare_for_matching(final_table)