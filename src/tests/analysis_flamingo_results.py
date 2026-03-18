import argparse
import pandas as pd
import os
from pathlib import Path

QUESTIONNAIRE="13"
def main():
    args = parse_args()
    main_path = "//vms-e34n-databr/2025-handwriting\\results_from_flamingo"
    ref_path = os.path.join(main_path, "ref_pdf")

    ref_dir = Path(ref_path)
    exclude_name = f"updated_ref_pdf_Q{QUESTIONNAIRE}.csv"

    # f is a Path object; str(f) gives you the full absolute or relative path
    csv_paths = [
        str(f) for f in ref_dir.glob("*.csv") 
        if f.name.lower() != exclude_name.lower()
    ]

    # 2. Read each CSV and store them in a list
    # Use a list comprehension for memory efficiency and speed
    df_list = [pd.read_csv(path) for path in csv_paths]

    # 3. Concatenate all DataFrames in the list into one
    # axis=0 means stack rows (vertically); ignore_index=True resets the row numbers
    if df_list:
        final_df = pd.concat(df_list, axis=0, ignore_index=True)
        print(f"Successfully combined {len(df_list)} files.")
    else:
        print("No CSV files found to combine.")
        final_df = pd.DataFrame() # Create empty DF to prevent downstream errors
    
    #select only the ids with status="success" amd save to a success_df
    success_df = final_df[final_df['status'] == 'success'].copy()
    #select only the ids with status!="success" and save to a failed_df
    failed_df = final_df[final_df['status'] != 'success'].copy()
    #load the updated_ref_pdf_Q13.csv 
    updated_ref_df = pd.read_csv(os.path.join(ref_path, exclude_name))
    #set the Used column of the updated_ref_df to True if the id is in the success_df, False 
    updated_ref_df['Used'] = updated_ref_df['e3n_id_hand'].isin(success_df['e3n_id_hand'])
    #overwrite the updated_ref_df 
    #updated_ref_df.to_csv(os.path.join(ref_path, exclude_name), index=False)

    print(final_df.columns)
    print(len(final_df))
    print(final_df.head())
    #get the unique values and their count for the columns "Warning_ordering", "Warning_censoring", "status"
    for col in ["Warning_ordering", "Warning_censoring", "status"]:
        print(f"Unique values for {col}:")
        print(final_df[col].value_counts())
        print("\n")
    # get the 5 ids with larger value of the 'time' column
    top_5_time = final_df.nlargest(5, 'time')
    print("Top 5 IDs with largest time:")
    print(top_5_time[['e3n_id_hand', 'time']])
    # get the list of ids with 'status'=='timeout'
    timeout_ids = final_df[final_df['status'] == 'timeout']['e3n_id_hand'].tolist()
    print(f"IDs with status 'timeout': {timeout_ids[:min(10, len(timeout_ids))]}") # Print only the first 10 for brevity
    #return the average time and the number of ids with time>100 over the total
    avg_time = final_df['time'].mean()
    count_time_gt_100 = (final_df['time'] > 100).sum()
    print(f"Average time: {avg_time}")
    print(f"Number of IDs with time > 100: {count_time_gt_100} over {len(final_df)} total IDs")
    

def parse_args():
    """Handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Test script")
    '''parser.add_argument(
        "-f", "--folder_path",
        default=TEMPLATES_PATH,
        help="Path to the PDF file to convert",
    )
    parser.add_argument(
        "-s", "--save_path",
        default=SAVE_PATH,
        help="Directory to save the converted PNG images",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )'''
    return parser.parse_args()

if __name__ == "__main__":
    main()