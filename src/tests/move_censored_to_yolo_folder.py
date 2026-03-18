import argparse
import pandas as pd
import os
from pathlib import Path
import shutil

def get_first_level_subfolders(directory_path):
    path = Path(directory_path)
    # .iterdir() only looks at the immediate contents
    return [str(f) for f in path.iterdir() if f.is_dir()]
def get_subfolder_names(directory_path):
    path = Path(directory_path)
    # We use f.name to get just the folder name
    return [f.name for f in path.iterdir() if f.is_dir()]
def copy_selected_files(source_folder, destination_folder, file_list, added_string):
    src_path = Path(source_folder)
    dest_path = Path(destination_folder)

    # 1. Create the destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    for filename in file_list:
        filename=f"{added_string}_{filename}"
        file_to_copy = os.path.join(src_path,filename)
        
        # 2. Check if the file actually exists before trying to copy
        if file_to_copy.exists():
            # copy2 preserves metadata like timestamps
            shutil.copy2(file_to_copy, dest_path / filename)
            #print(f"Successfully copied: {filename}")


NUM_IDS = -1
def main():
    q_to_transfer = [i for i in range(1,14)] #i can specify different numbers for each questionanaire
    num_to_transfer = [NUM_IDS for _ in q_to_transfer]

    args = parse_args()
    main_source_path = "//vms-e34n-databr/2025-handwriting\\data\\censored_for_yolo\\censored_images"
    main_destination_path = "//vms-e34n-databr/2025-handwriting\\data\\training_obj_det_model_png"

    # Configurazione del file di log (quali id ho già trasferito per quali questionari)
    colonne = ["ID", "Q"] # Sostituisci con i nomi che preferisci
    file_path = os.path.join(main_source_path, "transferred_ids.csv")
    if file_path.exists():
        # Se il file esiste, lo carichiamo
        df = pd.read_csv(file_path)
        print(f"File caricato con successo.")
    else:
        # Se non esiste, creiamo un DataFrame vuoto con le due colonne
        df = pd.DataFrame(columns=colonne)
        # Salviamolo come CSV
        df.to_csv(file_path, index=False)
        print(f"File non trovato. Creato un nuovo file con le colonne: {colonne}")

    #get the list of folders in the main_source_path 
    folders = get_subfolder_names(main_source_path)
    
    for i,q in enumerate(q_to_transfer):
        count_questionnairres = 0
        #get the list of IDs for which i have the Q column equal to q in the df
        transferred_ids = df[df['Q'] == q]['ID'].tolist()
        #get the list of folders that are not in the transferred_ids list
        updated_folders = [folder for folder in folders if folder not in transferred_ids]
        for folder in updated_folders:
            #check if there is a folder named q in the folder
            q_folder_path = os.path.join(main_source_path, folder, f"{q}")
            if os.path.exists(q_folder_path):
                #transfer all the png files in the q_folder_path to the main_destination_path/q/ folder
                destination_q_folder_path = os.path.join(main_destination_path, f"q_{q}")
                os.makedirs(destination_q_folder_path, exist_ok=True)
                #get the list of png files filenames in the q_folder_path
                png_files = [f for f in os.listdir(q_folder_path) if f.endswith('.png')]
                #transfer the png files to the destination_q_folder_path
                copy_selected_files(q_folder_path, destination_q_folder_path, png_files, folder)
                count_questionnairres += 1
            #add the id and the q to the df
            new_row = {'ID': folder, 'Q': q}
            df = df.append(new_row, ignore_index=True)

            if count_questionnairres >= num_to_transfer[i] and num_to_transfer[i] > 0:
                print(f"Transferred {count_questionnairres} questionnaires for Q{q}. Moving to the next questionnaire.")
                break
    #save the df to the csv file
    df.to_csv(file_path, index=False)
                
    

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