import sys, pickle, os
sys.path.append('/users/pa21/ptzouv/tkaravangelis/mice/')
import pandas as pd
from tqdm import tqdm
from src.utils import *
from helpers import *

# utility functions for loading and saving objects
def save_pickle(obj, filename):
    with open(filename,'wb') as f:
        pickle.dump(obj,f)
  
def load_pickle(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

# Get the input folder path where results are located
folders_path = sys.argv[1]
# Can be one of imdb, newsgroups
task = sys.argv[2]
output_path = '/users/pa21/ptzouv/tkaravangelis/pickle_files/'
# Name specified by the user
output_name = sys.argv[3]

# Load the predictor
# predictor = load_predictor(task)

folders = os.listdir(folders_path)
for folder in folders:
    if folder.split('_')[-1]=='0':
        first_folder = folder
        break

picklist = []
edits = read_edits(f"{folders_path}/{first_folder}/edits.csv")
edits = get_best_edits(edits)
for index, row in tqdm(edits.iterrows()):
  orig_input = row['orig_input']
  probs = [row['orig_pred']]
  my_tupe = (orig_input, probs)
  picklist.append([my_tupe])

folder_name = first_folder.replace(f'_0','')
repo = first_folder.split('_')[0]

for sublist in tqdm(picklist):
  text = sublist[0][0]
  for i in range(10):
    edits = read_edits(f"{folders_path}/{folder_name}_{i}/edits.csv")
    edits = get_best_edits(edits) 
    selected_edits = edits.loc[edits["orig_input"] == text]
    probs = ""
    if selected_edits.shape[0] > 0:
        probs = [selected_edits.iloc[0]['orig_pred']]
    if i!=0:
        sublist.append((text, probs))
    #print(orig_probs)
    if selected_edits.shape[0] > 0:
        edited_input = selected_edits.iloc[0]["edited_input"]
    else: 
        break
    
    if pd.isna(edited_input):
        break
    else:
        #display_classif_results(selected_edits)
        text = str(edited_input)

save_pickle(picklist, f"{output_path}/{task}_{repo}_{output_name}.pkl")
