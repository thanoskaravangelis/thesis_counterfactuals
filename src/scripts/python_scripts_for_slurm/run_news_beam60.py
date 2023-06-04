from helpers import *
from tqdm import tqdm 
import pandas as pd
import os

inputs_path = "/users/pa21/ptzouv/tkaravangelis/data/newsgroups"
targeted_pos_tag = "ADJ" #@param ["ADJ", "VERB", "NOUN", "PRON", "ADV"]

#os.system(f"python3 /users/pa21/ptzouv/tkaravangelis/mice_grad/run_stage_two.py -task newsgroups -generate_type=beam -generation_num_beams=60 -num_generations=60 -stage2_exp mice_news_beam60_0 -editor_path results/newsgroups/editors/mice/newsgroups_editor.pth -targeted_pos_tag {targeted_pos_tag} -inputs_path {inputs_path}")

for num_of_phase in tqdm(range (1, 11)):
    # διαβάζω ένα αρχείο csv με τα αποτελέσματα του ΠΡΟΗΓΟΥΜΕΝΟΥ ΓΥΡΟΥ 
    edits = read_edits(f"/users/pa21/ptzouv/tkaravangelis/mice_grad/results/newsgroups/edits/mice_news_beam60_{num_of_phase - 1}/edits.csv")
    edits = get_best_edits(edits)
    # ορίζω σαν νέο foldername την νέα φάση
    folder_name = f"mice_news_beam60_{num_of_phase}"
    # και φτιάχνω τα αντίστοιχα txt αρχεία
    create_files_grad_newsgroups(edits, folder_name)

    os.system(f"python3 run_stage_two.py -task newsgroups -generate_type=beam -generation_num_beams=60 -num_generations=60 -stage2_exp {folder_name} -editor_path results/newsgroups/editors/mice/newsgroups_editor.pth -targeted_pos_tag {targeted_pos_tag} -inputs_path /users/pa21/ptzouv/tkaravangelis/mice_grad/results/newsgroups/edits/{folder_name}")
