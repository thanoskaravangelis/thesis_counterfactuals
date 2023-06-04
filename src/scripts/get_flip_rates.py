from helpers import *
import sys

folder_to = sys.argv[1]
folders_path = f"/users/pa21/ptzouv/tkaravangelis/polyjuice_results/{folder_to}"

folders = os.listdir(folders_path)
for folder in folders:
    if folder.split('_')[-1]=='0':
        first_folder = folder
        break
folder_name = first_folder.replace(f'_0','')

file_name = sys.argv[2]
outfile = open(f"/users/pa21/ptzouv/tkaravangelis/flip_rates/{file_name}.txt", "w")

for i in range(10):
    edits = read_edits(f"{folders_path}/{folder_name}_{i}/edits.csv")
    edits = get_best_edits(edits)
    outfile.write(f"Step {i+1}:\n")
    metrics = evaluate_edits(edits)
    for k, v in metrics.items():
         outfile.write(f"{k}: \t{round(v, 3)}\n")   
    outfile.write("----"*5)
    outfile.write("\n")
outfile.close()

