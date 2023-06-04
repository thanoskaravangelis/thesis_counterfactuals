import numpy as np
import pandas as pd
import os 

def read_edits(path):
    edits = pd.read_csv(path, sep="\t", lineterminator="\n", error_bad_lines=False, warn_bad_lines=True)

    if edits['new_pred'].dtype == pd.np.dtype('float64'):
        edits['new_pred'] = edits.apply(lambda row: str(int(row['new_pred']) if not np.isnan(row['new_pred']) else ""), axis=1)
        edits['orig_pred'] = edits.apply(lambda row: str(int(row['orig_pred']) if not np.isnan(row['orig_pred']) else ""), axis=1)
        edits['contrast_pred'] = edits.apply(lambda row: str(int(row['contrast_pred']) if not np.isnan(row['contrast_pred']) else ""), axis=1)
    else:
        edits['new_pred'].fillna(value="", inplace=True)
        edits['orig_pred'].fillna(value="", inplace=True)
        edits['contrast_pred'].fillna(value="", inplace=True)
    return edits

def get_best_edits(edits):
    """ MiCE writes all edits that are found in Stage 2, 
    but we only want to evaluate the smallest per input. 
    Calling get_sorted_e() """
    return edits[edits['sorted_idx'] == 0]

def evaluate_edits(edits):
    temp = edits[edits['sorted_idx'] == 0]
    minim = temp['minimality'].mean()
    flipped = temp[temp['new_pred'].astype(str)==temp['contrast_pred'].astype(str)]
    nunique = temp['data_idx'].nunique()
    flip_rate = len(flipped)/nunique
    duration=temp['duration'].mean()
    metrics = {
        "num_total": nunique,
        "num_flipped": len(flipped),
        "flip_rate": flip_rate,
        "minimality": minim,
        "duration": duration,
    }
    for k, v in metrics.items():
        print(f"{k}: \t{round(v, 3)}")
    return metrics

def create_files(edits, folder_name):
  parent_dir = f'/users/pa21/ptzouv/tkaravangelis/mice/results/imdb/edits/{folder_name}'

  os.makedirs(parent_dir, exist_ok=True)

  path_pos = os.path.join(parent_dir, 'pos')
  path_neg = os.path.join(parent_dir, 'neg')
  os.mkdir(path_pos)
  os.mkdir(path_neg)

  for _, row in edits.iterrows():
    prob = row['new_pred']
    if prob in [1, '1']:
      f = open(path_pos+f"/{row['data_idx']}.txt", "x")
      f.write(row["edited_editable_seg"])
      f.close()
    elif prob in[0, '0']:
      f = open(path_neg+f"/{row['data_idx']}.txt", "x")
      f.write(row["edited_editable_seg"])
      f.close()

def create_files_newsgroups(edits, folder_name):
  parent_dir = f'/users/pa21/ptzouv/tkaravangelis/mice/results/newsgroups/edits/{folder_name}'

  os.makedirs(parent_dir, exist_ok=True)

  path_pos = os.path.join(parent_dir, 'pos')
  path_neg = os.path.join(parent_dir, 'neg')
  os.makedirs(path_pos, exist_ok=True)
  os.makedirs(path_neg, exist_ok=True)

  for _, row in edits.iterrows():
    if not np.isnan(row['new_contrast_prob_pred']):
      prob = round(row['new_contrast_prob_pred'])
      if prob in [1, '1']:
        f = open(path_pos+f"/{row['data_idx']}.txt", "x")
        f.write(row["edited_editable_seg"])
        f.close()
      elif prob in [0, '0']:
        f = open(path_neg+f"/{row['data_idx']}.txt", "x")
        f.write(row["edited_editable_seg"])
        f.close()

def create_files_random_pos_newsgroups(edits, folder_name):
  parent_dir = f'/users/pa21/ptzouv/tkaravangelis/mice_pos_newsgroups/results/newsgroups/edits/{folder_name}'

  os.makedirs(parent_dir, exist_ok=True)

  path_pos = os.path.join(parent_dir, 'pos')
  path_neg = os.path.join(parent_dir, 'neg')
  os.makedirs(path_pos, exist_ok=True)
  os.makedirs(path_neg, exist_ok=True)

  for _, row in edits.iterrows():
    if not np.isnan(row['new_contrast_prob_pred']):
      prob = round(row['new_contrast_prob_pred'])
      if prob in [1, '1']:
        f = open(path_pos+f"/{row['data_idx']}.txt", "x")
        f.write(row["edited_editable_seg"])
        f.close()
      elif prob in [0, '0']:
        f = open(path_neg+f"/{row['data_idx']}.txt", "x")
        f.write(row["edited_editable_seg"])
        f.close()

def create_files_pos_newsgroups(edits, folder_name):
  parent_dir = f'/users/pa21/ptzouv/tkaravangelis/mice_newsgroups/results/newsgroups/edits/{folder_name}'

  os.makedirs(parent_dir, exist_ok=True)

  path_pos = os.path.join(parent_dir, 'pos')
  path_neg = os.path.join(parent_dir, 'neg')
  os.makedirs(path_pos, exist_ok=True)
  os.makedirs(path_neg, exist_ok=True)

  for _, row in edits.iterrows():
    if not np.isnan(row['new_contrast_prob_pred']):
      prob = round(row['new_contrast_prob_pred'])
      if prob in [1, '1']:
        f = open(path_pos+f"/{row['data_idx']}.txt", "x")
        f.write(row["edited_editable_seg"])
        f.close()
      elif prob in [0, '0']:
        f = open(path_neg+f"/{row['data_idx']}.txt", "x")
        f.write(row["edited_editable_seg"])
        f.close()

def create_files_polyjuice(edits, folder_name):
  parent_dir = f'/users/pa21/ptzouv/tkaravangelis/polyjuice_results/{folder_name}'

  os.makedirs(parent_dir, exist_ok=True)

  path_pos = os.path.join(parent_dir, 'pos')
  path_neg = os.path.join(parent_dir, 'neg')
  os.mkdir(path_pos)
  os.mkdir(path_neg)

  for _, row in edits.iterrows():
    prob = row['new_pred']
    if prob in [1, '1']:
      f = open(path_pos+f"/{row['data_idx']}.txt", "x")
      f.write(row["edited_editable_seg"])
      f.close()
    elif prob in[0, '0']:
      f = open(path_neg+f"/{row['data_idx']}.txt", "x")
      f.write(row["edited_editable_seg"])
      f.close()
