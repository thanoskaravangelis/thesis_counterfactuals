import os,sys,ssl,nltk

#encountered multiple issues in HPC environment and these lines solved them
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM']='0'
sys.path.append('/users/pa21/ptzouv/tkaravangelis/')
sys.path.append('/users/pa21/ptzouv/tkaravangelis/mice/')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#imports
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from tqdm import tqdm
import re, time, logging, csv
import torch
#local imports
from scripts.helpers import *
from polyjuice.polyjuice.helpers import create_processor
from polyjuice.polyjuice.polyjuice_wrapper import Polyjuice
from src.utils import load_predictor, wrap_text, get_dataset_reader, get_ints_to_labels, add_probs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

def get_pos_tag_indexes(sentence, pos_tag):
    """Returns the indexes of the words with the specified POS tag."""
    nlp = create_processor(False)
    doc = nlp(sentence)
    pos_tags = [pos_tag]
    if pos_tag == "VERB":
        pos_tags = ["VERB", "AUX"]
    indexes = [i for i, x in enumerate(doc) if x.pos_ in pos_tags and str(x) not in ['.',',','!','%','#','-','~',';']]
    words = [doc[i].text for i in indexes]
    return words, indexes

def score_min_minimality(inp, perturbations):
    """Finds the most minimal perturbation"""
    min_minimality = 1
    min_pert = None
    new_label = None
    edited_contrast_pred = None
    logger.info("Scoring minimality.")
    for pert, label, contrast_pred  in perturbations:
        if pert is None or pert == "":
            continue
        score = score_minimality(inp, pert)
        if score < min_minimality:
            min_minimality = score
            min_pert = pert
            new_label = label
            edited_contrast_pred = contrast_pred
    logger.info("Finished scoring minimality.")
    return min_pert, new_label, edited_contrast_pred, min_minimality

def score_minimality(orig_sent, edited_sent, normalized=True):
    """Computes the edit distance between two sentences."""
    spacy = SpacyTokenizer()
    tokenized_original = [t.text for t in spacy.tokenize(orig_sent)]
    tokenized_edited = [t.text for t in spacy.tokenize(edited_sent)]
    lev = nltk.edit_distance(tokenized_original, tokenized_edited)
    if normalized: 
        return lev/len(tokenized_original)
    else:
        return lev

def predict_flip(orig_label, contrast_label, contrast_pred_idx, perturbations):
    """Runs predictor on perturbations and returns the one that flips the target prob."""
    logger.info("Running predictor to find target prob.")
    final_prob = -1
    validated = []
    for pert, j in tqdm(zip(perturbations, range(len(perturbations)))):
        if pert is None or pert == "":
            continue
        pred = predictor.predict(pert)
        pred = add_probs(pred)
        #get the prob of the contrast label
        edited_contrast_pred = pred['probs'][contrast_pred_idx]
        predicted_label = ints_to_labels[np.argmax(pred['probs'])]

        logger.info(wrap_text(f"Perturbation {j}: {pert}"))
        logger.info(f"Original prob: {orig_label}, new prob: {predicted_label}")
        #positive to negative flip
        if predicted_label == contrast_label:
            validated.append((pert, predicted_label, edited_contrast_pred))
    logger.info("Finished running predictor.")
    return validated

task="imdb"
inputs_path = "/users/pa21/ptzouv/tkaravangelis/data/newsgroups"

targeted_pos_tag = sys.argv[1]
if targeted_pos_tag not in ["ADJ", "NOUN", "VERB"]:
    print("Wrong POS tag.")
    sys.exit()

#="/users/pa21/ptzouv/tkaravangelis/gptneo"
#="/users/pa21/ptzouv/tkaravangelis/poly_model"
pj = Polyjuice(model_path="/users/pa21/ptzouv/tkaravangelis/poly_model", is_cuda=True)
ctrl_codes = ["resemantic", "restructure", "negation", "insert", "lexical", "shuffle", "quantifier"]
predictor = load_predictor("newsgroups")
ints_to_labels = get_ints_to_labels(predictor)

for num_of_phase in tqdm(range (2, 10)):
    # ορίζω σαν νέο foldername την νέα φάση
    folder_name = f"polyjuice_newsgroups_random_{targeted_pos_tag}_{num_of_phase}"
    out_folder = f"/users/pa21/ptzouv/tkaravangelis/polyjuice_results/{folder_name}"
    out_file = os.path.join(out_folder, "edits.csv")
    if num_of_phase != 0:
        prev_phase = num_of_phase - 1
        prev_file = os.path.join(f"/users/pa21/ptzouv/tkaravangelis/polyjuice_results/polyjuice_newsgroups_random_{targeted_pos_tag}_{prev_phase}", "edits.csv")
        edits = read_edits(prev_file)
        edits = get_best_edits(edits)
        # και φτιάχνω τα αντίστοιχα txt αρχεία
        create_files_polyjuice_news(edits, folder_name)
        inputs_path = out_folder
    else:
        os.makedirs(out_folder, exist_ok=True)
    # ορίζω τον dataset reader
    dr = get_dataset_reader(task, predictor)
    inputs = dr.get_inputs('test', inputs_path)
    inputs = [x for x in inputs if len(x) > 0 and re.search('[a-zA-Z]', x)]
    input_indices = np.array(range(len(inputs)))

    with open(out_file, "w") as csv_file:
        # write header
        fieldnames = ["data_idx", "sorted_idx", "orig_pred", "new_pred", 
                "contrast_pred", "orig_contrast_prob_pred", 
                "new_contrast_prob_pred", "orig_input", "edited_input", 
                "orig_editable_seg", "edited_editable_seg", 
                "minimality", "duration", "error"]
        writer = csv.writer(csv_file, delimiter="\t")
        writer.writerow(fieldnames)

        for idx, i in tqdm(enumerate(input_indices), total=len(input_indices)):
            inp = inputs[i]
            if len(inp) > 512:
                inp = inp[:512]
            inp = re.sub('[^a-zA-Z0-9 \n\.,!?]', '', inp)
            logger.info(wrap_text(f"ORIGINAL INSTANCE ({i}): {inp}"))
            # get original prediction
            orig_pred = predictor.predict(inp)
            orig_pred = add_probs(orig_pred)
            orig_probs = orig_pred['probs']
            orig_label = ints_to_labels[np.argmax(orig_probs)]
            # get target prediction
            contrast_pred_idx = np.array(orig_probs).argsort()[-2]
            target_label = ints_to_labels[contrast_pred_idx]
            # get original contrast prediction
            orig_contrast_prob = orig_probs[contrast_pred_idx]

            logger.info(f"Original prob: {str(orig_label)}")
            logger.info(f"Target prob: {str(target_label)}")

            start_time = time.time()
            error = False
            try:
                #text="This film promised a lot, so many beautiful and well playing actors but with a plot that had virtually NOTHING to say. So many potentially promising conflicts between the family members that could have been developed and elaborated but it was all dropped and not taken care of. There was no story to be told, just a show off of acting, technique, beautiful scenes - that were all EMPTY. But again, the acting was excellent so many of the individual scenes were entertaining, but as you became increasingly aware of the lack of underpinning ideas, even the acting lost its sense. So from the promising start you became increasingly disappointed as the non-story went along."[:512]
                pos_words, pos_indexes = get_pos_tag_indexes(inp, targeted_pos_tag)
                logger.info(f"POS words: {pos_words}")
	        
                perturbations = pj.perturb(inp,
                            blanked_sent = pj.get_random_blanked_sentences(inp, 
                                                                            pre_selected_idxes=pos_indexes, is_token_only=True, max_blank_sent_count=15, max_blank_block=15),
                            ctrl_codes=ctrl_codes,
                            num_beams=30,
                            perplex_thred=None,
                            num_perturbations=50)
                    

                logger.info(f"Generated {len(perturbations)} perturbations.")
                validated = predict_flip(orig_label, target_label, contrast_pred_idx, perturbations)

                if validated != []:
                    edited_sent, new_label, edited_contrast_pred, minimality = score_min_minimality(inp, validated)
                    logger.info(wrap_text(f"EDITED INSTANCE ({i}): {edited_sent}"))

                torch.cuda.empty_cache()

            except Exception as e:
                logger.info("ERROR: ", e)
                error = True
                validated = []
                
            end_time = time.time()
            duration = end_time - start_time

            if validated == []:
                writer.writerow([i, 0, orig_label, 
                    '', target_label, 
                    orig_contrast_prob, None, 
                    inp, None, 
                    inp, None, None, duration, error])
            else:
                writer.writerow([i, 0, orig_label, 
                    new_label, target_label, 
                    orig_contrast_prob, edited_contrast_pred, 
                    inp, edited_sent, 
                    inp, edited_sent, minimality, duration, error])
            csv_file.flush()
    csv_file.close()
