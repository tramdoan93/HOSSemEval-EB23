# This file contains the evaluation functions

import re
import editdistance
from sentence_transformers import SentenceTransformer
import torch
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentiment_word_list = ['positive', 'negative', 'neutral']
aspect_cate_list = ['location general',
 'food prices',
 'food quality',
 'ambience general',
 'service general',
 'restaurant prices',
 'drinks prices',
 'restaurant miscellaneous',
 'drinks quality',
 'drinks style_options',
 'restaurant general',
 'food style_options',
  'facility',
  'amenity',
  'service',
  'experience',
  'branding',
  'loyalty',
]

def replace_special_symbols(input_string):
    # Define a regular expression pattern to match special symbols
    pattern = re.compile(r'[^a-zA-Z0-9\s]')

    # Replace special symbols with a dot
    result_string = re.sub(pattern, '', input_string)

    return result_string

def resolve_incomplete_sentiment(sentiment):

    for s in sentiment_word_list:
        if sentiment in s and sentiment != s:
            return s

    return 

def post_process_spans_extraction(triplets, type_ext):
  tmp = []
  for triplet in triplets:
    if type_ext == 'pred':
      print('Sentence:', triplet)
    if triplet[0] == '' or \
      (triplet[-2] not in aspect_cate_list) or \
      (triplet[-1] not in sentiment_word_list):
      tmp.append(['', '', ''])
    else:

      # Replace special symbols with a non-space
      triplet[-1] = replace_special_symbols(triplet[-1])

      # Replace imcomplete sentiment with fully form sentiment
      triplet[-2] = resolve_incomplete_sentiment(triplet[-2])

      tmp.append(triplet)
    # if (triplet[-1] not in aspect_cate_list) or (triplet[-2] not in sentiment_word_list):
    #   count_error += 1
    #   triplet = ['', '', '']
  return tmp




def extract_spans_extraction(task, seq, type_ext):
    extractions = []
    if task == 'uabsa' and seq.lower() == 'none':
        return []
    else:
        if task in ['uabsa', 'aope']:
            all_pt = seq.split('; ')
            for pt in all_pt:
                pt = pt[1:-1]
                try:
                    a, b = pt.split(', ')
                except ValueError:
                    a, b = '', ''
                extractions.append((a, b))
        elif task in ['tasd', 'aste']:
            all_pt = seq.split('; ')
            # print('Type of extraction: ', type_ext)
            for pt in all_pt:

                # print('Aspect list: ', pt)
                pt = pt[1:-1]
                # print('New aspect list: ', pt)
                try:
                    ptl = pt.split(', ')
                    if len(ptl) > 1:
                      a, b, c = '', '', ''
                      if ptl[-2].lower() in aspect_cate_list and len(ptl) > 3:
                          a, b, c = ', '.join(ptl[:-2]), ptl[-2], ptl[-1]
                      else:
                          a, b, c = ptl
                      a, b, c = a.lower(), b.lower(), c.lower()
                    else:
                      raise ValueError('')
                except ValueError:
                    a, b, c = '', '', ''
                # print('Result of extraction: ', a + ',' + b + ',' + c)
                extractions.append([a, b, c])

        # Post-process extractions
        extractions = post_process_spans_extraction(extractions, type_ext)
        return extractions


def extract_spans_annotation(task, seq):
    if task in ['aste', 'tasd']:
        extracted_spans = extract_triplets(seq)
    elif task in ['aope', 'uabsa']:
        extracted_spans = extract_pairs(seq)

    return extracted_spans


def extract_pairs(seq):
    aps = re.findall('\[.*?\]', seq)
    aps = [ap[1:-1] for ap in aps]
    pairs = []
    for ap in aps:
        # the original sentence might have 
        try:
            at, ots = ap.split('|')
        except ValueError:
            at, ots  = '', ''
        
        if ',' in ots:     # multiple ots 
            for ot in ots.split(', '):
                pairs.append((at, ot))
        else:
            pairs.append((at, ots))    
    return pairs        


def extract_triplets(seq):
    aps = re.findall('\[.*?\]', seq)
    aps = [ap[1:-1] for ap in aps]
    triplets = []
    for ap in aps:
        try:
            a, b, c = ap.split('|')
        except ValueError:
            a, b, c = '', '', ''
        
        # for ASTE
        if b in sentiment_word_list:
            if ',' in c:
                for op in c.split(', '):
                    triplets.append((a, b, op))
            else:
                triplets.append((a, b, c))
        # for TASD
        else:
            if ',' in b:
                for ac in b.split(', '):
                    triplets.append((a, ac, c))
            else:
                triplets.append((a, b, c))

    return triplets


def recover_terms_with_editdistance(original_term, sent):
    words = original_term.split(' ')
    new_words = []
    for word in words:
        edit_dis = []
        for token in sent:
            edit_dis.append(editdistance.eval(word, token))
        smallest_idx = edit_dis.index(min(edit_dis))
        new_words.append(sent[smallest_idx])
    new_term = ' '.join(new_words)
    return new_term


def fix_preds_uabsa(all_pairs, sents):

    all_new_pairs = []
    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                # AT not in the original sentence
                if pair[0] not in  ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                if pair[1] not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(pair[1], sentiment_word_list)
                else:
                    new_sentiment = pair[1]

                new_pairs.append((new_at, new_sentiment))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


def fix_preds_aope(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                #print(pair)
                # AT not in the original sentence
                if pair[0] not in  ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]

                # OT not in the original sentence
                ots = pair[1].split(', ')
                new_ot_list = []
                for ot in ots:
                    if ot not in ' '.join(sents[i]):
                        # print('Issue')
                        new_ot_list.append(recover_terms_with_editdistance(ot, sents[i]))
                    else:
                        new_ot_list.append(ot)
                new_ot = ', '.join(new_ot_list)

                new_pairs.append((new_at, new_ot))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)

    return all_new_pairs


# for ASTE
def fix_preds_aste(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                #two formats have different orders
                p0, p1, p2 = pair
                # for annotation-type
                if p1 in sentiment_word_list:
                    at, ott, ac = p0, p2, p1
                    io_format = 'annotation'
                # for extraction type
                elif p2 in sentiment_word_list:
                    at, ott, ac = p0, p1, p2
                    io_format = 'extraction'

                #print(pair)
                # AT not in the original sentence
                if at not in  ' '.join(sents[i]):
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(at, sents[i])
                else:
                    new_at = at
                
                if ac not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(ac, sentiment_word_list)
                else:
                    new_sentiment = ac
                
                # OT not in the original sentence
                ots = ott.split(', ')
                new_ot_list = []
                for ot in ots:
                    if ot not in ' '.join(sents[i]):
                        # print('Issue')
                        new_ot_list.append(recover_terms_with_editdistance(ot, sents[i]))
                    else:
                        new_ot_list.append(ot)
                new_ot = ', '.join(new_ot_list)
                if io_format == 'extraction':
                    new_pairs.append((new_at, new_ot, new_sentiment))
                else:
                    new_pairs.append((new_at, new_sentiment, new_ot))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)
    
    return all_new_pairs


def fix_preds_tasd(all_pairs, sents):

    all_new_pairs = []

    for i, pairs in enumerate(all_pairs):
        new_pairs = []
        if pairs == []:
            all_new_pairs.append(pairs)
        else:
            for pair in pairs:
                #print(pair)
                # AT not in the original sentence
                sents_and_null = ' '.join(sents[i]) + 'NULL'
                if pair[0] not in  sents_and_null:
                    # print('Issue')
                    new_at = recover_terms_with_editdistance(pair[0], sents[i])
                else:
                    new_at = pair[0]
                
                # AC not in the list
                acs = pair[1].split(', ')
                new_ac_list = []
                for ac in acs:
                    if ac not in aspect_cate_list:
                        new_ac_list.append(recover_terms_with_editdistance(ac, aspect_cate_list))
                    else:
                        new_ac_list.append(ac)
                new_ac = ', '.join(new_ac_list)
                
                if pair[2] not in sentiment_word_list:
                    new_sentiment = recover_terms_with_editdistance(pair[2], sentiment_word_list)
                else:
                    new_sentiment = pair[2]
            
                new_pairs.append((new_at, new_ac, new_sentiment))
                # print(pair, '>>>>>', word_and_sentiment)
                # print(all_target_pairs[i])
            all_new_pairs.append(new_pairs)
    
    return all_new_pairs


def fix_pred_with_editdistance(all_predictions, sents, task):
    if task == 'uabsa':
        fixed_preds = fix_preds_uabsa(all_predictions, sents)
    elif task == 'aope':
        fixed_preds = fix_preds_aope(all_predictions, sents) 
    elif task == 'aste': 
        fixed_preds = fix_preds_aste(all_predictions, sents) 
    elif task == 'tasd':
        fixed_preds = fix_preds_tasd(all_predictions, sents) 
    else:
        print("*** Unimplemented Error ***")
        fixed_preds = all_predictions

    return fixed_preds

def length_of_null_quads(list_span: list):
    return sum(1 for x in list_span if (x[0] == '' and x[1] == '' and x[2] == ''))


def check_label(pred_i, gold, model):
    
    
    for g in gold:
        if pred_i == g:
            return 1
        else:
              # Option 1
            encode_gold = model.encode(g[1])
            encode_pred = model.encode(pred_i[1])
            
            cos = torch.nn.CosineSimilarity(dim = 0, eps=1e-6)
            sim = cos(torch.Tensor(encode_gold), torch.Tensor(encode_pred)).item()
            # Option 2
            # vectors = model.fit_transform([g[0], pred_i[0]])
            # sim = cosine_similarity(vectors)
            # sim = sim[0,1]
            
            # Check conditional
            if sim >= 0.45 and pred_i[0] == g[0] and pred_i[2:] == g[2:]:
                return 1
    return 0



def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred, n_gold_null, n_pred_null = 0, 0, 0, 0, 0

    # Option 1: sentence bert - encoder transformers
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    # Option 2: statistic technique: tf-idf
    # vectorizer = TfidfVectorizer()
  
    for i in range(len(pred_pt)):
            # n_gold += len(gold_pt[i])
            # n_pred += len(pred_pt[i])

            # for t in pred_pt[i][]:
            #     if t in gold_pt[i]:
            #         n_tp += 1

            ####### CONFIG 1 #######
            # n_gold += len(gold_pt[i])
            # n_gold_null += length_of_null_quads(gold_pt[i])

            # n_pred += len(pred_pt[i])
            # n_pred_null += length_of_null_quads(pred_pt[i])

            # for p, g in zip(pred_pt[i], gold_pt[i]):
            #       if (p[0] != '' and g[0] != '') and (p[1] != '' and g[1] != ''):

            #           if p == g:
            #               n_tp += 1
            #           else:
            #               # Similarity between gold and pred by bert embedding
            #               encode_gold = sbert_model.encode(g[1])
            #               encode_pred = sbert_model.encode(p[1])

            #               # Define cosine similarity
            #               cos = torch.nn.CosineSimilarity(dim = 0, eps=1e-6)
            #               sim = cos(torch.Tensor(encode_gold), torch.Tensor(encode_pred)).item()

            #               if sim >= 0.4 and p[0].lower() == g[0].lower() and p[2:]== g[2:]:
            #                   n_tp += 1

            ####### ........ #######

             ####### CONFIG 2 #######
            n_gold += len(gold_pt[i])
            n_gold_null += length_of_null_quads(gold_pt[i])

            n_pred += len(pred_pt[i])
            n_pred_null += length_of_null_quads(pred_pt[i])
            for p in pred_pt[i]:
                if p[0] != '' and p[1] != '':
                    n_tp += check_label(p, gold_pt[i], sbert_model)
            
            
            ####### ........ #######



    n_gold = n_gold - n_gold_null
    n_pred = n_pred - n_pred_null

    print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    print(f"number of null spans: {n_pred_null}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores, n_pred_null


def compute_scores(pred_seqs, gold_seqs, sents, io_format, task):
    """
    compute metrics for multiple tasks
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_predictions = [], []

    for i in range(num_samples):
        if io_format == 'annotation':
            gold_list = extract_spans_annotation(task, gold_seqs[i])
            pred_list = extract_spans_annotation(task, pred_seqs[i])
        elif io_format == 'extraction':
            gold_list = extract_spans_extraction(task, gold_seqs[i], 'gold')
            print('Original pred list: ', pred_seqs[i])
            pred_list = extract_spans_extraction(task, pred_seqs[i], 'pred')
            print()

        all_labels.append(gold_list)
        all_predictions.append(pred_list)

    print("\nResults of raw output")
    raw_scores, n_pred_null = compute_f1_scores(all_predictions, all_labels)

    print('Number of predicted null triplets: ', n_pred_null)
    print(raw_scores)
    # # fix the issues due to generation
    # all_predictions_fixed = fix_pred_with_editdistance(all_predictions, sents, task)
    # print("\nResults of fixed output")
    # fixed_scores = compute_f1_scores(all_predictions_fixed, all_labels)
    # print(fixed_scores)
    return raw_scores, all_labels, all_predictions
