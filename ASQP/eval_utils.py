# -*- coding: utf-8 -*-

# This script handles the decoding functions and performance measurement

import re

import sklearn
import torch
from data_utils import aspect_cate_list
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sentiment_word_list = ["positive", "negative", "neutral"]
opinion2word = {"great": "positive", "bad": "negative", "ok": "neutral"}
opinion2word_under_o2m = {
    "good": "positive",
    "great": "positive",
    "best": "positive",
    "bad": "negative",
    "okay": "neutral",
    "ok": "neutral",
    "average": "neutral",
}
numopinion2word = {"SP1": "positive", "SP2": "negative", "SP3": "neutral"}


def extract_spans_para(task, seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split("[SSEP]")]

    # Replace subword of "[[SSEP]]" -> ''
    list_Words = ["[SSEP", "[SSE", "[SS", "[S", "["]
    big_regex = re.compile("|".join(map(re.escape, list_Words)))
    sents = [big_regex.sub("", s) for s in sents]

    if task == "aste":
        for s in sents:
            # It is bad because editing is problem.
            try:
                c, ab = s.split(" because ")
                c = opinion2word.get(c[6:], "nope")  # 'good' -> 'positive'
                a, b = ab.split(" is ")
            except ValueError:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                a, b, c = "", "", ""
            quads.append((a, b, c))
    elif task == "tasd":

        for s in sents:
            # food quality is bad because pizza is bad.

            # print(f'\nSentence: {s}')
            try:
                # ac_sp, at_sp = s.split(' because ')
                # Check sentence is consist two or more than 'because'
                s_tmp = s.split(" because ")
                # ac_sp, at_sp = s_tmp
                if len(s_tmp) > 2:
                    ac_sp = s_tmp[0]
                    at_sp = " because ".join(sent for sent in s_tmp[1:])
                else:
                    ac_sp, at_sp = s_tmp

                #  Extract sentiment level in statement 1 of sentence
                ac, sp = ac_sp.split(" is ")

                # Check statement 2 consist two or more than "is"
                at_sp2 = at_sp.split(" is ")
                if len(at_sp2) > 2:
                    at = " is ".join(sent for sent in at_sp2[:-1])
                    at = at.strip()
                    sp2 = at_sp2[-1]
                else:
                    at, sp2 = at_sp2
                if "is" in sp2:
                    sp2 = sp2.replace("is", "").strip()

                sp = opinion2word.get(sp, "nope")
                sp2 = opinion2word.get(sp2, "nope")
                if sp != sp2:
                    print(
                        f"Sentiment polairty of AC({sp}) and AT({sp2}) is inconsistent!"
                    )
                    print(f"Sentence: {s}\n")

                # if the aspect term is implicit
                if at.lower() == "it":
                    at = "NULL"

                # print(f'AC({ac}), AT({at}), SP({sp})\n')
            except ValueError:
                # print(f'In {seq_type} seq, cannot decode: {s}')
                ac, at, sp = "", "", ""

            quads.append((ac, at, sp))

    elif task == "asqp":
        for s in sents:
            # food quality is bad because pizza is over cooked.
            try:
                ac_sp, at_ot = s.split(" because ")
                ac, sp = ac_sp.split(" is ")
                at, ot = at_ot.split(" is ")

                # if the aspect term is implicit
                if at.lower() == "it":
                    at = "NULL"
            except ValueError:
                try:
                    # print(f'In {seq_type} seq, cannot decode: {s}')
                    pass
                except UnicodeEncodeError:
                    # print(f'In {seq_type} seq, a string cannot be decoded')
                    pass
                ac, at, sp, ot = "", "", "", ""

            quads.append((ac, at, sp, ot))
    else:
        raise NotImplementedError
    return quads


def length_of_null_quads(list_span: list):
    return sum(
        1 for x in list_span if (x[0] == "" and x[1] == "" and x[2] == "")
    )


def check_label(pred_i, gold):

    # Option 1: sentence bert - encoder transformers
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

    # Option 2: statistic technique: tf-idf
    vectorizer = TfidfVectorizer()

    for g in gold:
        if pred_i == g:
            return 1
        else:
            # Option 1
            encode_gold = sbert_model.encode(g[1])
            encode_pred = sbert_model.encode(pred_i[1])

            cos = torch.nn.CosineSimilarity(dim = 0, eps=1e-6)
            sim = cos(torch.Tensor(encode_gold), torch.Tensor(encode_pred)).item()
            # Option 2
            # vectors = vectorizer.fit_transform([g[1], pred_i[1]])
            # sim = cosine_similarity(vectors)
            # sim = sim[0, 1]

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

    # Option 1: Sentence Bert
    # sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

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
            if p[0] != "" and p[1] != "":
                n_tp += check_label(p, gold_pt[i])

        ####### ........ #######

    n_gold = n_gold - n_gold_null
    n_pred = n_pred - n_pred_null

    print(
        f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}"
    )
    print(f"number of null spans: {n_pred_null}")
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision != 0 or recall != 0
        else 0
    )
    scores = {"precision": precision, "recall": recall, "f1": f1}

    return scores


def compute_scores(pred_seqs, gold_seqs, sent, task="asqp"):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(task, gold_seqs[i], "gold")
        pred_list = extract_spans_para(task, pred_seqs[i], "pred")

        all_labels.append(gold_list)
        all_preds.append(pred_list)

    print("\nResults:")
    scores = compute_f1_scores(all_preds, all_labels)
    print(scores)

    return scores, all_labels, all_preds
