import sys, os

sys.path.append(os.pardir)
import re
import spacy
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import tqdm

# parent_dir = os.path.dirname(os.getcwd())
# NER = spacy.load(os.path.join(parent_dir, 'models/NER_output_train-dev/model-best')) #load the NER best model
# CATS = spacy.load(os.path.join(parent_dir, 'models/cat_output_train-train_268_0719/model-best')) #load the text_cat best model


def test_NER(doc, NER):
    doc_ner = NER(doc)
    predict_ent = doc_ner.ents[0]  # get predicted entity string
    return predict_ent


def mask_doc(doc, ent):
    m = re.search(str(ent), doc)
    idxs = m.span()
    doc_masked = doc[0:idxs[0]] + '<e>' + doc[idxs[1]:-1]
    return doc_masked


def test_CATS(doc_masked, CATS):
    doc_cats = CATS(doc_masked)
    pedict_relation = max(doc_cats.cats, key=doc_cats.cats.get)
    return pedict_relation


def test_pipe(doc, NER, CATS):
    predict_ent = test_NER(doc, NER)
    predict_masked = mask_doc(doc, predict_ent)
    pedict_relation = test_CATS(predict_masked, CATS)
    return pedict_relation


def NER_metrics(test_q, test_ents, NER):
    total_num = len(test_q)
    precision = 0
    recall = 0
    f1 = 0
    for idx, i in enumerate(test_q):
        test_ent = test_ents[idx].split()
        true_num = len(test_ent)

        doc_ner = NER(i)
        predict_ent = doc_ner.ents[0]
        predict_ent = str(predict_ent).split()
        predict_num = len(predict_ent)

        correct_num = len([w for w in test_ent if w in predict_ent])

        p = correct_num / predict_num
        precision += p
        r = correct_num / true_num
        recall += r
        if p + r > 0:
            f1 += (2 * p * r) / (p + r)
        else:
            f1 += 0
    print(precision / total_num, recall / total_num, f1 / total_num)


def CATS_metrics(test_q_space, test_r, CATS):
    predict_r = []
    for i in test_q_space:
        doc_cats = CATS(i)
        pedict_relation = max(doc_cats.cats, key=doc_cats.cats.get)
        predict_r.append(pedict_relation)
    f1_CATS = precision_recall_fscore_support(test_r, predict_r, average='weighted')
    print(f1_CATS)