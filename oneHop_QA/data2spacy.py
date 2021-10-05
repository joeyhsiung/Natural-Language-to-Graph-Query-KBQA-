import pandas as pd
import spacy
import re
from tqdm import tqdm
from spacy.tokens import DocBin
from multiHop_QA.configures import Config_path, Config_output_path

c = Config_path()
c_output = Config_output_path()


def data2jason(input_data, q_col='question', ent_col='entity',
               label_name='SimpleQuestions'):  # input data should be dataframe
    data = []
    for _, item in input_data.iterrows():
        q = item[q_col]
        pattern = item[ent_col]
        s, e = 0, 0
        try:
            for match in re.finditer(pattern, q):
                s = match.start()
                e = match.end()

            entity = (s, e, label_name)
            temp = (item[q_col], {'entities': [entity]})
            data.append(temp)
        except:
            pass
    return data


def jason2spacy(data, output_name="dev.spacy"):
    nlp = spacy.blank("en")  # load a new spacy model
    db = DocBin()  # create a DocBin object
    total = 0
    for text, annot in tqdm(data):  # data in previous format
        doc = nlp.make_doc(text)  # create doc object from text
        ents = []
        for start, end, label in annot["entities"]:  # add character indexes
            span = doc.char_span(start, end, label=label)
            if span:
                total += 1
                ents.append(span)
        doc.ents = ents  # label the text with the ents
        db.add(doc)
    db.to_disk(c_output.ner_data_path + output_name)  # save the docbin object


# import json
# with open('train_jason.json', 'w', encoding='utf-8') as f:
#     json.dump(train_jason, f, ensure_ascii=False, indent=2)
#
# with open('test_jason.json', 'w', encoding='utf-8') as f:
#     json.dump(test_jason, f, ensure_ascii=False, indent=2)


# read train data
train_combination = pd.read_csv(c.train_combination_path)
train_combination = train_combination[train_combination.entity != '*']
# read test data
test_combination = pd.read_csv(c.test_combination_path)
test_combination = test_combination[test_combination.entity != '*']
# read validation(dev) data
dev_wiki = pd.read_csv(c.wiki_dev_path)
dev_wiki = dev_wiki[dev_wiki.entity != '*']


# extract train_entity column to list
train_entities = train_combination['entity'].unique().tolist()
# extract test_entity column to list
test_entities = test_combination['entity'].unique().tolist()
# extract dev_entity column to list
dev_entities = dev_wiki['entity'].unique().tolist()

train_jason = data2jason(train_combination)
test_jason = data2jason(test_combination)
dev_jason = data2jason(dev_wiki)

jason2spacy(train_jason, output_name="train.spacy")
jason2spacy(test_jason, output_name="test.spacy")
jason2spacy(dev_jason, output_name="dev.spacy")
