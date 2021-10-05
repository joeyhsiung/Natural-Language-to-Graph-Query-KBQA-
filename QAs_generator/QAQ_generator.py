import time
import re
import pandas as pd
import wikipedia

import spacy
from fuzzysearch import find_near_matches
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
from transformers import AutoModelWithLMHead, AutoTokenizer

from multiHop_QA.configures import Config_path

c = Config_path()
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

# read triple files
wiki_triples = pd.read_csv(c.triple)
# Specify the title of the Wikipedia page
wiki = wikipedia.page('Rolls-Royce Trent')
# Extract the plain text content of the page
text = wiki.content
source_entities = wiki_triples['source'].to_list()
answer_entities = wiki_triples['end'].to_list()

# Clean text
text = re.sub(r'==.*?==+', '', text)
# text = text.replace('\n', '')
s = time.time()
sent_list_clean = []
sent_list_raw = []
doc = nlp(text)
for sent in doc.sents:
    # for token in sent:
    text = sent.lemma_.replace('\s+', "")
    text = text.split('\n')
    text = list(filter(lambda x: x != "", text))
    text_raw = sent.text
    text_raw = text_raw.replace('\s+', "").split('\n')
    text_raw = list(filter(lambda x: x != "", text_raw))
    for i in range(len(text)):
        if len(text[i]) > 10:  # remove < 10 length sents
            # print(sent.lemma_)
            t = re.sub("[^a-zA-Z0-9_. ,:;]", " ", text[i])
            # t = re.sub("^[^a-zA-Z]+", "", t)
            t = " ".join(t.split())
            # t = t.lower()

            r = re.sub("[^a-zA-Z0-9_. ,:;]", " ", text_raw[i])
            r = " ".join(r.split())
            # r = r.lower()

            sent_list_clean.append(t)  # add if condition
            sent_list_raw.append(r)
            # print('================================')
print('Process Time:', time.time() - s, 's')


def make_unique(doc):
    unique = []
    for ent in doc:
        entity = nlp(ent)
        entity_lemma = " ".join([token.lemma_ for token in entity])
        entity_lemma = re.sub("[^a-zA-Z\d]", " ", entity_lemma)
        entity_lemma = re.sub(' +', ' ', entity_lemma)
        # entity_lemma = entity_lemma.lower()
        if len(entity_lemma) > 2 and entity_lemma not in unique:
            unique.append(entity_lemma)
    return unique


def match_end(sent_list, sent_list_raw, entities):
    data = []
    leng = len(sent_list)
    for i in range(leng):
        end = {}
        for ent in entities:
            matches = find_near_matches(ent, sent_list[i], max_l_dist=1)
            temp = [(m.start, m.end) for m in matches
                        if re.findall('\d+', ent) == re.findall('\d+', m.matched)]
            if temp:
                end[ent] = temp
        if end:
            data.append([sent_list_raw[i],sent_list[i], end])
    return data


def match_source(QAs_list, entities):
    data = []
    data_raw = []
    for q in QAs_list:
        for ent in entities:
            matches = find_near_matches(ent, q[0], max_l_dist=1)
            if matches:
                ent_match = [{ent:(m.start,m.end)} for m in matches
                             if re.findall('\d+', ent) == re.findall('\d+', m.matched)]
                if ent_match:
                    q[1]['entity'] = ent_match
                    data.append(q)
                    # print(q)
    return data


def question_generator(matched,max_length = 256):
    QAs = []
    QAs_ctx = []
    for i in matched:
        temp = []
        for k,v in i[2].items():
            dic = {}
            input_text = "answer: %s  context: %s </s>" % (k, i[0])
            print(k,i[0])
            features = tokenizer([input_text], return_tensors='pt')
            output = model.generate(input_ids=features['input_ids'],
                                    max_length=max_length)
            questions = tokenizer.decode(output[0]).replace("<pad> question: ","").replace("</s>","")
            temp.append(questions)
            dic['answer'] = k
            QAs.append([questions,dic])
            QAs_ctx.append([questions, dic,{'context':i[0]}])
        # i[1]['questions'] = temp
    return QAs_ctx,QAs


# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
#
# model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl")
# features = tokenizer(['Python is a programming language. It is developed by <hl> Guido Van Rossum <hl>. </s>'], return_tensors='pt')
# output = model.generate(input_ids=features['input_ids'])
# questions = tokenizer.decode(output[0])
# print(questions)


if __name__ == '__main__':
    source_entities = make_unique(source_entities)
    answer_entities = make_unique(answer_entities)

    # matched_source = match_entity(sent_list, source_entities,name = 'source')
    matched_answer = match_end(sent_list_clean, sent_list_raw, answer_entities)
    # print(matched_answer)
    QAs_ctx,QAs = question_generator(matched_answer,max_length = 128)
    print(QAs_ctx)
    matched_source = match_source(QAs, source_entities)
    # print(QAs)
    print(matched_source)
    print(len(matched_source))

    # neo4j query
    from neo4j import GraphDatabase
    uri = "neo4j://localhost:7687"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "123"))

    cypher_query = '''
    MATCH  res = (p)-[*..3]->(m)
    WHERE p.name = '{0}' and m.name = '{1}'
    RETURN res
    '''
    # in 3 hop relations
    res = []
    with driver.session(database="neo4j") as session:
        for i in matched_source:
            question = i[0].lower()
            answer = i[1]['answer'].lower()
            for j in i[1]['entity']:
                for ent, pos in j.items():
                    # print(ent, pos)
                    results = session.read_transaction(
                        lambda tx: tx.run(cypher_query.format(ent, answer)).data())
                    for record in results:
                        rels = [re.sub("[^a-zA-Z\d]", " ", rel) for rel in record['res']
                                if type(rel) is not dict]
                        # insert best relation selection code
                        res.append([question,{'answer':answer,'entity':ent,'pos':pos,'rels':rels}])
                        print(question,{'answer':answer,'entity':ent,'pos':pos,'rels':rels})
                        # print(rels)
    driver.close()
    print(res)











# print(matched_source)
# print(matched_answer)
# random.shuffle(answer_entities)

# my_string = "It has the Trent family three shaft architecture, " \
#             "it has a 280 cm (110 in) fan"
# matches = find_near_matches('it', my_string, max_l_dist=1)
# print(matches)
# print([my_string[m.start:m.end] for m in matches])

# result=[]
# for entity in answer_entities:
#     entity_ = nlp(entity)
#     for token in entity_:
#         print(token.lemma_)

# def findcarfeatures(features, document, match=80): #features=["hello I am"] document ="hello We are. I was going to talk with you"
#     result=[]
#     for feature in features:
#         lenfeature = len(feature.split(" "))
#         word_tokens = nltk.word_tokenize(document)
#         filterd_word_tokens = [w for w in word_tokens if not w in stop_words]
#         for i in range (len(word_tokens)-lenfeature+1):
#             wordtocompare = ""
#             j=0
#             for j in range(i, i+lenfeature):
#                 if re.search(r'[,!?{}\[\]\"\"\'\']',word_tokens[j]):
#                     break
#                 wordtocompare = wordtocompare+" "+word_tokens[j].lower()
#             wordtocompare.strip()
#             if not wordtocompare=="":
#                 if(fuzz.ratio(wordtocompare,feature.lower())> match):
#                     result.append([wordtocompare,feature,i,j])
#     return result

# import spacy
# nlp = spacy.load("en_core_web_sm")
# doc1 = nlp('I like dogs')
# doc2 = nlp('I like cats')
# doc1.similarity(doc2)
