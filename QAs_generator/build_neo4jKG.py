import re
import spacy
from multiHop_QA.configures import Config_path
import pandas as pd
from tqdm import tqdm
from py2neo import Graph,Node
from neo4j import GraphDatabase


c = Config_path()
nlp = spacy.load("en_core_web_sm")
##################################
## Create Neo4j Knowledge Graph ##
##################################

# push triple csv data to neo4j graph database
# initialize local database
graph = Graph("http://localhost:7474",user='neo4j',password='123')
# read triple files
wiki_triples = pd.read_csv(c.triple)

# clean text
num_cols = wiki_triples.shape[1]
for col in range(num_cols):
    # tokens = nlp(list(wiki_triples.iloc[:,col]))
    col_list = list(wiki_triples.iloc[:,col])
    col_list_lemma = []
    for i in col_list:
        temp_doc = nlp(str(i))
        token_lemma = str()
        for token in temp_doc.sents:
            token_lemma += token.lemma_
        col_list_lemma.append(token_lemma)
    temp_re = [re.sub("[^a-zA-Z\d]", " ", str(i).lower()) for i in col_list_lemma]
    wiki_triples.iloc[:, col] = temp_re
    # print(temp_re)


# replace space to underline
wiki_triples['relation'] = wiki_triples['relation'].str.replace(' ','_',regex=True)
wiki_triples['label'] = wiki_triples['label'].str.replace(' ','_',regex=True)


# create source nodes
temp = []
for i,row in tqdm(wiki_triples.iterrows()):
    start_node = Node(row['label'],name=row['source'])
    end_node = Node(row['label'],name=row['end'])
    if row['source'] not in temp:
        graph.create(start_node)
        temp.append(row['source'])
    # if row['end'] not in temp:
    #     graph.create(end_node)
    #     temp.append(row['end'])
    # graph.create(Relationship(start_node, row['relation'], end_node))


# create relations with end nodes
uri = "neo4j://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "123"))

def create(tx, label, rel, start, end):
    tx.run("MATCH (a:"+label+") WHERE a.name = $start "
           "CREATE (a)-[:"+rel+"]->(:"+label+" {name: $end})",
           start=start, end=end)

with driver.session() as session:
    for i,row in tqdm(wiki_triples.iterrows()):
        session.write_transaction(create,row['label'],row['relation'],row['source'],row['end'])
    print('pushing data to neo4j is done')
driver.close()
