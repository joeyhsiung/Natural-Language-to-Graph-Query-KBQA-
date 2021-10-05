import os
from multiHop_QA.configures import Config_path
from transformers import AutoTokenizer, AutoModelForMaskedLM

config = Config_path()
os.chdir(config.parent_path)


def read_data():
    pass


model_name = "roberta-base"

query_tokenizer = AutoTokenizer.from_pretrained(model_name)
passage_tokenizer = AutoTokenizer.from_pretrained(model_name)
query_model = AutoModelForMaskedLM.from_pretrained(model_name)
passage_model = AutoModelForMaskedLM.from_pretrained(model_name)

query_model.train()
passage_model.train()





# with open(config.train_web, encoding='utf-8') as f:
#     data = json.load(f)
#
# questions,ents_Q,ents_KG,queries,answers = [],[],[],[],[]
# for i in data["Questions"]:
#     questions.append(i["ProcessedQuestion"])
#     ents_Q.append(i["Parses"][0]["PotentialTopicEntityMention"])
#     ents_KG.append(i["Parses"][0]["TopicEntityName"])
#     queries.append(i["Parses"][0]["InferentialChain"])
    # answers.append(i["Parses"][0]["Answers"])

#
# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
