import torch
import configparser
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from rich import print
import sys

from EmoBERT import EmoBERT
from utils.utils import *
from utils.preprocessor import *

if len(sys.argv) < 2:
    raise ValueError("Pass a config file as the first argument!")

config = configparser.ConfigParser()
config.read(sys.argv[1])
config_data = dict(config["TRAIN_MODEL"])

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
test_data = read_data(config_data["test_data"])
train_data = read_data(config_data["train_data"])
class_and_emoji = get_classes(train_data)

if "PREPROCESSING" in config and "steps" in config["PREPROCESSING"]:
    test_data = apply_preprocessing(test_data, [function_mapping[preprocessing_step] for preprocessing_step in config["PREPROCESSING"]["steps"].replace(" ", "").split(",")])
    train_data = apply_preprocessing(train_data, [function_mapping[preprocessing_step] for preprocessing_step in config["PREPROCESSING"]["steps"].replace(" ", "").split(",")])


train_dataset = tokenize_data_to_torch_dataset_roberta(train_data, "tweet", "emoji_class", tokenizer)
test_dataset = tokenize_data_to_torch_dataset_roberta(test_data, "tweet", "emoji_class", tokenizer)

batch_size = int(config_data["batchsize"])
dataloader_train = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
dataloader_test = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

model_name = config_data["modelname_to_save"]
model = EmoBERT(int(config_data["n_classes"]))


optimizer = AdamW(params =  model.parameters(), lr=float(config_data["learning_rate"]))
epochs = int(config_data["epochs"])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

results = train_model_roberta(model, device, optimizer, dataloader_train, dataloader_test, epochs, model_name)
   
for epoch, result in results.items():
    print_scores_with_emojis(result, class_and_emoji)