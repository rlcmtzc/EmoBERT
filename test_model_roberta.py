import torch
import configparser
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from rich import print

from EmoBERT import EmoBERT
import sys

from utils.utils import *
from utils.preprocessor import *

if len(sys.argv) < 2:
    raise ValueError("Pass a config file as the first argument!")

config = configparser.ConfigParser()
config.read(sys.argv[1])
config_data = dict(config["TEST_MODEL"])

tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
test_data = read_data(config_data["test_data"])

if "PREPROCESSING" in config and "steps" in config["PREPROCESSING"]:
    test_data = apply_preprocessing(test_data, [function_mapping[preprocessing_step] for preprocessing_step in config["PREPROCESSING"]["steps"].replace(" ", "").split(",")])

class_and_emoji = get_classes(test_data)

test_dataset = tokenize_data_to_torch_dataset_roberta(test_data, "tweet", "emoji_class", tokenizer)

batch_size = int(config_data["batchsize"])
dataloader_test = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

model = EmoBERT(int(config_data["n_classes"]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.load_state_dict(torch.load(config_data["modelname_to_load"], map_location=torch.device('cpu')))

loss_val_avg, y_pred, y_true, propability_class_wise = evaluate_model_roberta(model, device, dataloader_test)

results = claculate_metrics_dictionary(y_true, y_pred)
print_scores_with_emojis(results, class_and_emoji)
