import pandas as pd
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from rich.progress import track
from sklearn.metrics import precision_score, accuracy_score, recall_score
import json



def read_data(path_to_data: str) -> pd.DataFrame:
    return pd.read_pickle(path_to_data)

def get_classes(data_frame: pd.DataFrame) -> Dict[str, str]:
    unique_labels_collumns = data_frame.drop_duplicates(subset = ["emoji_class"])
    return dict(zip(unique_labels_collumns["emoji_class"].tolist(),unique_labels_collumns["emoji"].tolist()))

def plot_data_distribution(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    values_count_train = pd.value_counts(train_data['emoji_class'])
    values_count_test = pd.value_counts(test_data['emoji_class'])
    df = pd.DataFrame({'train': values_count_train,
                       'test': values_count_test})
    df.plot.bar(rot=1, xlabel="Emoji Class", ylabel="Emoji Count", title="Train and Test Data distribution")
    plt.savefig("train_test_data_distribution.png")

def tokenize_data_to_torch_dataset_roberta(data: pd.DataFrame, train_column: str, target_column: str,  tokenizer: BertTokenizer) -> TensorDataset:
    tokenized_data = tokenizer.batch_encode_plus(batch_text_or_text_pairs = data[train_column].values, 
                                                 add_special_tokens=True, 
                                                 return_attention_mask=True, 
                                                 pad_to_max_length=True, 
                                                 max_length=128, 
                                                 return_tensors='pt',
                                                 return_token_type_ids=True
                                                )
    input_ids_data = tokenized_data['input_ids']
    attention_masks_data = tokenized_data['attention_mask']
    token_type_ids = tokenized_data["token_type_ids"]
    labels_data = torch.tensor(data[target_column].values)
    return TensorDataset(input_ids_data, attention_masks_data, token_type_ids,  labels_data)

def evaluate_model_roberta(model: torch.nn.Module, device: torch.device , dataloader: DataLoader, st_progress_bar = None) -> Tuple[np.ndarray]:
    model.eval()
    loss_val_total = 0
    y_pred, y_true = [], []
    loss_function = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        # for i,batch in enumerate(dataloader):
        i = 0
        for batch in track(dataloader, description=f"Evaluating Model"):
            if st_progress_bar is not None:
                st_progress_bar.progress((i+1)/len(dataloader))
                i += 1

            ids = batch[0].to(device, dtype = torch.long)
            mask = batch[1].to(device, dtype = torch.long)
            token_type_ids = batch[2].to(device, dtype=torch.long)
            targets = batch[3].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            
            loss = loss_function(outputs, targets)
            loss_val_total += loss.item()
            propability_class_wise, logits = torch.max(outputs.data, dim=1)

            logits = logits.detach().cpu().numpy()
            y_pred.append(logits)
            y_true.append(targets.cpu().numpy())
    
    loss_val_avg = loss_val_total/len(dataloader) 
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
   
    y_true = y_true.flatten()   

    return loss_val_avg, y_pred, y_true, propability_class_wise

def claculate_metrics_dictionary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    return {
            "precision_score": precision_score(y_true, y_pred, average=None).tolist(),
            "precision_score_macro": precision_score(y_true, y_pred, average="macro").tolist(),
            "precision_score_micro": precision_score(y_true, y_pred, average="micro").tolist(),
            "accuracy_score": accuracy_score(y_true, y_pred).tolist(),
            "recall_score": recall_score(y_true, y_pred, average=None).tolist(),
            "recall_score_macro": recall_score(y_true, y_pred, average="macro").tolist(),
            "recall_score_micro": recall_score(y_true, y_pred, average="micro").tolist()
        }


def train_model_roberta(model: torch.nn.Module, device: torch.device, optimizer, dataloader_train: DataLoader, dataloader_test: DataLoader, epochs: int, model_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    results = {}
    loss_function = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_train_total = 0

        for data in track(dataloader_train, description=f"Finetuning Model Epoch {epoch}"):

            ids = data[0].to(device, dtype = torch.long)
            mask = data[1].to(device, dtype = torch.long)
            token_type_ids = data[2].to(device, dtype = torch.long)
            targets = data[3].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            loss = loss_function(outputs, targets)
            loss_train_total += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)

            optimizer.zero_grad()
            loss.backward()
            # # When using GPU
            optimizer.step()


        torch.save(model.state_dict(), f'models/finetuned_{model_name}_epoch_{epoch}.model')
        
        loss_train_avg = loss_train_total/len(dataloader_train)
        print(f"Training loss after Epoch {epoch}: {loss_train_avg:.3f}")
        
        try:
            test_loss, y_pred, y_true, _ = evaluate_model_roberta(model, device, dataloader_test)
        except:
            print("Some error in evaluate")
            continue
            
        results[epoch] = {"train_loss": loss_train_avg,
                          "test_loss": test_loss}
        print(f"Testing loss after Epoch {epoch}: {test_loss:.3f}")
        for key, value in claculate_metrics_dictionary(y_true, y_pred).items():
            print(f"    {key}: {value}")
        print("-"*50)
        results[epoch].update(claculate_metrics_dictionary(y_true, y_pred))

        # save results as json
        with open(f"results/{model_name}_results.json", "w") as f:
            json.dump(results, f, indent=4)

    return results

def print_scores_with_emojis(metrics_dictionary: Dict[str, np.ndarray], class_and_emoji: Dict[str, str]) -> None:
    for metric, score in metrics_dictionary.items():
        if isinstance(score, np.ndarray):
            print(f"{' '.join(metric.split('_')).title()} Class wise:")
            for i, emoji in enumerate(class_and_emoji.values()):
                try:
                    print(f"  {emoji}: {score[i]}")
                except:
                    print(f"  {emoji}: N/A")
        else:
            print(f"{' '.join(metric.split('_')).title()}: {score}")
    