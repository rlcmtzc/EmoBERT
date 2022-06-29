from transformers import RobertaTokenizer, RobertaModel
import torch

class EmoBERT(torch.nn.Module):
    def __init__(self, n_output_classes: int = 7):
        super(EmoBERT, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, n_output_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Sigmoid()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output