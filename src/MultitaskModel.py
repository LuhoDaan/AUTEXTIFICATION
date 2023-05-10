import pytorch_lightning as pl
from pytorch_lightning.profilers import PyTorchProfiler
from torch import nn, optim
import torchmetrics
import transformers
from torch.utils.data import DataLoader

from stats import*
from transformers import DistilBertModel, BertTokenizer
from torchmetrics.classification import BinaryConfusionMatrix
import seaborn as sns
from dataloader import FullDataloader
from callbacks import *


class NN(pl.LightningModule):
    def __init__(self,num_classes, learning_rate):
        super().__init__()
        #self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.bcm = BinaryConfusionMatrix(device=self.device)
        #self.classification_head = nn.Linear(768, num_classes)
        self.heads_dict = {
            'origin': nn.Linear(768, 2),
            'language': nn.Linear(768, 2),
            'bot': nn.Linear(768, 7)
        }
        self.loss_dict = {
            'origin': nn.CrossEntropyLoss(),
            'language': nn.CrossEntropyLoss(),
            'bot': nn.CrossEntropyLoss()
        }
        #self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        #create metrics for each head considering that origin and language are binary tasks and bot is a multiclass task.
        self.accuracy_dict = {
            'origin': torchmetrics.Accuracy(task='binary', num_classes=1),
            'language': torchmetrics.Accuracy(task='binary', num_classes=1),
            'bot': torchmetrics.Accuracy(task='multiclass', num_classes=7)
        }
        self.f1_dict = {
            'origin': torchmetrics.F1Score(task='binary', num_classes=1, average='macro'),
            'language': torchmetrics.F1Score(task='binary', num_classes=1, average='macro'),
            'bot': torchmetrics.F1Score(task='multiclass', num_classes=7, average='macro')
        }
        self.validation_step_outputs = []
        self.validation_targets = []
        #create an accuracy that considers if the concatenation of the three heads is correct
        self.accuracy = torchmetrics.Accuracy(task='binary', num_classes=1)
        self.f1 = torchmetrics.F1Score(task='binary', num_classes=1, average='macro')


        #self.accuracy = torchmetrics.Accuracy(task=task)
        #self.f1 = torchmetrics.F1Score(task='binary', num_classes=num_classes, average='macro')

    def forward(self, input_encodings):
        input_ids, attention_mask = input_encodings.input_ids, input_encodings.attention_mask
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]

        #compute logits for each different head as a dictionary
        logits_dict = {}
        for head in self.heads_dict:
            logits_dict[head] = self.heads_dict[head](pooled_output)
        #logits = self.classification_head(pooled_output)
        return logits_dict

    def training_step(self, batch, batch_idx):
        input_encodings, origin_lb, bot_lb, lang_lb = batch
        labels_dict = {
            'origin': origin_lb,
            'bot': bot_lb,
            'language': lang_lb
        }

        logits_dict = self.forward(input_encodings)
        # compute loss for each different logit as a dictionary
        loss_dict = {}
        for head, logits in logits_dict.items():
            valid_mask = labels_dict[head] != -1
            task_bs = valid_mask.sum()
            if task_bs > 0:
                loss_dict[head] = self.loss_dict[head](logits[valid_mask], labels_dict[head][valid_mask])

        #log each loss
        for loss in loss_dict:
            self.log(f'train_{loss}_loss', loss_dict[loss], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        loss = sum(loss_dict.values())
        #log full loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch, batch_idx):
        input_encodings, origin_lb, bot_lb, lang_lb = batch
        labels_dict = {
            'origin': origin_lb,
            'bot': bot_lb,
            'language': lang_lb
        }

        logits_dict = self.forward(input_encodings)
        # compute loss for each different logit as a dictionary
        loss_dict = {}
        for head, logits in logits_dict.items():
            valid_mask = labels_dict[head] != -1
            task_bs = valid_mask.sum()
            if task_bs > 0:
                loss_dict[head] = self.loss_dict[head](logits[valid_mask], labels_dict[head][valid_mask])

        # log each loss
        for loss in loss_dict:
            self.log(f'val_{loss}_loss', loss_dict[loss], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        loss = sum(loss_dict.values())
        # log full loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #compute preds for each class as a dictionary, consider the valid mask
        preds_dict = {}
        for head in self.accuracy_dict:
            preds_dict[head] = torch.argmax(logits_dict[head], dim=1)
        #compute metrics for each different prediction, consider the valid mask
        for head in self.accuracy_dict:
            valid_mask = labels_dict[head] != -1

            task_bs = valid_mask.sum()
            if task_bs > 0:
                accuracy = self.accuracy_dict[head](preds_dict[head][valid_mask], labels_dict[head][valid_mask])
                f1_score = self.f1_dict[head](preds_dict[head][valid_mask], labels_dict[head][valid_mask])
                self.log_dict({f'val_{head}_accuracy': accuracy, f'val_{head}_f1': f1_score}, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                #compute full accuracy and f1
                preds = torch.cat([preds_dict[head][valid_mask] for head in preds_dict])

                labels = torch.cat([labels_dict[head][valid_mask] for head in labels_dict])

                self.validation_step_outputs.append(preds)
                self.validation_targets.append(labels)
        return loss


    def _common_step(self, batch, batch_idx):
        input_encodings, origin_lb,bot_lb,lang_lb = batch
        labels_dict = {
            'origin': origin_lb,
            'bot': bot_lb,
            'language': lang_lb
        }
        logits_dict = self.forward(input_encodings)
        #compute loss for each different logit as a dictionary
        loss_dict = {}
        for head in self.loss_dict:

            #if head is 'bot', mask the target to ignore the ones with value = -1
            if head == 'bot':
                mask = labels_dict[head] != -1
                loss_dict[head] = self.loss_dict[head](logits_dict[head][mask], labels_dict[head][mask])
            else:
                loss_dict[head] = self.loss_dict[head](logits_dict[head], labels_dict[head])


            #self.log_dict({f'{head}_loss': loss_dict[head]}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #loss = self.loss_fn(logits, labels)

        return loss_dict, logits_dict, labels_dict
    '''
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        x = x.reshape(x.shape[0], -1)
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        return preds
    '''

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=5e-6)



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
profiler = PyTorchProfiler()
log_dir = '/content/drive/MyDrive/tb_logs'
#logger = TensorBoardLogger(log_dir, name='Autextification_model')
model = NN(2, 5e-6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_data = FullDataloader('train', '/home/matteo/PycharmProjects/AUTEXTIFICATION/src/cache', tokenizer, None, None, task ='multi')
val_data = FullDataloader('validation', '/home/matteo/PycharmProjects/AUTEXTIFICATION/src/cache', tokenizer, None, None, task='multi')
test_data = FullDataloader('test', '/home/matteo/PycharmProjects/AUTEXTIFICATION/src/cache', tokenizer, None, None, task ='multi')

train_loader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=train_data.collate)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, collate_fn=val_data.collate)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=test_data.collate)

trainer = pl.Trainer(
    max_epochs=5,
    callbacks=[PrintingCallback(), EarlyStopping(monitor = 'val_loss_epoch',patience=2)],
fast_dev_run=8)
trainer.fit(model, train_loader, val_loader)
trainer.validate(model, val_loader)