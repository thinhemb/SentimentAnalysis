from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW
from dataloader import *
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import os
import torch
import numpy as np 
from clearml import Task,Logger
from tensorboardX import SummaryWriter
task = Task.init(project_name='SA_comment_VN', task_name='task_1')
S_writer=SummaryWriter('run/SA_comment_VN')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    F1_score = f1_score(pred_flat, labels_flat, average='macro')
    
    return accuracy_score(pred_flat, labels_flat), F1_score


config = RobertaConfig.from_pretrained(
    "./main/transformers/PhoBERT_base_transformers/config.json", from_tf=False, num_labels = 6, output_hidden_states=False,
)
BERT_SA = RobertaForSequenceClassification.from_pretrained(
    "./main/model.bin",
    config=config
)

BERT_SA.cuda()
device = 'cuda'
epochs = 8

param_optimizer = list(BERT_SA.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False)
Best_val_f1=0
Best_val_acc=0

for epoch in range(0, epochs):
    print('======================== Epoch {:} / {:} ======================== '.format(epoch + 1, epochs))
    print('============Training============')
   
    total_loss = 0
    BERT_SA.train()
    train_accuracy = 0
    nb_train_steps = 0
    train_f1 = 0
    
    for step, batch in tqdm(enumerate(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        BERT_SA.zero_grad()
        outputs = BERT_SA(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask, 
            labels=b_labels)
        loss = outputs[0]
        S_writer.add_scalar('Train_loss', loss,epoch)
        total_loss += loss.item()
        
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        label_ids = np.int64(label_ids)
        tmp_train_accuracy, tmp_train_f1 = flat_accuracy(logits, label_ids)
        train_accuracy += tmp_train_accuracy
        train_f1 += tmp_train_f1
        nb_train_steps += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(BERT_SA.parameters(), 1.0)
        optimizer.step()
        
    avg_train_loss = total_loss / len(train_dataloader)
    Train_Acc=train_accuracy/nb_train_steps
    Train_F1_Score=train_f1/nb_train_steps


    

    print("Loss:{0:10.4f}\t Average training loss: {1:10.4f}\n Accuracy: {2:10.4f}\t F1 score: {3:10.4f}".format(loss,avg_train_loss,Train_Acc,Train_F1_Score))

    S_writer.add_scalar('Train_avg_loss',avg_train_loss,epoch)
     
    S_writer.add_scalar(' Train_Acc', Train_Acc,epoch)

       
     
    S_writer.add_scalar('Train_F1_Score',Train_F1_Score,epoch)
    

    print("============Running Validation============")
    BERT_SA.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_f1 = 0
    for batch in tqdm(val_dataloader):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = BERT_SA(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy, tmp_eval_f1 = flat_accuracy(logits, label_ids)
           
            eval_accuracy += tmp_eval_accuracy
            eval_f1 += tmp_eval_f1
            nb_eval_steps += 1
    Eval_Acc=eval_accuracy/nb_eval_steps      
    Eval_F1_Score=eval_f1/nb_eval_steps
    
            
    print(" Accuracy: {0:10.4f} \tF1 score: {1:10.4f}".format(Eval_Acc,Eval_F1_Score))
    
    S_writer.add_scalar('Eval_F1_Score',Eval_F1_Score,epoch)
     
    S_writer.add_scalar('Eval_accuracy',Eval_Acc,epoch)

    if Best_val_acc<=Eval_Acc and Best_val_f1<=Eval_F1_Score:
        torch.save(BERT_SA.state_dict(),os.path.join('./main', f"model.bin"))
        Best_val_acc=Eval_Acc
        Best_val_f1=Eval_F1_Score
print("Training complete!")