import json
import copy
import re
import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, BertForMaskedLM, AdamW, BertConfig, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
import random
from sklearn.model_selection import train_test_split
import pickle
from pytorchtools import EarlyStopping


if __name__ == "__main__":
    print("=================================Loading training datasets====================================")
    with open('Part5_training.txt', 'rb') as f:
        dataset = pickle.load(f)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    training, val = train_test_split(dataset, test_size=0.1, random_state=2)

    toeic_training = []
    max_len = 64
    for pset in training:
        q = pset['question'].strip().lower()
        ans = pset['answer'].strip().lower()
        for j in range(1, 5):
            if pset[str(j)].strip().lower() == ans:
                label = 1
            else:
                label = 0
            sentence = re.sub('_', ' ' + pset[str(j)] + ' ', q).lower()

            input_ids = tokenizer.encode(sentence, add_special_tokens=True)
            attn_mask = [1] * len(input_ids) + [0] * (max_len - len(input_ids))
            input_ids = input_ids + [0] * (max_len - len(input_ids))
            segment_ids = [0] * max_len
            if label == 1:
                for _ in range(3):
                    toeic_training.append([input_ids, segment_ids, attn_mask, label])
            else:
                toeic_training.append([input_ids, segment_ids, attn_mask, label])
    random.shuffle(toeic_training)


    toeic_val = []
    max_len = 64
    for pset in val:
        q = pset['question'].strip().lower()
        ans = pset['answer'].strip().lower()
        for j in range(1, 5):
            if pset[str(j)].strip().lower() == ans:
                label = 1
            else:
                label = 0
            sentence = re.sub('_', ' ' + pset[str(j)] + ' ', q).lower()
            input_ids = tokenizer.encode(sentence, add_special_tokens=True)
            attn_mask = [1] * len(input_ids) + [0] * (max_len - len(input_ids))
            input_ids = input_ids + [0] * (max_len - len(input_ids))
            segment_ids = [0] * max_len
            toeic_val.append([input_ids, segment_ids, attn_mask, label])
    random.shuffle(toeic_val)

    input_ids, segment_ids, attn_mask, label = zip(*toeic_training)
    input_ids = torch.LongTensor(input_ids)
    segment_ids = torch.LongTensor(segment_ids)
    label = torch.LongTensor(label)
    attn_mask = torch.LongTensor(attn_mask)
    input_ids_loader = torch.utils.data.DataLoader(input_ids, batch_size=16)
    segment_ids_loader = torch.utils.data.DataLoader(segment_ids, batch_size=16)
    label_loader = torch.utils.data.DataLoader(label, batch_size=16)
    attn_mask_loader = torch.utils.data.DataLoader(attn_mask, batch_size=16)

    # validation
    input_ids_val, segment_ids_val, attn_mask_val, label_val = zip(*toeic_val)
    input_ids_val = torch.LongTensor(input_ids_val)
    segment_ids_val = torch.LongTensor(segment_ids_val)
    label_val = torch.LongTensor(label_val)
    attn_mask_val = torch.LongTensor(attn_mask_val)
    input_ids_loader_val = torch.utils.data.DataLoader(input_ids_val, batch_size=16)
    segment_ids_loader_val = torch.utils.data.DataLoader(segment_ids_val, batch_size=16)
    label_loader_val = torch.utils.data.DataLoader(label_val, batch_size=16)
    attn_mask_loader_val = torch.utils.data.DataLoader(attn_mask_val, batch_size=16)

    print('batch size : 16, num of batch_train : {}, num of batch_validation : {}'.format(len(input_ids_loader),
                                                                                          len(input_ids_loader_val)))



    epochs = 4
    # number of batches * epochs = total number of training step
    total_steps = len(input_ids_loader) * epochs
    bert_grammer = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    bert_grammer.cuda()
    optimizer = AdamW(bert_grammer.parameters(), lr=2e-5, eps=1e-8)

    # create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)


    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    loss_train = []
    loss_val = []
    early_stopping = EarlyStopping(verbose=True)

    for epoch in range(0, epochs):

        # train
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        train_loss = 0
        val_loss = 0
        bert_grammer.train()
        step = 0
        for input_ids, segment_ids, label, attn_mask in zip(input_ids_loader, segment_ids_loader, label_loader,
                                                            attn_mask_loader):

            if step % 200 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(input_ids_loader)))
            step += 1
            input_ids = input_ids.to('cuda')
            segment_ids = segment_ids.to('cuda')
            label = label.to('cuda')
            attn_mask = attn_mask.to('cuda')
            bert_grammer.zero_grad()
            outputs = bert_grammer(input_ids=input_ids, token_type_ids=segment_ids, labels=label,
                                 attention_mask=attn_mask)
            clf_loss = outputs[0]
            train_loss += clf_loss.item()
            clf_loss.backward()

            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(bert_grammer.parameters(), 1.0)
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        avg_train_loss = train_loss / len(input_ids_loader)
        loss_train.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        # validation
        print("")
        print('Running Validaiton...')
        bert_grammer.eval()

        for input_ids_val, segment_ids_val, label_val, attn_mask_val in zip(input_ids_loader_val,
                                                                            segment_ids_loader_val, label_loader_val,
                                                                            attn_mask_loader_val):
            input_ids_val = input_ids_val.to('cuda')
            segment_ids_val = segment_ids_val.to('cuda')
            label_val = label_val.to('cuda')
            attn_mask_val = attn_mask_val.to('cuda')

            with torch.no_grad():
                outputs = bert_grammer(input_ids=input_ids_val, token_type_ids=segment_ids_val, labels=label_val,
                                     attention_mask=attn_mask_val)
            clf_loss_val = outputs[0]
            val_loss += clf_loss_val.item()
        avg_val_loss = val_loss / len(input_ids_loader_val)
        loss_val.append(avg_val_loss)

        early_stopping(avg_val_loss, bert_grammer)
        print("Average validation loss: {0:.2f}".format(avg_val_loss))

        if early_stopping.early_stop:
            print("Early stopping executed")
            break

        bert_grammer.load_state_dict(torch.load('checkpoint.pt'))

    print("Finetuning Bert for grammer is finished!")