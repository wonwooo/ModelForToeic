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

def get_logit(model, input_ids, segment_ids):
    input_ids_tensor = torch.tensor(input_ids).unsqueeze(0).to('cuda')
    segment_ids_tensor = torch.tensor(segment_ids).to('cuda')
    outputs = model(input_ids=input_ids_tensor, token_type_ids=segment_ids_tensor)
    logit = outputs[0][0][1]

    return logit.item()

def get_score(model, tokenizer, question_tensors, segment_tensors, masked_index, candidate):
    question_tensors = torch.tensor(question_tensors).unsqueeze(0).to('cuda')
    segment_tensors = torch.tensor(segment_tensors).to('cuda')

    candidate_tokens = tokenizer.tokenize(candidate)  # warranty -> ['warrant', '##y']
    candidate_ids = tokenizer.convert_tokens_to_ids(candidate_tokens)
    with torch.no_grad():
        predictions = model(input_ids=question_tensors, token_type_ids=segment_tensors)
        predictions_candidates = predictions[0][0][masked_index][candidate_ids].mean()

    return predictions_candidates.item()

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open('Part5_test.txt', 'rb') as f:
        testset = pickle.load(f)

    print(f'\n{len(testset)} Toeic part5 questions are loaded! Our model will solve {len(testset)} questions like below.\n')

    for k, v in testset[random.randrange(1, len(testset))].items():
        if k == 'question':
            v = re.sub('_', '[ ? ]', v)
        print(f'{k} : {v}')

    if input("If you want to start the test, Type 'start' and press enter key : ") == 'start':


        print('Preparing models...')
        bert_lm = BertForMaskedLM.from_pretrained('bert-base-uncased')
        bert_lm.cuda()
        bert_lm.eval()
        bert_grammer = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        bert_grammer.load_state_dict(torch.load('checkpoint.pt'))
        bert_grammer.cuda()
        bert_grammer.eval()
        cnt_bert_base = 0
        cnt_bert_grammer = 0


        for i, pset in enumerate(testset):

            grammer_score = []
            if (i + 1) % 100 == 0:
                print("Testing {} in {}".format(i + 1, len(testset)))
            q = pset['question'].lower()
            ans = pset['answer'].lower()
            sentence_lm = re.sub('_', ' [MASK] ', q)
            input_ids_lm = tokenizer.encode(sentence_lm, add_special_tokens=True)
            masked_index_lm = input_ids_lm.index(103)
            segment_ids_lm = [0] * len(input_ids_lm)

            lm_score = [get_score(bert_lm, tokenizer, input_ids_lm, segment_ids_lm, masked_index_lm, pset[str(j)]) for j in
                        range(1, 5)]

            for k in range(1, 5):
                sentence_grammer = re.sub('_', ' ' + pset[str(k)] + ' ', q).lower()
                input_ids_grammer = tokenizer.encode(sentence_grammer, add_special_tokens=True)
                segment_ids_grammer = [0] * len(input_ids_grammer)
                grammer_score.append(get_logit(bert_grammer, input_ids_grammer, segment_ids_grammer))

            softmax = torch.nn.Softmax(dim=0)
            pred_tunedModel = np.argmax(grammer_score) + 1
            pred_baseModel = torch.argmax(softmax(torch.tensor(lm_score))).item() + 1

            if pset[str(pred_tunedModel)].lower() == ans:
                cnt_bert_grammer += 1
            if pset[str(pred_baseModel)].lower() == ans:
                cnt_bert_base += 1
            if 0 <= i < 20:
                if i == 0:
                    print(
                        '==================================================Predictions Example==================================================')
                    print('{0:15s} {1:<30s} {2:<30s} {3:<30s}'.format('', 'Correct answer', 'Pretrained BertForMaskedLM', 'Finetuned BertForSeqClf'))

                print('{0:15s} {1:<30s} {2:<30s} {3:<30s}'.format('Question' + str(i + 1), ans, pset[str(pred_baseModel)].lower(),
                                                                  pset[str(pred_tunedModel)].lower()))
            if i == 20:
                print('.\n.\n.')
        print(
            '=================================================Test finished=================================================\n')
        print('Pretrained BertForMaskedLM : {}\nFinetuned BertForSequenceClassification(Bert_grammer) : {}'.format(
            cnt_bert_base / len(testset), cnt_bert_grammer / len(testset)))

    else:
        print('Testing finished')