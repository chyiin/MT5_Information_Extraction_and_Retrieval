import os
import json
import wandb
import datetime
import argparse
import numpy as np
import pandas as pd
import random
import torch
import logging
from torch.utils.data import DataLoader
from customdataset import CustomDataset
from tqdm import tqdm
from transformers import AdamW, Adafactor
from transformers import get_linear_schedule_with_warmup
from distutils.util import strtobool
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from eval.evaluate import Evaluate

def same_seeds(seed, gpu):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filename', type=str, default='dataset/product_dataset.csv')
    parser.add_argument('--validation_filename', type=str, default='dataset/validation_dataset.csv')
    parser.add_argument('--test_filename', type=str, default='dataset/testing_dataset.csv')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'])
    parser.add_argument('--task', type=str, default='summarize', choices=['summarize', 'product', 'tag', 'tag_t2c', 'gpt_summarize', 'category'])
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--input_length', type=int, default=1024)
    parser.add_argument('--output_length', type=int, default=500)
    parser.add_argument('--train_size', type=float, default=0.9)
    parser.add_argument('--date', type=str, default=datetime.date.today())
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--wandb', type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)
    args = parser.parse_args()

    return args

def set_logger(args, batch, epoch, lr):
    model_path = os.path.join('model_{}_{}/log_{}'.format(args['task'], args['date'], args['mode']))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    log_file = os.path.join(model_path, 'train_{}_{}_{}.log'.format(lr, batch, epoch)) # , args.batch_size, args.n_layers))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def trainer(epoch, total_epoch, tokenizer, model, device, loader, optimizer, scheduler, MAX_GRAD_NORM):

    total_loss = 00
    model.train()
    for _, data in enumerate(tqdm(loader), 0):

        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)
        
        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
        loss = outputs[0]
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
    
    return total_loss/len(loader)
    
def validate(epoch, tokenizer, model, device, loader, output_length):

    save_preds = []
    save_targets = []
    model.eval()
    eval_loss, scores = 0, 0
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader), 0):

            y = data['target_ids'].to(device, dtype = torch.long)

            y_ids = y[:, :-1].contiguous()
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
            loss = outputs[0]
            eval_loss += loss.item()

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=output_length, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                # top_p=0.95,
                # top_k=50
                )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(' ', '') for g in generated_ids][0]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(' ', '') for t in y][0]
            
            logging.info("")
            logging.info(f"Prediction_{_+1}") 
            logging.info(f"<golden> {target}")
            dup_preds = preds.split('|')
            for p_id, pred in enumerate(list(dict.fromkeys(dup_preds))):
                logging.info(f"<predict_{p_id}> {pred}")
            
            save_preds.append(preds)
            save_targets.append(target)

    return eval_loss/len(loader), save_preds, save_targets

def predict(tokenizer, model, device, loader, output_length):

    save_preds = []
    save_targets = []
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader), 0):
            
            y = data['target_ids'].to(device, dtype = torch.long)

            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=output_length, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True,
                # top_p=0.95,
                # top_k=50
                )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(' ', '') for g in generated_ids][0]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True).replace(' ', '') for t in y][0]

            logging.info("")
            logging.info(f"Prediction_{_+1}") 
            logging.info(f"<golden> {target}")
            dup_preds = preds.split('|')
            for p_id, pred in enumerate(list(dict.fromkeys(dup_preds))):
                logging.info(f"<predict_{p_id}> {pred}")

            save_preds.append(preds)
            save_targets.append(target)

    logging.info("")
    recall_1, f1_1, recall_2, f1_2, recall_L, f1_L = Evaluate().evaluate(save_preds, save_targets)
    logging.info("Rouge-1-recall Score: {:.4f}, Rouge-1-f1 Score: {:.4f}".format(recall_1, f1_1))
    logging.info("Rouge-2-recall Score: {:.4f}, Rouge-2-f1 Score: {:.4f}".format(recall_2, f1_2))
    logging.info("Rouge-L-recall Score: {:.4f}, Rouge-L-f1 Score: {:.4f}".format(recall_L, f1_L,))
    return save_preds, save_targets

def main():

    args = parse_args()
    args = vars(args)

    train(args)

def train(args):

    BATCH_SIZE = args['batch'] 
    TRAIN_EPOCHS = args['epoch']
    SEED = args['seed']   
    GPU = args['gpu']       
    MAX_LEN = args['input_length']
    SUMMARY_LEN = args['output_length'] 
    LEARNING_RATE = args['learning_rate']  
    TASK = args['task']

    WARM_UP = 0 
    MAX_GRAD_NORM = 1.0
    
    same_seeds(SEED, GPU)
    set_logger(args, BATCH_SIZE, TRAIN_EPOCHS, LEARNING_RATE)

    mt5_dir = "mt5-base_QA_DRCD-hug_20211026-170002_QA_DRCD-hug_20211028-134553"
    tokenizer = AutoTokenizer.from_pretrained(mt5_dir)
    # tokenizer.add_tokens(['|'])

    model = AutoModelForSeq2SeqLM.from_pretrained(mt5_dir).to(device)
    # model.resize_token_embeddings(len(tokenizer))

    if args['mode'] == 'train':

        if args['wandb']:

            wandb.init(project=f'{TASK}_{datetime.date.today()}', entity="ex_summ")
            
            wandb.config = {
                "data": args['date'],
                "learning_rate": LEARNING_RATE,
                "epochs": TRAIN_EPOCHS,
                "batch_size": BATCH_SIZE
                }

        train_dataset = pd.read_csv(args['train_filename']).sample(frac=1).reset_index(drop=True) 
        train_dataset = train_dataset[['text', 'ctext']]
        train_dataset.ctext = 'summarize: ' + train_dataset.ctext 

        val_dataset = pd.read_csv(args['validation_filename'])
        val_dataset = val_dataset[['text', 'ctext']]
        val_dataset.ctext = 'summarize: ' + val_dataset.ctext
        
        logging.info("TRAIN Dataset: {}".format(train_dataset.shape))
        logging.info("VALID Dataset: {}".format(val_dataset.shape))
        logging.info("")

        training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
        val_set = CustomDataset(val_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

        train_params = {
            'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0
            }

        val_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0
            }

        training_loader = DataLoader(training_set, **train_params)
        val_loader = DataLoader(val_set, **val_params)

        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer1 = list(model.named_parameters())
            no_decay1 = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters1 = [
                {'params': [p for n, p in param_optimizer1 if not any(nd in n for nd in no_decay1)],
                'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer1 if any(nd in n for nd in no_decay1)],
                'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer1 = list(model.classifier.named_parameters())
            optimizer_grouped_parameters1 = [{"params": [p for n, p in param_optimizer1]}]

        optimizer = AdamW(optimizer_grouped_parameters1, lr=LEARNING_RATE)
        total_steps = len(training_loader) * TRAIN_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps*WARM_UP,
            num_training_steps=total_steps
        )

        max_rouge = -100000 # min_loss = 1000000
        logging.info('Initiating Fine-Tuning ...')
        for epoch in range(TRAIN_EPOCHS):

            training_loss = trainer(epoch, TRAIN_EPOCHS, tokenizer, model, device, training_loader, optimizer, scheduler, MAX_GRAD_NORM)
            validation_loss, predictions, target = validate(epoch, tokenizer, model, device, val_loader, SUMMARY_LEN)
            recall_1, f1_1, recall_2, f1_2, recall_L, f1_L = Evaluate().evaluate(predictions, target)
            logging.info("")
            logging.info("Epoch [{}/{}]".format(epoch+1, TRAIN_EPOCHS))
            logging.info("Training Loss:  {:.4f}, Validation Loss: {:.4f}".format(training_loss, validation_loss))
            logging.info("Validation Rouge-1-recall Score: {:.4f}, Rouge-1-f1 Score: {:.4f}".format(recall_1, f1_1))
            logging.info("Validation Rouge-2-recall Score: {:.4f}, Rouge-2-f1 Score: {:.4f}".format(recall_2, f1_2))
            logging.info("Validation Rouge-L-recall Score: {:.4f}, Rouge-L-f1 Score: {:.4f}".format(recall_L, f1_L,))
            logging.info("")

            if recall_L > max_rouge: # validation_loss < min_loss:
                max_rouge = recall_L
                logging.info("saving state dict")
                torch.save(model.state_dict(), 'model_{}_{}/train_{}_{}_{}.pt'.format(args['task'], args['date'], LEARNING_RATE, BATCH_SIZE, TRAIN_EPOCHS))

            if args['wandb']:

                wandb.log({
                    'Training Loss': training_loss,
                    'Validation Loss': validation_loss,
                    'Validation Rouge1-R': recall_1,
                    'Validation Rouge2-R': recall_2,
                    'Validation RougeL-R': recall_L,
                    'Validation Rouge1-F1': f1_1,
                    'Validation Rouge2-F1': f1_2,
                    'Validation RougeL-F1': f1_L,
                    })
                wandb.watch(model)
        
    elif args['mode'] == 'predict':

        df = pd.read_csv(args['test_filename'])
        df = df[['text', 'ctext']]
        df.ctext = 'summarize: ' + df.ctext

        logging.info("TEST Dataset: {}".format(df.shape))
        logging.info("")

        testing_set = CustomDataset(df, tokenizer, MAX_LEN, SUMMARY_LEN)

        testing_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0
            }

        testing_loader = DataLoader(testing_set, **testing_params)

        model.load_state_dict(torch.load('model_{}_{}/train_{}_{}_{}.pt'.format(args['task'], args['date'], LEARNING_RATE, BATCH_SIZE, TRAIN_EPOCHS))) 
        logging.info('Predict on testing data ...')
        predictions, target = predict(tokenizer, model, device, testing_loader, SUMMARY_LEN)

if __name__ == '__main__':

    main()