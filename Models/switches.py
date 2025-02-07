from log import get_logger

import os
import torch
import numpy as np

import argparse
from packaging import version

from Model.training import Training
from Model.model import Propaganda_Detection
from Model.inference import Inferencer
from dataPreparation import Dataset_Preparation

import transformers
from transformers import AutoTokenizer

pytorch_version = version.parse(transformers.__version__)
assert pytorch_version >= version.parse('3.0.0'), \
    'We now only support transformers version >=3.0.0, but your version is {}'.format(pytorch_version)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device == torch.device('cuda')

if __name__ == '__main__':

    # https://github.com/uf-hobi-informatics-lab/ClinicalTransformerNER/blob/master/src/run_transformer_ner.py
    parser = argparse.ArgumentParser()

    # ADD ARGUEMENTS
    parser.add_argument("--domain_type", default='CS', type=str,
                        help="valid values: MEMES, ENGLISH, CS")
    parser.add_argument("--model_run", default='XLM_RoBerta', type=str,
                        help="valid values: XLM_RoBerta")
    parser.add_argument("--model_type", default='xlm-roberta-base', type=str,
                        help="valid values: xlm-roberta-base")
    parser.add_argument("--tokenizer_type", default='xlm-roberta-base', type=str,
                        help="valid values: xlm-roberta-base")
    parser.add_argument("--training", default=0, type=int,
                        help="valid values: 1 for Training, 0 for Validation")
    parser.add_argument("--seed", default=42, type=int,
                        help='random seed')
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="maximum number of tokens allowed in each sentence")
    parser.add_argument("--validation_batch_size", default=16, type=int,
                        help="The batch size for evaluation.")
    parser.add_argument("--training_batch_size", default=16, type=int,
                        help="The batch size for training")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for optimizer.")
    parser.add_argument("--num_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="Weight Decay for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--optimizer", default='AdamW', type=str,
                        help="valid values: AdamW, SGD")
    parser.add_argument("--scheduler", default='LinearWarmup', type=str,
                        help="valid values: LinearWarmup, LRonPlateau")
    parser.add_argument("--full_finetuning", default=1, type=int,
                        help="Update weights for all layers or finetune last classification layer")
    parser.add_argument("--debugging", default=1, type=int,
                        help="Debugging Mode")

    global_args = parser.parse_args()

    print('##################################################')
    LogFileExist = os.path.exists(os.getcwd() + '/Log_Files')
    ModelFileExist = os.path.exists(os.getcwd() + '/Model_Files')
    SwitchesFileExist = os.path.exists(os.getcwd() + '/Switch_Files')
    if not SwitchesFileExist:
        os.makedirs(os.getcwd() + '/Switch_Files')
    if not LogFileExist:
        os.makedirs(os.getcwd() + '/Log_Files')
    if not ModelFileExist:
        os.makedirs(os.getcwd() + '/Model_Files')

    print('1. MODEL AND LOG FILE CREATED')
    paths = {
            "Techniques":"./techniques.json",
            "Log_Folder":"./Log_Files/",
            "Model_Files":"./Model_Files/",
            "Switch_Files": "./Switch_Files/",
            "Training_Data": "./Data_Files/Splits/train_split.json",
            "Validation_Data": "./Data_Files/Splits/val_split.json",
            "Testing_Data": "./Data_Files/Splits/test_split.json",
            "Meme_Training_Data": "./Data_Files/Meme_Data_Splits/training_set_.json",
    }
    print('2. PATHS CREATED')

    hyper_params = {
        "domain_type": global_args.domain_type,
        "model_run": global_args.model_run,
        "model_type": global_args.model_type,
        "tokenizer_type": global_args.tokenizer_type,
        "training": bool(global_args.training),
        "max_seq_length": global_args.max_seq_length,
        "random_seed": global_args.seed,
        "training_batch_size": global_args.training_batch_size,
        "validation_batch_size": global_args.validation_batch_size,
        "learning_rate": global_args.learning_rate,
        "epsilon": global_args.adam_epsilon,
        "weight_decay": global_args.weight_decay,
        "epochs": global_args.num_epochs,
        "scheduler": global_args.scheduler,
        "optimizer": global_args.optimizer,
        "max_grad_norm": global_args.max_grad_norm,
        "full_finetuning": bool(global_args.full_finetuning),
        "debugging": bool(global_args.debugging),
        "log_file": None,
        "datetime": None
    }

    hyper_params['mode'] = 'SWITCHES'  

    print('3. HYPER_PARAMS CREATED')


    
    ################################################## SEEDS
    seed = hyper_params['random_seed']
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    from datetime import datetime
    current_datetime = datetime.now()
    date_time = current_datetime.strftime("%d-%m-%Y_%H:%M:%S")
    hyper_params['datetime'] = date_time
    print('4. SEEDS AND DATATIME SET')


    ################################################## LOG FILE SET UP
    
    if not hyper_params["debugging"]:
        if hyper_params["training"]:
            file_name = paths["Switch_Files"] + hyper_params['model_run'] + '-Train'
        else:
            file_name = paths["Switch_Files"] + hyper_params['model_run'] + '-Inference'
        hyper_params['log_file'] = file_name
        logger_meta = get_logger(name='META', file_name=file_name, type='meta')
        logger_progress = get_logger(name='PORGRESS', file_name=file_name, type='progress')
        logger_results = get_logger(name='RESULTS', file_name=file_name, type='results')
        for i, (k, v) in enumerate(hyper_params.items()):
            if i == (len(hyper_params) - 1):
                logger_meta.warning("{}: {}\n".format(k, v))
            else:
                logger_meta.warning("{}: {}".format(k, v))
    else:
        logger_meta = None
        logger_progress = None
        logger_results = None
    
    logger_object = [logger_meta, logger_progress, logger_results]
    print('5. LOG FILE INITIALIZED')


############################################################# RUN MODELS
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<                 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def run_model(hyper_params, logger_object, paths):

    ################################################## MODELS 

    # Models
    # 1. XLM_RoBerta_MEMES-XLM_RoBerta_ENGLISH
    # 2. XLM_RoBerta_ENGLISH-XLM_RoBerta_Roman_Urdu                             
    # 3. XLM_RoBerta_MEMES-XLM_RoBerta_Roman_Urdu                  
    # 4. XLM_RoBerta_MEMES-XLM_RoBerta_ENGLISH-XLM_RoBerta_Roman_Urdu    

    # Domain Type
    # 1. MEMES
    # 2. ENGLISH
    # 3. CS                 

    df_val = Dataset_Preparation.read_json_files_to_df(paths['Validation_Data'],hyper_params, 'Validation')
    df_test = Dataset_Preparation.read_json_files_to_df(paths['Testing_Data'],hyper_params, 'Testing')

    hyper_params['model_type'] = 'xlm-roberta-base'
    hyper_params['tokenizer_type'] = 'xlm-roberta-base'

    if hyper_params['training']:
        if hyper_params['domain_type'] == 'MEMES':
            df_train = Dataset_Preparation.read_json_files_to_df(paths['Meme_Training_Data'],hyper_params, 'Training') # Here we use Memes
        elif hyper_params['domain_type'] == 'ENGLISH':
            df_train = Dataset_Preparation.read_json_files_to_df(paths['Training_Data'],hyper_params, 'Training')
        elif hyper_params['domain_type'] == 'CS':
            df_train = Dataset_Preparation.read_json_files_to_df(paths['Training_Data'],hyper_params, 'Training')
        else:
            raise Exception('Model and Domain Type don\'t match')


    ################################################## LOG FILE UPDATES
    logger_meta, logger_progress, logger_results = logger_object

    if hyper_params['training']:
        hyper_params['df_train'] = df_train
        hyper_params['df_val'] = df_val
    else:
        hyper_params['df_test'] = df_test


    if hyper_params['training']:
        print()
        print('Training Datatset: {}'.format(len(hyper_params['df_train'])))
        print('Validation Datatset: {}'.format(len(hyper_params['df_val'])))
        print()
    else:
        print('Testing Datatset: {}'.format(len(hyper_params['df_test'])))
        print()


    if not hyper_params["debugging"]:
        if hyper_params['training']:
            logger_meta.warning('Training Datatset: {}'.format(len(hyper_params['df_train'])))
            logger_meta.warning('Validation Datatset: {}'.format(len(hyper_params['df_val'])))
        else:
            logger_meta.warning('Testing Datatset: {}'.format(len(hyper_params['df_test'])))






    print('##################################################')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device == torch.device('cuda')

    techniques = Dataset_Preparation.read_techniques(paths['Techniques'])
    
    checkpoint_model = hyper_params['model_type']
    checkpoint_tokenizer = hyper_params['tokenizer_type']
        
    ##################################################  MODEL + TOKENIZER
    if hyper_params['training']:

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_tokenizer, do_lower_case = False)
        model = Propaganda_Detection(checkpoint_model=checkpoint_model, num_tags=len(techniques), device=device, hyper_params=hyper_params)
        model = model.to(device)
        if hyper_params['step'] != 'First':
                
            path = os.getcwd() + '/Switch_Files/'
            checkpoint = torch.load(path + 'INTERMEDIATE.pt')
            model.load_state_dict(checkpoint['model'])
            tokenizer = AutoTokenizer.from_pretrained(path + 'INTERMEDIATE_tokenizer/')

        print('##################################################')

        if not hyper_params["debugging"]:
            logger_meta.warning("Vocab Size: {}\n".format(tokenizer.vocab_size))

            logger_meta.warning("Domain Tyep: {}".format(hyper_params['domain_type']))
            logger_meta.warning("Model Type: {}".format(hyper_params['model_type']))
            logger_meta.warning("Tokenizer Type: {}\n".format(hyper_params['tokenizer_type']))

            logger_progress.critical('Model + Tokenizer Initialized')

        ##################################################  DATA PROCESSING
        dataPrep = Dataset_Preparation(paths, tokenizer, hyper_params, techniques)
        train_dataloader, valid_dataloader = dataPrep.run()

        if not hyper_params["debugging"]:
            logger_progress.critical('Tokenizing sentences and encoding labels')
            logger_progress.critical('Data Loaders Created')


        ##################################################  TRAINING
        if not hyper_params["debugging"]:
            logger_progress.critical('Training Started')

        train = Training(paths, model, tokenizer, hyper_params, train_dataloader, valid_dataloader, techniques, logger_results)
        train.run()

        if not hyper_params["debugging"]:
            logger_progress.critical('Training Finished')
            logger_progress.critical('Model Saved\n\n\n')
    else:
        ################################################# INFERENCE
        if not hyper_params["debugging"]:
            logger_progress.critical('Starting Inference')

        inference = Inferencer(paths, checkpoint_tokenizer, checkpoint_model, hyper_params, techniques, logger_results)
        inference.run()

        if not hyper_params["debugging"]:
            logger_progress.critical('Inference Complete')

    return
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<                 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

print('7. MODEL RUN FUNCTION')

if hyper_params['training']:
    model_list = hyper_params['model_run'].split('-')
    for i, m in enumerate(model_list):
        domain = m.split('_')[-1]
        if domain == 'MEMES':
            hyper_params['domain_type'] = 'MEMES'
        elif domain == 'ENGLISH':
            hyper_params['domain_type'] = 'ENGLISH'
        else:
            hyper_params['domain_type'] = 'CS'


        if m == model_list[-1]:
            hyper_params['step'] = 'Last'
        elif m == model_list[0]:
            hyper_params['step'] = 'First'
        else:
            hyper_params['step'] = 'Intermediate'


        run_model(hyper_params, logger_object, paths)

else:
    run_model(hyper_params, logger_object, paths)
    # Domain Type
    # 1. MEMES
    # 2. ENGLISH
    # 3. CS


    # 1. XLM_RoBerta_MEMES-XLM_RoBerta_ENGLISH
    # 2. XLM_RoBerta_MEMES-XLM_RoBerta                
    # 3. XLM_RoBerta_ENGLISH-XLM_RoBerta                            
    # 4. XLM_RoBerta_MEMES-XLM_RoBerta_ENGLISH-XLM_RoBerta


script = """
python3 switches.py \
    --domain_type DEFAULT \
    --model_run XLM_RoBerta_MEMES-XLM_RoBerta_ENGLISH-XLM_RoBerta \
    --model_type default \
    --tokenizer_type default \
    --max_seq_length 256 \
    --training_batch_size 16 \
    --validation_batch_size 16 \
    --learning_rate 3e-5 \
    --num_epochs 5 \
    --seed 42 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --optimizer AdamW \
    --scheduler LinearWarmup \
    --full_finetuning 1 \
    --training 0 \
    --debugging 0
"""

