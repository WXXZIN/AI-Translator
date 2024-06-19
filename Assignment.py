# Huggingface transformers를 이용한 한-영, 영-한 번역기

import os
import sys
import logging
from glob import glob

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import Trainer, TrainingArguments
import sacrebleu

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# ==================================================================

# 데이터 전처리
def preprocess_data():
    df1 = pd.read_excel('./origin_data/1_구어체(1)_200226.xlsx')[['원문', '번역문']]
    df1.columns = ['ko', 'en']
    df1 = df1[:9800]

    df2 = pd.read_csv('./origin_data/custom_data.csv')

    df = pd.concat([df1, df2])

    if not os.path.exists('./data'):
        os.makedirs('./data')

    df.to_csv('./data/data.csv', index=False)

    data = pd.read_csv('./data/data.csv')

    # 데이터셋 분리
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, valid_data = train_test_split(train_data, test_size=0.25, random_state=42)

    # 데이터 저장
    train_data.to_csv('./data/train_data.csv', index=False)
    valid_data.to_csv('./data/valid_data.csv', index=False)
    test_data.to_csv('./data/test_data.csv', index=False)

# 데이터 준비
def prepare_data():
    data_files = {'train': './data/train_data.csv', 'valid': './data/valid_data.csv'}
    dataset = load_dataset('csv', data_files=data_files)
    return dataset

# 학습
def train_translate(totalEpochs=3, src_lang='ko', tgt_lang='en'):
    dataset = prepare_data()

    model_name = 'facebook/m2m100_418M'
    tokenizer = M2M100Tokenizer.from_pretrained(model_name)
    model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)

    # 언어 설정 및 토크나이징
    def tokenize_function(examples):
        srcs = tokenizer(examples[src_lang], truncation=True, padding='max_length', max_length=128)
        targets = tokenizer(examples[tgt_lang], truncation=True, padding='max_length', max_length=128)
        
        return {'input_ids': srcs['input_ids'], 'attention_mask': srcs['attention_mask'], 'labels': targets['input_ids']}
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 학습 파라미터 설정
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=totalEpochs,    # number of training epochs
        per_device_train_batch_size=8,   # batch size for training
        per_device_eval_batch_size=16,   # accumulate gradients over 4 steps
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch"      # evaluation is done at the end of each epoch
    )

    # trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['valid']
    )

    # 모델 학습 시키기
    trainer.train()

    # 학습결과 저장
    model_path = f"./{src_lang}_{tgt_lang}_translation_model/"  # 결과를 저장할 폴더명
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

# 번역
def translate(text: str, src_lang: str, tgt_lang: str, model_path: str) -> str:
    tokenizer = M2M100Tokenizer.from_pretrained(model_path)
    model = M2M100ForConditionalGeneration.from_pretrained(model_path).to(device)

    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))

    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
# ===============================================================================
# main, train or translate
# ===============================================================================
if __name__ == "__main__":
    print('\n', '='*80)
    print('Usage [Training]: python Assignment.py   train   totalEpochNo[예:10]')
    print('Usage [Translate]: python Assignment.py   translate   ')
    print('Usage [Evaluation]: python Assignment.py   eval')

    if len(glob('./data/*.csv')) != 4:
        print("Data preprocessing...")
        preprocess_data()

    if len(sys.argv) > 1:
        # ---------------------------------- 학습
        if sys.argv[1] == 'train':
            totalEpochs = int(sys.argv[2])
            train_translate(totalEpochs, 'en', 'ko')
            train_translate(totalEpochs, 'ko', 'en')

        # ---------------------------------- 번역
        elif sys.argv[1] == 'translate':
            while True:
                query_sentence = input("\n*** Type a sentence or 'quit' to stop: ")
                if query_sentence == 'quit':
                    break
                print("===> query sentence:", query_sentence)

                # 입력 문장이 영어
                if query_sentence.isascii():
                    print(translate(query_sentence, 'en', 'ko', './en_ko_translation_model/'))

                # 익렵 문장이 한국어
                else :
                    print(translate(query_sentence, 'ko', 'en' , './ko_en_translation_model/'))

        # ---------------------------------- 평가
        elif sys.argv[1] == 'eval':

            logging.getLogger("transformers").setLevel(logging.ERROR)

            dataset = load_dataset('csv', data_files={'test': './data/test_data.csv'})
            test_data = dataset['test'].select(range(10))

            for direction in [('en', 'ko'), ('ko', 'en')]:
                src_lang, tgt_lang = direction
                model_path = f'./{src_lang}_{tgt_lang}_translation_model/'

                references = []  # 정답 문장
                translations = []  # 모델 번역 결과

                # 번역 및 결과 저장
                for example in test_data:
                    reference = example[tgt_lang]
                    translation = translate(example[src_lang], src_lang, tgt_lang, model_path)
                    
                    references.append(reference)
                    translations.append(translation)

                # BLEU 스코어 계산
                bleu_score = round(sacrebleu.corpus_bleu(translations, [references]).score, 3)
                print(f"BLEU Score ({src_lang} to {tgt_lang}):", bleu_score)

    else:
        print("Usage: python Assignment.py train|translate|eval")
        sys.exit(1)