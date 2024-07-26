# -*- coding: utf-8 -*-
r""" 
DataModule
==========
    The DataModule encapsulates all the steps needed to process data:
    - Download / tokenize
    - Save to disk.
    - Apply transforms (tokenize, pad, batch creation, etc…).
    - Load inside Dataset.
    - Wrap inside a DataLoader.
"""
import hashlib
import multiprocessing
import os
from argparse import Namespace
from collections import defaultdict
from typing import Dict, List


import click
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import sys



os.environ["TOKENIZERS_PARALLELISM"] = "false"
from model.tokenizer import Tokenizer

PADDED_INPUTS = ["input_ids"]
MODEL_INPUTS = ["input_ids", "input_lengths", "labels"]


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule.

    :param hparams: Namespace with data specific arguments.
    :param tokenizer: Model Tokenizer.

    """

    def __init__(self, hparams: Namespace, tokenizer: Tokenizer):
        super().__init__()
        self.config = hparams
        self.tokenizer = tokenizer

    def build_input(
        self,
        tokenizer: Tokenizer,
        sentence: List[int],
        label: float = None,
        prepare_labels: bool = True,
    ) -> Dict[str, List[int]]:
        if not prepare_labels:
            return {"input_ids": sentence, "input_lengths": len(sentence)}

        output = {
            "input_ids": sentence,
            "input_lengths": [len(sentence)],
            "labels": label
        }

        return(output)

    def build_input_context(
            self,
            tokenizer: Tokenizer,
            sentences: List[int],
            label: float,
            
    ) -> Dict[str, List[int]]:
       
        input_ids=[tokenizer.bos_index]
        
        for s in sentences:
            input_ids.extend(s)
            input_ids.extend([tokenizer.eos_index])
        
        if self.config.pretrained_model == "roberta-large" or self.config.pretrained_model == "roberta-base":
            if len(input_ids) >= 512:
                input_ids = input_ids[:511]
                input_ids.extend([tokenizer.eos_index])

        output = {
            "input_ids": input_ids,
            "input_lengths": len(input_ids),
            "labels": label
        }

        return output
    
    def build_input_convd(
            self,
            tokenizer: Tokenizer,
            sentences: List[int],
            label: float,
            person: int,
            
    ) -> Dict[str, List[int]]:
       
        if person == 1:
            input_ids = self.tokenizer.encode("<p1>")
        elif person == 2:
            input_ids = self.tokenizer.encode("<p2>")
        input_ids.extend([tokenizer.bos_index])
        
        for s in sentences:
            input_ids.extend(s)
            input_ids.extend([tokenizer.eos_index])
        
        if self.config.pretrained_model == "roberta-large" or self.config.pretrained_model == "roberta-base":
            if len(input_ids) >= 512:
                input_ids = input_ids[:511]
                input_ids.extend([tokenizer.eos_index])

        output = {
            "input_ids": input_ids,
            "input_lengths": len(input_ids),
            "labels": label
        }

        return output

    def _tokenize(self, data: List[Dict[str, str]]):
        for i in tqdm(range(len(data))):
            if self.config.task=="1_convd" or self.config.context==True:
                data[i]["text"] = [self.tokenizer.encode(str(sample)) for sample in data[i]["text"]]
            else:
                data[i]["text"] = self.tokenizer.encode(data[i]["text"])
            
            data[i]["label"]= float(data[i]["label"])
    
        return data

    def _get_dataset(
        self,
    ):
        """Loads an Emotion Dataset.

        :param dataset_path: Path to a folder containing the training csv, the development csv's
             and the corresponding labels.
        :param data_folder: Folder used to store data.

        :return: Returns a dictionary with the training and validation data.
        """
        if self.config.task == "2_convt_emp" or self.config.task == "2_convt_emop" or self.config.task == "2_convt_emoi":
            
            f = "data/trac2_CONVT_train.csv"
            df_train = pd.read_csv(f,  escapechar='\\')

            f = "data/trac2_CONVT_dev.csv"
            df_test = pd.read_csv(f,  escapechar='\\')

            df_train["new_text"] = ""
            df_test["new_text"] = ""

            if self.config.context==True:
                limit = len(df_train) - 1
                for i, sample in df_train.iterrows():
                    if i >= limit: break
                    samples = []
                    samples.append(sample["text"])
                    if i > self.config.context_turns:
                        for turn in range(self.config.context_turns):
                            if sample["conversation_id"] == df_train["conversation_id"][i - (turn + 1)]:
                                samples.append(df_train["text"][i - (turn + 1)])
                    df_train["new_text"][i]=samples

                df_train["text"]=df_train["new_text"]

                limit = len(df_test) - 1
                for i, sample in df_test.iterrows():
                    if i >= limit: break
                    samples = []
                    samples.append(sample["text"])
                    if i > self.config.context_turns:
                        for turn in range(self.config.context_turns):
                            if sample["conversation_id"] == df_test["conversation_id"][i - (turn + 1)]:
                                samples.append(df_test["text"][i - (turn + 1)])
                    df_test["new_text"][i]=samples

                df_test["text"]=df_test["new_text"]

            if self.config.task == "2_convt_emp":
                df_train.rename(columns={'Empathy': 'label'}, inplace=True)
                df_test.rename(columns={'Empathy': 'label'}, inplace=True)
                df_train["label"]=df_train["label"]/5


            if self.config.task == "2_convt_emop":
                df_train.rename(columns={'EmotionalPolarity': 'label'}, inplace=True)
                df_test.rename(columns={'EmotionalPolarity': 'label'}, inplace=True)
                df_train["label"]=df_train["label"]/5
               
            if self.config.task == "2_convt_emoi":
                df_train.rename(columns={'Emotion': 'label'}, inplace=True)
                df_test.rename(columns={'Emotion': 'label'}, inplace=True)
                df_train["label"]=df_train["label"]/5

        if self.config.task == "1_convd": 

            f = "data/trac2_CONVT_train.csv"
            df_train = pd.read_csv(f,  escapechar='\\')
            f2 = "data/trac1_CONVD_train.csv"
            df2_train = pd.read_csv(f2,  escapechar='\\')

            f = "data/trac2_CONVT_dev.csv"
            df_test = pd.read_csv(f,  escapechar='\\')
            f2 = "data/trac1_CONVD_dev.csv"
            df2_test = pd.read_csv(f2,  escapechar='\\')        
    
            df2_train.rename(columns={'this_persons_perceived_empathy_of_other_person': 'label'}, inplace=True)
            df2_test.rename(columns={'this_persons_perceived_empathy_of_other_person': 'label'}, inplace=True)
            df2_train["label"]=df2_train["label"]/7
           
            df2_train["new_text"] = ""
            df2_train["person"] = ""
    
            for i in range(len(df2_train)):
                if not df_train[df_train['conversation_id'] == df2_train["conversation_id"][i]].empty:    
                    df2_train["new_text"][i] = df_train[df_train['conversation_id'] == df2_train["conversation_id"][i]]["text"].tolist()
                
                    if df2_train["person_id_1"][i] == df_train[df_train['conversation_id'] == df2_train["conversation_id"][i]].reset_index()["person_id_1"][0]:
                        df2_train["person"][i] = 2
                    elif df2_train["person_id_1"][i] == df_train[df_train['conversation_id'] == df2_train["conversation_id"][i]].reset_index()["person_id_2"][0]:
                        df2_train["person"][i] = 1
                else:
                    df2_train=df2_train.drop(i)
            df2_train["text"]=df2_train["new_text"]
          
            df2_test["new_text"] = ""
            df2_test["person"] = ""
            for i in range(len(df2_test)):
                if not df_test[df_test['conversation_id'] == df2_test["conversation_id"][i]].empty:    
                    df2_test["new_text"][i] = df_test[df_test['conversation_id'] == df2_test["conversation_id"][i]]["text"].tolist()
        
                    if df2_test["person_id"][i] == df_test[df_test['conversation_id'] == df2_test["conversation_id"][i]].reset_index()["person_id_1"][0]:
                        df2_test["person"][i] = 2
                    elif df2_test["person_id"][i] == df_test[df_test['conversation_id'] == df2_test["conversation_id"][i]].reset_index()["person_id_2"][0]:
                        df2_test["person"][i] = 1
                else:
                    df2_test=df2_test.drop(i)
            df2_test["text"]=df2_test["new_text"]
            


            train=df2_train.sample(frac=0.9,random_state=42)
            valid=df2_train.drop(train.index)
            test=df2_test 

            print(len(train))
            print(len(valid))
            print(len(test))

        if self.config.task != "1_convd":
            train=df_train.sample(frac=0.9,random_state=42)
            valid=df_train.drop(train.index)
            test=df_test 


        dataset = {
            "train": train.to_dict("records"),
            "valid": valid.to_dict("records"),
            "test": test.to_dict("records")
        }

        dataset["train"] = self._tokenize(dataset["train"])
        dataset["valid"] = self._tokenize(dataset["valid"])
        dataset["test"] = self._tokenize(dataset["test"])

        return dataset
    
    @classmethod
    def pad_dataset(
        cls, dataset: dict, padding: int = 0, padded_inputs: List[str] = PADDED_INPUTS
    ):
        """
        Pad the dataset.
        NOTE: This could be optimized by defining a Dataset class and
        padding at the batch level, but this is simpler.

        :param dataset: Dictionary with sequences to pad.
        :param padding: padding index.
        :param padded_inputs:
        """
        
        max_l = max(len(x) for x in dataset["input_ids"])
        for name in padded_inputs:
            dataset[name] = [x + [padding] * (max_l - len(x)) for x in dataset[name]]
        return dataset

    def prepare_data(self):
        """
        Lightning DataModule function that will be used to load/download data,
        build inputs with padding and to store everything as TensorDatasets.
        """
        data = self._get_dataset()
        click.secho("Building inputs and labels.", fg="yellow")
        
      
        datasets = {
            "train": defaultdict(list),
            "valid": defaultdict(list),
            "test": defaultdict(list),
        }

        if self.config.task=="1_convd":
            for dataset_name, dataset in data.items():
                for sample in dataset:
                    instance = self.build_input_convd(
                        self.tokenizer, sample["text"], sample["label"], sample["person"]
                    )
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)


        elif self.config.context==True:
            for dataset_name, dataset in data.items():
                for sample in dataset:
                    instance = self.build_input_context(
                        self.tokenizer, sample["text"], sample["label"] 
                    )
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)

        else:
            for dataset_name, dataset in data.items():
                for sample in dataset:
                    instance = self.build_input(
                        self.tokenizer, sample["text"], sample["label"]
                    )
                    for input_name, input_array in instance.items():
                        datasets[dataset_name][input_name].append(input_array)

        click.secho("Padding inputs and building tensors.", fg="yellow")
        tensor_datasets = {"train": [], "valid": [], "test": []}
        for dataset_name, dataset in datasets.items():

            dataset = self.pad_dataset(dataset, padding=self.tokenizer.pad_index)

            for input_name in MODEL_INPUTS:
         
                if input_name == "labels":
                    tensor = torch.tensor(dataset[input_name], dtype=torch.float32)
                
                else:
                    tensor = torch.tensor(dataset[input_name])
              
                tensor_datasets[dataset_name].append(tensor)
     
        self.train_dataset = TensorDataset(*tensor_datasets["train"])
        self.valid_dataset = TensorDataset(*tensor_datasets["valid"])
        self.test_dataset = TensorDataset(*tensor_datasets["test"])

        click.secho(
           "Train dataset (Batch, Candidates, Seq length): {}".format(
               self.train_dataset.tensors[0].shape
           ),
           fg="yellow",
        )
        click.secho(
           "Valid dataset (Batch, Candidates, Seq length): {}".format(
               self.valid_dataset.tensors[0].shape
           ),
           fg="yellow",
        )
        click.secho(
            "Test dataset (Batch, Candidates, Seq length): {}".format(
                self.test_dataset.tensors[0].shape
            ),
            fg="yellow",
        )

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
