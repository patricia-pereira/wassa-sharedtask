# -*- coding: utf-8 -*-
r""" 
EmotionTransformer Model
==================
    Hugging-face Transformer Model implementing the PyTorch Lightning interface that
    can be used to train an Emotion Classifier.
"""
import multiprocessing
import os
from argparse import Namespace
from typing import Any, Dict, List, Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml

from torch.utils.data import DataLoader, TensorDataset
from torchnlp.utils import lengths_to_mask
from transformers import AdamW, AutoModel

from model.data_module import DataModule
from model.tokenizer import Tokenizer
from utils import Config
from scipy.stats import pearsonr, spearmanr

class EmotionTransformer(pl.LightningModule):
    """Hugging-face Transformer Model implementing the PyTorch Lightning interface that
    can be used to train an Emotion Classifier.

    :param hparams: ArgumentParser containing the hyperparameters.
    """

    class ModelConfig(Config):
        """The ModelConfig class is used to define Model settings.

        ------------------ Architecture --------------------- 
        :param pretrained_model: Pretrained Transformer model to be used.
        
        ----------------- Tranfer Learning --------------------- 
        :param nr_frozen_epochs: number of epochs where the `encoder` model is frozen.
        :param encoder_learning_rate: Learning rate to be used to fine-tune parameters from the `encoder`.
        :param learning_rate: Learning Rate used during training.
        :param layerwise_decay: Learning rate decay for to be applied to the encoder layers.

        ----------------------- Data --------------------- 
        :param dataset_path: Path to a json file containing our data.
        :param labels: Label set (options: `ekman`, `goemotions`)
        :param batch_size: Batch Size used during training.
        """

        task: str = "3_essay_emo"
        pretrained_model: str = "roberta-base"
    

        # Optimizer
        nr_frozen_epochs: int = 1
        encoder_learning_rate: float = 1.0e-5
        learning_rate: float = 5.0e-5
        layerwise_decay: float = 0.95

        # Training details
        batch_size: int = 4

        context: bool = True
        context_turns: int = 3

    def __init__(self, hparams: Namespace):
        super().__init__()
        self.config = hparams
        self.save_hyperparameters(self.config)
        print(self.config)
        self.validation_step_outputs = []
        self.training_step_outputs = []
        
        self.transformer = AutoModel.from_pretrained(self.config.pretrained_model)
        self.tokenizer = Tokenizer(self.config.pretrained_model, self.config.context)
       
        self.encoder_features = self.transformer.config.hidden_size

        # Resize embeddings to include the added tokens
        self.transformer.resize_token_embeddings(self.tokenizer.vocab_size)

        self.num_layers = self.transformer.config.num_hidden_layers + 1

        # Regression head
        self.classification_head = nn.Linear(
            self.encoder_features, 1
        )

        self.loss = nn.MSELoss()
        
    def on_epoch_start(self):
        if self.current_epoch < self.config.nr_frozen_epochs:
            # Freeze the encoder parameters for the first epoch
            for param in self.transformer.parameters():
                param.requires_grad = False

        else:
            for param in self.transformer.parameters():
                param.requires_grad = True

    def layerwise_lr(self, lr: float, decay: float) -> list:
        """ Separates layer parameters and sets the corresponding learning rate to each layer.

        :param lr: Initial Learning rate.
        :param decay: Decay value.

        :return: List with grouped model parameters with layer-wise decaying learning rate
        """
        opt_parameters = [
            {
                "params": self.transformer.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        ]
        opt_parameters += [
            {
                "params": self.transformer.encoder.layer[l].parameters(),
                "lr": lr * decay ** (self.num_layers - 1 - l),
            }
            for l in range(self.num_layers - 1)
        ]
        return opt_parameters
    
    # Pytorch Lightning Method
    def configure_optimizers(self):
        layer_parameters = self.layerwise_lr(
            self.config.encoder_learning_rate, self.config.layerwise_decay
        )
        head_parameters = [
            {
                "params": self.classification_head.parameters(),
                "lr": self.config.learning_rate,
            }
        ]

        optimizer = AdamW(
            layer_parameters + head_parameters,
            lr=self.config.learning_rate,
            correct_bias=True,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "mse", 
            }
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # Reduce unnecessary padding.
        input_ids = input_ids[:, : input_lengths.max()]
        mask = lengths_to_mask(input_lengths, device=input_ids.device)

        # Run model.
        output = self.transformer(
            input_ids=input_ids,
            attention_mask=mask,
            output_hidden_states=True,
        )
        
        last_hidden_state = output['last_hidden_state']
       
        # Pooling Layer
        sentemb = last_hidden_state[:, 0, :] 
          
        return torch.squeeze(self.classification_head(sentemb))

    # Pytorch Lightning Method
    def training_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        input_ids, input_lengths, labels= batch
        logits = self.forward(input_ids, input_lengths)
        
        loss_value = self.loss(logits, labels) 
        self.training_step_outputs.append({"train_loss":  loss_value})
        
        return {"loss": loss_value, "log": {"train_loss": loss_value}}

    # Pytorch Lightning Method
    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        input_ids, input_lengths, labels = batch
        logits = self.forward(input_ids, input_lengths)
          
        if logits.size() == torch.Size([]):
            logits=torch.unsqueeze(logits, 0)
       
        loss_value = self.loss(logits, labels)
        predictions=logits

        self.validation_step_outputs.append({"val_loss":  loss_value, "predictions": predictions, "labels": labels})

        return {"val_loss":  loss_value, "predictions": predictions, "labels": labels}

    # Pytorch Lightning Method
    def on_validation_epoch_end(
        self, 
    ) -> Dict[str, torch.Tensor]:

        predictions = torch.cat([o["predictions"] for o in self.validation_step_outputs], dim=0)
        labels = torch.cat([o["labels"] for o in self.validation_step_outputs], dim=0)
       
        self.validation_step_outputs=[]

        # We will log the macro and micro-averaged metrics:
        metrics = {
            "mse": torch.tensor(self.loss(torch.tensor(predictions), torch.tensor(labels)))
        }
        
        self.log("mse", metrics["mse"].to(self.transformer.device), prog_bar=True)

        return {
            "progress_bar": metrics,
            "log": metrics,
        }

     # Pytorch Lightning Method
    def on_train_epoch_end(
        self, 
    ) -> Dict[str, torch.Tensor]:
        
        tlosses= [o["train_loss"] for o in self.training_step_outputs] 
        tloss=sum(tlosses) / float(len(tlosses))
        self.log("train_loss", tloss.to(self.transformer.device), prog_bar=True)
        self.training_step_outputs=[]
        
        return 

    # Pytorch Lightning Method
    def test_step(
        self, batch: Tuple[torch.Tensor], batch_nb: int, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """ Same as validation_step. """
        return self.validation_step(batch, batch_nb)

    # Pytorch Lightning Method
    def on_test_epoch_end(
        self, 
    ) -> Dict[str, float]:
        """ Similar to the validation_step_end but computes precision, recall, f1 for each label."""
       
        predictions = torch.cat([o["predictions"] for o in self.validation_step_outputs], dim=0)
        labels = torch.cat([o["labels"] for o in self.validation_step_outputs], dim=0)

        print("pearson:", pearsonr(predictions.detach().cpu().numpy(),labels.detach().cpu().numpy()))
        print("spearman:", spearmanr(predictions.detach().cpu().numpy(),labels.detach().cpu().numpy()))

        # We will log the macro and micro-averaged metrics:
        metrics = {
            "mse": torch.tensor(self.loss(torch.tensor(predictions), torch.tensor(labels)))
        }
          
        self.log('metrics', metrics)
     
        return {
            "progress_bar": metrics,
            "log": metrics,
        }

    # Pytorch Lightning Method
    def on_epoch_end(self):
        """ Pytorch lightning hook """
        if self.current_epoch + 1 >= self.nr_frozen_epochs:
            self.unfreeze_encoder()

    @classmethod
    def from_experiment(cls, experiment_folder: str):
        """Function that loads the model from an experiment folder.

        :param experiment_folder: Path to the experiment folder.

        :return: Pretrained model.
        """
        hparams_file = experiment_folder + "hparams.yaml"
        hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file for file in os.listdir(experiment_folder +"checkpoints/") if file.endswith(".ckpt")
        ]
    
        checkpoint_path = experiment_folder +"checkpoints/"+ checkpoints[-1]
        model = cls.load_from_checkpoint(
            checkpoint_path, hparams=Namespace(**hparams), strict=True
        )
        # Make sure model is in prediction mode
        model.eval()
        model.freeze()
        
        return model
