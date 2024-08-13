# coding=utf-8

# Reference: https://github.com/huggingface/pytorch-pretrained-BERT

"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function

import copy
import json
import math

import six
import torch
from torchcrf import CRF
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import tensorflow as tf
import datetime
from transformers import BertModel, BertConfig, AdamW, AutoModel, AutoTokenizer,AutoConfig

# BERT + softmax
class BertForTABSAJoint(nn.Module):
	def __init__(self, model_name,num_labels, num_ner_labels, max_seq_length):
		super(BertForTABSAJoint, self).__init__()
		config = AutoConfig.from_pretrained(model_name)
		self.bert = AutoModel.from_pretrained(model_name)
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is the type sum of 0 & 1
		self.ner_hidden2tag = nn.Linear(config.hidden_size, num_ner_labels) # num_ner_labels is the type sum of ner labels: TO or BIO etc
		self.num_labels = num_labels
		self.num_ner_labels = num_ner_labels
		self.max_seq_length = max_seq_length

	def forward(self, input_ids, token_type_ids, attention_mask, labels, ner_labels):
		outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
		all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		# get the last hidden layer
		sequence_output = outputs.last_hidden_state
		# cross a dropout layer
		# sequence_output = self.dropout(sequence_output)
		pooled_output = sequence_output[:, 0, :]
		# pooled_output = self.dropout(pooled_output)
		# the Classifier of category & polarity
		logits = self.classifier(pooled_output)
		ner_logits = self.ner_hidden2tag(sequence_output)
		ner_logits.reshape([-1, self.max_seq_length, self.num_ner_labels])

		loss_fct = CrossEntropyLoss()
		loss = loss_fct(logits, labels)
		ner_loss_fct = CrossEntropyLoss(ignore_index=0)
		ner_loss = ner_loss_fct(ner_logits.view(-1, self.num_ner_labels), ner_labels.view(-1))
		return loss, ner_loss, logits, ner_logits


# BERT + CRF
class BertForTABSAJoint_CRF(nn.Module):

	def __init__(self, model_name, num_labels, num_ner_labels):
		super(BertForTABSAJoint_CRF, self).__init__()
		config = AutoConfig.from_pretrained(model_name)
		self.bert = AutoModel.from_pretrained(model_name)
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels) # num_labels is the type sum of 0 & 1
		self.ner_hidden2tag = nn.Linear(config.hidden_size, num_ner_labels) # num_ner_labels is the type sum of ner labels: TO or BIO etc
		self.num_labels = num_labels
		self.num_ner_labels = num_ner_labels
		# CRF
		self.CRF_model = CRF(num_ner_labels, batch_first=True)

	def forward(self, input_ids, token_type_ids, attention_mask, labels, ner_labels, ner_mask):
		outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
		# all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
		# get the last hidden layer
		sequence_output = outputs.last_hidden_state
		# cross a dropout layer
		# sequence_output = self.dropout(sequence_output)
		pooled_output = sequence_output[:, 0, :]
		# pooled_output = self.dropout(pooled_output)
		# the Classifier of category & polarity
		logits = self.classifier(pooled_output)
		ner_logits = self.ner_hidden2tag(sequence_output)

		# the CRF layer of NER labels
		ner_loss_list = self.CRF_model(ner_logits, ner_labels, ner_mask.type(torch.ByteTensor).cuda(), reduction='none')
		ner_loss = torch.mean(-ner_loss_list)
		ner_predict = self.CRF_model.decode(ner_logits, ner_mask.type(torch.ByteTensor).cuda())

		# the classifier of category & polarity
		loss_fct = CrossEntropyLoss()
		loss = loss_fct(logits, labels)
		return loss, ner_loss, logits, ner_predict
