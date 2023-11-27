from __future__ import absolute_import, division, print_function

import math

import numpy as np
import torch
from torch import nn
from modules import BertModel, BertTokenizer, MLP, Attention, Gate
from transformers_xlnet import XLNetModel, XLNetTokenizer
from helper import save_json, load_json
import subprocess
import os


DEFAULT_HPARA = {
    'max_seq_length': 128,
    'use_bert': False,
    'use_xlnet': False,
    'use_zen': False,
    'do_lower_case': False,
    'mlp_dropout': 0.33,
    'n_mlp': 400,
}


class Tagger(nn.Module):

    def __init__(self, labelmap, tag_label_map, hpara, model_path, from_pretrained=True):
        super().__init__()
        self.labelmap = labelmap
        self.tag_label_map = tag_label_map
        self.hpara = hpara
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']

        self.tokenizer = None
        self.bert = None
        self.xlnet = None
        self.zen = None
        self.zen_ngram_dict = None

        if self.hpara['use_bert']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            if from_pretrained:
                self.bert = BertModel.from_pretrained(model_path, cache_dir='')
            else:
                from modules import CONFIG_NAME, BertConfig
                config_file = os.path.join(model_path, CONFIG_NAME)
                config = BertConfig.from_json_file(config_file)
                self.bert = BertModel(config)
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_xlnet']:
            self.tokenizer = XLNetTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            if from_pretrained:
                self.xlnet = XLNetModel.from_pretrained(model_path)
                state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'))
                key_list = list(state_dict.keys())
                reload = False
                for key in key_list:
                    if key.find('xlnet.') > -1:
                        reload = True
                        state_dict[key[key.find('xlnet.') + len('xlnet.'):]] = state_dict[key]
                    state_dict.pop(key)
                if reload:
                    self.xlnet.load_state_dict(state_dict)
            else:
                config, model_kwargs = XLNetModel.config_class.from_pretrained(model_path, return_unused_kwargs=True)
                self.xlnet = XLNetModel(config)
            hidden_size = self.xlnet.config.hidden_size
            self.dropout = nn.Dropout(self.xlnet.config.summary_last_dropout)
        else:
            raise ValueError()

        # self.tokenizer.add_never_split_tokens(["[V]", "[/V]"])

        self.mlp_e = MLP(n_in=hidden_size,
                         n_hidden=self.hpara['n_mlp'],
                         dropout=self.hpara['mlp_dropout'])
        self.mlp_s = MLP(n_in=hidden_size,
                         n_hidden=self.hpara['n_mlp'],
                         dropout=self.hpara['mlp_dropout'])

        self.attention = Attention(self.hpara['n_mlp'])

        self.gate = Gate(self.hpara['n_mlp'])

        self.tag_decoder = nn.Linear(self.hpara['n_mlp'], len(self.tag_label_map) + 2)
        self.label_decoder = nn.Linear(self.hpara['n_mlp'] * 2, len(self.labelmap) + 2)

        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None,
                attention_mask_label=None,
                labels=None, tag_labels=None,
                input_ngram_ids=None, ngram_position_matrix=None,
                ):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.xlnet is not None:
            transformer_outputs = self.xlnet(input_ids, token_type_ids, attention_mask=attention_mask)
            sequence_output = transformer_outputs[0]
        else:
            raise ValueError()

        batch_size, _, feat_dim = sequence_output.shape
        max_len = attention_mask_label.shape[1]
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=sequence_output.dtype, device=input_ids.device)
        for i in range(batch_size):
            temp = sequence_output[i][valid_ids[i] == 1]
            sent_len = attention_mask_label[i].sum()
            # valid_output[i][:temp.size(0)] = temp
            valid_output[i][:sent_len] = temp[:sent_len]

        valid_output = self.dropout(valid_output)

        h_e = self.mlp_e(valid_output)
        h_s = self.mlp_s(valid_output)

        a = self.attention(h_s)
        o = self.gate(h_e, a)

        label_outs = self.label_decoder(o)

        if tag_labels is not None and labels is not None:
            tag_outs = self.tag_decoder(h_s)
            tag_outs = tag_outs[attention_mask_label]
            tag_gold = tag_labels[attention_mask_label]
            tag_loss = self.loss_function(tag_outs, tag_gold)

            label_outs = label_outs[attention_mask_label]
            label_gold = labels[attention_mask_label]
            label_loss = self.loss_function(label_outs, label_gold)
            return tag_loss + label_loss
        else:
            pre_labels = torch.argmax(label_outs, dim=2)
            return pre_labels


    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_length
        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_xlnet'] = args.use_xlnet
        hyper_parameters['do_lower_case'] = args.do_lower_case
        hyper_parameters['mlp_dropout'] = args.mlp_dropout
        hyper_parameters['n_mlp'] = args.n_mlp

        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    def save_model(self, output_model_dir, vocab_dir):
        best_eval_model_dir = os.path.join(output_model_dir, 'model')
        if not os.path.exists(best_eval_model_dir):
            os.makedirs(best_eval_model_dir)

        output_model_path = os.path.join(best_eval_model_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        output_tag_file = os.path.join(best_eval_model_dir, 'labelset.json')
        save_json(output_tag_file, self.labelmap)

        output_tag_file = os.path.join(best_eval_model_dir, 'taglabelset.json')
        save_json(output_tag_file, self.tag_label_map)

        output_hpara_file = os.path.join(best_eval_model_dir, 'hpara.json')
        save_json(output_hpara_file, self.hpara)

        output_config_file = os.path.join(best_eval_model_dir, 'config.json')
        with open(output_config_file, "w", encoding='utf-8') as writer:
            if self.bert:
                writer.write(self.bert.config.to_json_string())
            elif self.xlnet:
                writer.write(self.xlnet.config.to_json_string())
            elif self.zen:
                writer.write(self.zen.config.to_json_string())
        output_bert_config_file = os.path.join(best_eval_model_dir, 'bert_config.json')
        command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
        subprocess.run(command, shell=True)

        if self.bert:
            vocab_name = 'vocab.txt'
        elif self.xlnet:
            vocab_name = 'spiece.model'
        elif self.zen:
            vocab_name = 'vocab.txt'
        else:
            raise ValueError()
        vocab_path = os.path.join(vocab_dir, vocab_name)
        command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(best_eval_model_dir, vocab_name))
        subprocess.run(command, shell=True)

    @classmethod
    def load_model(cls, model_path, device):
        tag_file = os.path.join(model_path, 'labelset.json')
        labelmap = load_json(tag_file)

        tag_file = os.path.join(model_path, 'taglabelset.json')
        taglabelmap = load_json(tag_file)

        hpara_file = os.path.join(model_path, 'hpara.json')
        hpara = load_json(hpara_file)
        DEFAULT_HPARA.update(hpara)

        res = cls(labelmap=labelmap, tag_label_map=taglabelmap, hpara=DEFAULT_HPARA, model_path=model_path)
        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
        return res

    def load_data(self, data_path, do_predict=False):
        if not do_predict:
            flag = data_path[data_path.rfind('/')+1: data_path.rfind('.')]
        else:
            flag = 'predict'

        lines = readfile(data_path, flag)

        examples = self.process_data(lines, flag)

        return examples

    @staticmethod
    def process_data(lines, flag):

        examples = []
        for i, (sentence, label, tag_label) in enumerate(lines):
            guid = "%s-%s" % (flag, i)
            examples.append(InputExample(guid=guid, text_a=sentence, text_b=None,
                                         label=label, tag_label=tag_label))
        return examples

    def convert_examples_to_features(self, examples):

        features = []

        length_list = []
        tokens_list = []
        labels_list = []
        tag_labels_list = []
        valid_list = []
        label_mask_list = []
        eval_mask_list = []

        for (ex_index, example) in enumerate(examples):
            text_list = example.text_a
            label_list = example.label
            tag_label_list = example.tag_label
            tokens = []
            labels = []
            tag_labels = []
            valid = []
            label_mask = []
            eval_mask = []

            if len(text_list) > self.max_seq_length - 2:
                continue

            for i, word in enumerate(text_list):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = label_list[i]
                tag_label_1 = tag_label_list[i]
                for m in range(len(token)):
                    if m == 0:
                        valid.append(1)
                        labels.append(label_1)
                        tag_labels.append(tag_label_1)
                        eval_mask.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)

            assert len(tokens) == len(valid)
            assert len(eval_mask) == len(label_mask)

            length_list.append(len(tokens))
            tokens_list.append(tokens)
            labels_list.append(labels)
            tag_labels_list.append(tag_labels)
            valid_list.append(valid)
            label_mask_list.append(label_mask)
            eval_mask_list.append(eval_mask)

        label_len_list = [len(label) for label in labels_list]
        seq_pad_length = max(length_list) + 2
        label_pad_length = max(label_len_list)

        for indx, (example, tokens, tag_labels, labels, valid, label_mask, eval_mask) in \
                enumerate(zip(examples, tokens_list, tag_labels_list, labels_list,
                              valid_list, label_mask_list, eval_mask_list)):

            ntokens = []
            segment_ids = []
            label_ids = []
            tag_label_ids = []

            ntokens.append("[CLS]")
            segment_ids.append(0)
            valid.insert(0, 0)

            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
            for i in range(len(labels)):
                if labels[i] in self.labelmap:
                    label_ids.append(self.labelmap[labels[i]])
                else:
                    label_ids.append(self.labelmap['<UNK>'])
                if tag_labels[i] in self.tag_label_map:
                    tag_label_ids.append(self.tag_label_map[tag_labels[i]])
                else:
                    tag_label_ids.append(self.tag_label_map['<UNK>'])
            ntokens.append("[SEP]")
            segment_ids.append(0)
            valid.append(0)

            assert sum(valid) == len(label_ids)

            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < seq_pad_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                valid.append(1)
            while len(label_ids) < label_pad_length:
                label_ids.append(0)
                tag_label_ids.append(0)
                label_mask.append(0)
                eval_mask.append(0)

            assert len(input_ids) == seq_pad_length
            assert len(input_mask) == seq_pad_length
            assert len(segment_ids) == seq_pad_length
            assert len(valid) == seq_pad_length

            assert len(label_ids) == label_pad_length
            assert len(label_mask) == label_pad_length
            assert len(eval_mask) == label_pad_length

            ngram_ids = None
            ngram_positions_matrix = None
            ngram_lengths = None
            ngram_tuples = None
            ngram_seg_ids = None
            ngram_mask_array = None

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              tag_label_id=tag_label_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              eval_mask=eval_mask,
                              ngram_ids=ngram_ids,
                              ngram_positions=ngram_positions_matrix,
                              ngram_lengths=ngram_lengths,
                              ngram_tuples=ngram_tuples,
                              ngram_seg_ids=ngram_seg_ids,
                              ngram_masks=ngram_mask_array,
                              ))
        return features

    def feature2input(self, device, feature):
        all_input_ids = torch.tensor([f.input_ids for f in feature], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in feature], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in feature], dtype=torch.long)
        all_tag_label_ids = torch.tensor([f.tag_label_id for f in feature], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in feature], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in feature], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in feature], dtype=torch.bool)
        all_eval_mask_ids = torch.tensor([f.eval_mask for f in feature], dtype=torch.bool)
        input_ids = all_input_ids.to(device)
        input_mask = all_input_mask.to(device)
        segment_ids = all_segment_ids.to(device)
        label_ids = all_label_ids.to(device)
        valid_ids = all_valid_ids.to(device)
        l_mask = all_lmask_ids.to(device)
        eval_mask = all_eval_mask_ids.to(device)
        tag_label_ids = all_tag_label_ids.to(device)

        if self.zen is not None:
            all_ngram_ids = torch.tensor([f.ngram_ids for f in feature], dtype=torch.long)
            all_ngram_positions = torch.tensor([f.ngram_positions for f in feature], dtype=torch.long)
            # all_ngram_lengths = torch.tensor([f.ngram_lengths for f in train_features], dtype=torch.long)
            # all_ngram_seg_ids = torch.tensor([f.ngram_seg_ids for f in train_features], dtype=torch.long)
            # all_ngram_masks = torch.tensor([f.ngram_masks for f in train_features], dtype=torch.long)

            ngram_ids = all_ngram_ids.to(device)
            ngram_positions = all_ngram_positions.to(device)
        else:
            ngram_ids = None
            ngram_positions = None

        return input_ids, input_mask, l_mask, eval_mask, tag_label_ids, label_ids, \
               ngram_ids, ngram_positions, segment_ids, valid_ids


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, tag_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.tag_label = tag_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, tag_label_id, label_id, valid_ids=None,
                 label_mask=None, eval_mask=None,
                 ngram_ids=None, ngram_positions=None, ngram_lengths=None,
                 ngram_tuples=None, ngram_seg_ids=None, ngram_masks=None,
                 ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.tag_label_id = tag_label_id
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.eval_mask = eval_mask

        self.ngram_ids = ngram_ids
        self.ngram_positions = ngram_positions
        self.ngram_lengths = ngram_lengths
        self.ngram_tuples = ngram_tuples
        self.ngram_seg_ids = ngram_seg_ids
        self.ngram_masks = ngram_masks


def readfile(filename, flag):
    data = []
    sentence = []
    label = []
    tag_label = []

    with open(filename, 'r', encoding='utf8') as f:
        lines = f.readlines()
        if not flag == 'predict':
            for line in lines:
                line = line.strip()
                if line == '':
                    if len(sentence) > 0:
                        data.append((sentence, label, tag_label))
                        sentence = []
                        label = []
                        tag_label = []
                    continue
                splits = line.split()
                sentence.append(splits[0])
                lb = splits[1]
                tag = splits[2]
                tag_label.append(tag)
                label.append(lb)
            if len(sentence) > 0:
                data.append((sentence, label, tag_label))
        else:
            raise ValueError()
    return data
