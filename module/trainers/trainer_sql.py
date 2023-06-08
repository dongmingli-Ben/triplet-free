from datetime import datetime
import json
import random
from typing import List, Tuple
from ..trainer import Trainer as BaseTrainer
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import re
import rouge
from collections import Counter
from copy import deepcopy
from ..losses.losses import soft_q_loss_with_sparse_rewards
from ..losses.rewards import RewardHelper
from math import e
from torch.utils.tensorboard import SummaryWriter

ROUGE_EVALUATOR = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=2,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=True,
                           apply_best=False,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s

class F1Metric:
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.
        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values
        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def compute(guess: str, answers: List[str]):
        if guess is None or answers is None:
            return 0
        g_tokens = normalize_answer(guess).split()
        scores = [
            F1Metric._prec_recall_f1_score(g_tokens, normalize_answer(a).split())
            for a in answers
        ]
        return max(f1 for p, r, f1 in scores)

def get_seq_len(tensor, eos_token_id):
    seq_len = []
    for seq in tensor:
        for i, token_id in enumerate(seq):
            if token_id == eos_token_id:
                break
        seq_len.append(i+1)
    return seq_len

def merge_batch_tokens(id_list, pad_idx=None):
    num = len(id_list)
    max_len = max(map(len, id_list))
    out = torch.zeros((num, max_len)
        ).fill_(pad_idx).long()
    mask = torch.zeros_like(out)
    for i, ids in enumerate(id_list):
        out[i, :len(ids)] = torch.LongTensor(ids)
        mask[i, :len(ids)] = 1
    return out, mask

def tokenize_comment_batch(comment_texts, tokenizer):
    comment_token_ids = []
    for comment_text in comment_texts:
        token_ids = [tokenizer.eos_token_id] + tokenizer(comment_text, 
                                          add_special_tokens=False)['input_ids'] \
                            + [tokenizer.eos_token_id]
        comment_token_ids.append(token_ids)
    comment_ids, comment_mask = merge_batch_tokens(comment_token_ids, tokenizer.eos_token_id)
    return comment_ids, comment_mask

def cast_to_device(batch: dict, device: torch.device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    return batch

def nested_log(info: dict, log: dict, prefix: str = '', tolist=False):
    prefix = '' if prefix == '' else f'{prefix}/'
    for key, val in info.items():
        if isinstance(val, dict):
            nested_log(val, log, f'{prefix}{key}')
        elif isinstance(val, torch.Tensor):
            val_ = val.detach().cpu()
            if tolist:
                if len(val_.shape) > 1:
                    raise RuntimeError(f'Cannot turn tensor of shape {val_.shape} to 1-dim list')
                log[f'{prefix}{key}'] = val_.tolist()
            else:
                if torch.numel(val_) > 1:
                    log[f'{prefix}{key}/max'] = val_.max().item()
                    log[f'{prefix}{key}/min'] = val_.min().item()
                    log[f'{prefix}{key}/mean'] = val_.mean().item()
                else:
                    log[f'{prefix}{key}'] = val_.item()
        else:
            log[f'{prefix}{key}'] = val

def _shaping_func(reward: torch.FloatTensor, old_min, old_max,
        new_min, new_max) -> torch.FloatTensor:
    percentile = (reward - old_min) / (old_max - old_min)
    return percentile * (new_max - new_min) + new_min

class Trainer(BaseTrainer):

    def __init__(self, model, t_model, opt, 
            tokenizer, train_dl, valid_dl, test_dl, args):
        checkpoint = args.checkpoint
        args.checkpoint = ''
        super().__init__(model, opt, tokenizer, 
            train_dl, valid_dl, test_dl, args)
        # target model
        args.checkpoint = checkpoint
        if isinstance(t_model, nn.Module):
            self.t_model = t_model
        elif t_model == 'copy':
            print('Initializing target model by copy')
            self.t_model = deepcopy(model)
        else:
            raise RuntimeError(f'target model initialization method {t_model} not recognized')
        if checkpoint:
            self.load_model_from_checkpoint(checkpoint, load_history=args.load_history,
                                            from_distributed=self.args.from_distributed,
                                            load_target_model=self.args.load_target_model)
        self.t_model.eval()
        # loss fun
        self.train_mode = args.train_mode
        self.warmup_mode = args.warmup_mode
        self.warmup_steps = args.warmup_steps
        self.tensorboard_writer = SummaryWriter(os.path.join(self.save_dir, 'logs')) if args.tensorboard else None
        # reward
        self.reward_helper = RewardHelper(args)
        self.comment_generator_reward = self.reward_helper.comment_generator_reward_eng
        self.knowledge_reward = self.reward_helper.knowledge_reward
        # for annotation
        self.model: nn.Module
        self.t_model: nn.Module
        self.opt_v: torch.optim.Optimizer
        self.opt_t: torch.optim.Optimizer

    def update_target_model(self, mode, polyak=1e-3):
        if mode == 'polyak':
            for param_, param in zip(
                    self.t_model.parameters(),
                    self.model.parameters()):
                param_.data.copy_(
                    (1 - polyak) * param_ + polyak * param)
        elif mode == 'copy':
            self.t_model.load_state_dict(self.model.state_dict())
        else:
            raise RuntimeError(f'Sync mode {mode} is not recognized')

    def batch_token_ids(self, token_id_list: List[List[int]]) -> torch.Tensor:
        bs = len(token_id_list)
        max_len = max(map(len, token_id_list))
        tensor = torch.zeros((bs, max_len))
        for i in range(bs):
            tensor[i, :len(token_id_list[i])] = torch.tensor(token_id_list[i])
        return tensor.long()

    def calculate_reward(self, post_ids, post_mask,
            docs, comment_ids, comment_texts, comment_mask, tolist=False):
        # import pdb; pdb.set_trace()
        generator_reward = self.comment_generator_reward(
            post_ids, 
            post_mask,
            comment_ids, 
            comment_mask,
            mode=self.args.nll_mode)
        doc_reward, doc_reward_info = self.knowledge_reward(docs, comment_texts, 
                                                       hyp_is_spaced=True, 
                                                       doc_id_spaced=True,
                                                       n=self.args.bleu_reward)
        length_reward = torch.tensor([min(len(s.split()), self.args.length_margin) for s in comment_texts]).to(doc_reward)
        reward = generator_reward*self.args.nll_coefficient + \
            torch.minimum(doc_reward, torch.tensor([self.args.bleu_margin]*doc_reward.size(0)).to(doc_reward)
                         )*self.args.doc_coefficient + \
            length_reward*self.args.length_coefficient
        shaped_reward = _shaping_func(reward, self.args.reward_min, self.args.reward_max, 
            self.args.reward_shaping_min, self.args.reward_shaping_max)

        info = {
            'reward': reward,
            'generator_reward': generator_reward,
            'length_reward': length_reward,
            'shaped_reward': shaped_reward,
        }
        log = {}
        nested_log(info, log, '', tolist=tolist)  # tolist: whether to turn tensor to list
        nested_log(doc_reward_info, log, '', tolist=tolist)
        return reward, shaped_reward, log

    def decode_batch(self, input_ids, input_mask, model=None, decode=None) -> List[List[int]]:
        if model is None:
            model = self.model
        if decode is None:
            decode = self.args.decode
        batch_size = input_ids.size(0)
        input_len = input_ids.size(1)
        eos_token_id = self.tokenizer.eos_token_id
        # import pdb; pdb.set_trace()
        eos_ids = torch.tensor([eos_token_id]*batch_size
            ).reshape(input_ids.size(0), 1).to(self.device)
        done = torch.tensor([False]*batch_size).to(self.device)
        for i in range(self.args.max_len):
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                logits = model(input_ids, attention_mask=input_mask).logits
                prob = F.softmax(logits[:, -1, :], dim=-1)

            if self.args.decode == 'top-k':
                values, token_ids = torch.topk(prob, k=self.args.top_k)
                indices = torch.randint(0, self.args.top_k, (batch_size,)
                    ).reshape(-1, 1).to(self.device)
                next_token_ids = torch.gather(token_ids, 1, indices)
            elif self.args.decode == 'greedy':
                values, token_ids = torch.topk(prob, k=1)
                next_token_ids = token_ids
            elif self.args.decode == 'random':
                # not tested
                next_token_ids = torch.multinomial(prob, 1)
            elif self.args.decode == 'top-p':
                sorted_prob, sorted_indices = torch.sort(prob, descending=True)
                cum_prob = torch.cumsum(sorted_prob, dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cum_prob > self.args.top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = torch.zeros_like(sorted_indices_to_remove).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

                # indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits = logits[:, -1, :]
                logits[indices_to_remove] = -float('Inf')

                probabilities = F.softmax(logits, dim=-1)
                next_token_ids = torch.multinomial(probabilities, 1)
            else:
                raise RuntimeError(f'decoding mode {self.args.decode} is not recognized')
                
            input_ids = torch.cat([input_ids, next_token_ids], dim=1)
            end = (next_token_ids == eos_token_id).reshape(-1)
            done = done | end
            input_mask = torch.cat([
                input_mask,
                ~done.unsqueeze(-1)], dim=1)
            if not torch.any(~done):
                break
        output = []
        for response in input_ids[:, input_len:].tolist():
            try:
                index = response.index(eos_token_id)
            except ValueError:
                index = len(response) - 1
            output.append(response[:index+1])
        return output

    def train_step_sql(self, batch: dict, mode: str) -> Tuple[torch.Tensor, dict]:
        batch = cast_to_device(batch, self.device)
        if mode == 'SQL_OFF':
            # import pdb; pdb.set_trace()
            comment_ids = batch['comment']
            comment_texts = batch['comment_text']
            comment_mask = batch['comment_mask']
            seq_len = comment_mask.sum(dim=1)
            post_doc_ids = batch['pseudo_post_doc_ids']
            post_doc_mask = batch['pseudo_post_doc_mask']
        else:
            # import pdb; pdb.set_trace()
            post_doc_ids = batch['post_doc_ids']
            post_doc_mask = batch['post_doc_mask']
            eos_ids = torch.zeros((post_doc_ids.size(0), 1)).fill_(self.tokenizer.eos_token_id).to(self.device).long()
            input_ids = torch.cat([post_doc_ids, eos_ids], dim=1)
            input_mask = torch.cat([post_doc_mask, torch.ones_like(eos_ids)], dim=1)
            with torch.no_grad():
                comment_ids = self.model.generate(
                    input_ids,
                    attention_mask=input_mask,
                    do_sample=True,
                    top_p=0.4,
                    max_length=input_ids.size(1)+self.args.max_len,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            comment_ids = comment_ids[:, input_ids.size(1)-1:]
            # remove [CLS], [SEP], ([PAD])
            comment_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in comment_ids]
            seq_len = torch.tensor(get_seq_len(comment_ids[:, 1:], self.tokenizer.eos_token_id)).to(self.device) + 1
            comment_mask = torch.arange(comment_ids.size(1)).expand(
                comment_ids.size(0), comment_ids.size(1)).to(self.device) < seq_len.unsqueeze(-1)
            # comment_ids = self.batch_token_ids(comment_ids).to(self.device)
        # rewards
        post_ids = batch['post_ids']
        post_mask = batch['post_mask']
        rewards, shaped_reward, rewards_info = self.calculate_reward(
            locals()[f'{self.args.nll_condition}_ids'],
            locals()[f'{self.args.nll_condition}_mask'],
            batch['doc_text'],
            comment_ids,
            comment_texts,
            comment_mask)
        # SQL
        decoder_input_ids = comment_ids[:, :-1]
        target_ids = comment_ids[:, 1:]
        input_ids = torch.cat([post_doc_ids, decoder_input_ids], dim=1)
        input_mask = torch.cat([post_doc_mask, comment_mask[:, :-1]], dim=1)
        with torch.cuda.amp.autocast(self.args.fp16):
            logits_full = self.model(input_ids, attention_mask=input_mask).logits
            logits = logits_full[:, post_doc_ids.size(1):]
            with torch.no_grad():
                t_logits_full = self.t_model(input_ids, attention_mask=input_mask).logits
                t_logits = t_logits_full[:, post_doc_ids.size(1):]
        target_len = seq_len - 1
        # sql loss
        sql_loss, sql_loss_log = soft_q_loss_with_sparse_rewards(
            implementation=self.args.sql_implementation,
            logits=logits.to(shaped_reward),
            logits_=t_logits.to(shaped_reward),
            logits_pi=None,
            actions=target_ids,
            sampled_actions=None,
            rewards=shaped_reward,
            sequence_length=target_len,
            coefficient=None,
            # Do not add margin losses unless the
            # actions are ground truth actions.
            margin_constant=None,
            margin_coefficient=None)
        # log info
        log_info = {}
        nested_log(rewards_info, log_info, 'reward')
        nested_log(sql_loss_log, log_info, '')
        return sql_loss, log_info
        
    def train_step(self, batch: dict, mode: str) -> Tuple[torch.Tensor, dict]:
        if mode == 'MLE':
            raise RuntimeError('MLE is not ready')
            post_doc_ids = batch['post_doc_ids'].to(self.device)
            comment_ids = batch['comment']
            decoder_input = comment_ids[:, :-1]
            target_ids = comment_ids[:, 1:].reshape(-1)
            input_ids = torch.cat([post_doc_ids, decoder_input], dim=1)

            with torch.cuda.amp.autocast(self.args.fp16):
                logits_full = self.model(input_ids)
                logits = logits_full[:, post_doc_ids.size(1):]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       target_ids, 
                                       ignore_index=self.tokenizer.pad_token_id)
            
            info = {'mle/loss': loss.item()}
            return loss, info
        elif mode == 'SQL_OFF' or mode == 'SQL_ON':
            return self.train_step_sql(batch, mode)
        else:
            raise RuntimeError(f'mode {mode} not recognized')

    def train_one_epoch(self, epoch, eval_batches, test_dl_list, test_names):
        self.model.train()
        print(f'Epoch {epoch}')
        total_loss = 0
        t0 = time()
        for i, batch in enumerate(self.train_dl):
            # if i == 15: break
            global_step = len(self.train_dl)*epoch + i
            if global_step < self.warmup_steps:
                train_mode = self.warmup_mode
            else:
                train_mode = self.train_mode
            # setup modes
            if train_mode == 'sql-mix':
                modes = ['SQL_ON', 'SQL_OFF']
            elif train_mode == 'sql-onpolicy':
                modes = ['SQL_ON']
            elif train_mode == 'sql-offpolicy':
                modes = ['SQL_OFF']
            elif train_mode == 'mle':
                modes = ['MLE']
            # update target model
            self.update_target_model(mode=self.args.update, polyak=self.args.polyak)
            # train step
            loss_list = []
            info_dict = {}
            for mode in modes:
                loss, additional_info = self.train_step(batch, mode)
                loss_list.append(loss)
                nested_log(additional_info, info_dict, mode)
            # import pdb; pdb.set_trace()
            loss = torch.stack(loss_list).mean()

            self.opt.zero_grad()

            if self.args.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()

                self.opt.step()

            total_loss += loss.item()
            # tensorboard
            if self.tensorboard_writer is not None:
                for key, val in info_dict.items():
                    self.tensorboard_writer.add_scalar(key, val, global_step)
            if (i+1) % self.log_period == 0:
                print(f'Batch {i+1:8d}/{len(self.train_dl)}, '
                      f'Avg {self.log_period/(time()-t0):2.3f} batch/s, '
                      f'Avg loss {total_loss/(i+1):.4f}')
                t0 = time()
            if (i+1) % eval_batches == 0:
                eval_results = self.evaluate_testsets([self.valid_dl] + test_dl_list, ['val'] + test_names, steps=global_step)
                self.log(dev=eval_results, new=True, epoch=epoch+(i+1)/len(self.train_dl))
                path = (f'reward-{eval_results["val/rewards/reward/mean"]}-'
                        f'll-{eval_results["val/rewards/generator_reward/mean"]}-'
                        f'bleu2-{eval_results["val/rewards/Bleu_2/mean"]}-'
                        f'steps-{global_step}.pt')
                self.save_checkpoint(path)
                self.log(model=path)
        return total_loss / len(self.train_dl)

    @torch.no_grad()
    def evaluate_testsets(self, dl_list, names, model=None, steps=0):
        if model is None:
            model = self.model
        model.eval()
        eval_results = {}
        for dl, name in zip(dl_list, names):
            print(f"Evaluating {name} ...")
            results = self.evaluate(dl, name, model, steps)
            eval_results.update(results)
        if self.args.mlflow:
            import mlflow

            def log_mlflow(args, eval_results):
                mlflow.set_tracking_uri('http://0.0.0.0:5000')
                mlflow.set_experiment(args.exp_name)
                mlflow.log_params(vars(args))
                mlflow.log_param('steps', steps)
                for key, val in eval_results.items():
                    if isinstance(val, (int, float)):
                        mlflow.log_metric(key, val)
                    else:
                        mlflow.log_param(key, val)
                mlflow.end_run()

            log_mlflow(self.args, eval_results)
        return eval_results

    @torch.no_grad()
    def evaluate(self, dl, name, model=None, steps=0):
        if model is None:
            model = self.model
        model.eval()
        # model.float()
        print('Evaluating ...')
        total_loss = 0
        t0 = time()
        posts = []
        rewards_dict = {}
        for idx, batch in enumerate(dl):
            # if idx == 15: break
            cast_to_device(batch, self.device)
            post_doc_ids = batch['post_doc_ids']
            post_ids = batch['post_ids']
            comment_ids = batch['comment']
            decoder_input = comment_ids[:, :-1]
            target_ids = torch.where(batch['comment_mask'] > 0, comment_ids, 
                                     -100)[:, 1:]
            target_ids = target_ids.reshape(-1)
            input_ids = torch.cat([post_doc_ids, decoder_input], dim=-1)
            input_mask = torch.cat([batch['post_doc_mask'], batch['comment_mask'][:, :-1]], dim=1)
            # import pdb; pdb.set_trace()
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                logits_full = model(input_ids, attention_mask=input_mask).logits
                logits = logits_full[:, post_doc_ids.size(1):]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                        target_ids, ignore_index=-100)

            total_loss += loss.item()
            # log
            if (idx+1) % self.log_period == 0:
                print(f'Batch {idx+1:8d}/{len(dl)}, '
                      f'Avg {self.log_period/(time()-t0):2.3f} batch/s, '
                      f'Avg MLE loss {total_loss/(idx+1):.4f}')
                t0 = time()
            # generate (other metrics)
            post_doc_mask = batch['post_doc_mask']
            post_mask = batch['post_mask']
            eos_ids = torch.zeros((post_doc_ids.size(0), 1)).fill_(self.tokenizer.eos_token_id).long().to(self.device)
            # import pdb; pdb.set_trace()
            response_ids = self.model.generate(
                torch.cat([post_doc_ids, eos_ids], dim=-1),
                attention_mask=torch.cat([post_doc_mask, torch.ones_like(eos_ids)], dim=1),
                do_sample=True,
                top_p=0.4,
                max_length=post_doc_ids.size(1)+1+self.args.max_len,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
            response_ids = response_ids[:, post_doc_ids.size(1):]
            comment_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in response_ids]
            seq_len = torch.tensor(get_seq_len(response_ids[:, 1:], self.tokenizer.eos_token_id)).to(self.device) + 1
            comment_mask = torch.arange(response_ids.size(1)).expand(
                response_ids.size(0), response_ids.size(1)).to(self.device) < seq_len.unsqueeze(-1)
            for i in range(response_ids.size(0)):
                posts.append({
                    'post_text': batch['post_text'][i],
                    'doc_text': batch['doc_text'][i],
                    'res': comment_texts[i],
                    'gts': batch['comment_text'][i],
                })
            # rewards
            # response_ids = self.batch_token_ids(response_ids).to(self.device)
            rewards, shaped_reward, rewards_info = self.calculate_reward(
                post_doc_ids if self.args.nll_condition == 'post_doc' else post_ids,
                post_doc_mask if self.args.nll_condition == 'post_doc' else post_mask,
                batch['doc_text'],
                response_ids,
                comment_texts,
                comment_mask,
                tolist=True)
            for key, val in rewards_info.items():
                rewards_dict[key] = rewards_dict.get(key, []) + val
        # save results to file
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, f'output-{name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(save_path, 'w') as f:
            f.write(json.dumps(posts, ensure_ascii=False))
        # calculate metrics
        results = {}
        # import pdb; pdb.set_trace()
        for i, info in enumerate(posts):
            hyp, ref, doc = info['res'], info['gts'], info['doc_text']
            assert hyp != 'NA'
            score = self.scorers.compute_individual_metrics([ref], hyp)
            score['f1'] = F1Metric.compute(hyp, [ref])
            rouge_score = ROUGE_EVALUATOR.get_scores(hyp, ref)
            score['rouge-1'] = rouge_score['rouge-1']['f']
            score['rouge-2'] = rouge_score['rouge-2']['f']
            score['doc-rouge-l'] = self.scorers.compute_individual_metrics([doc], hyp)['ROUGE_L']
            score['kf1'] = F1Metric.compute(hyp, [doc])
            score['human-doc-rouge-l'] = self.scorers.compute_individual_metrics([doc], ref)['ROUGE_L']
            score['human-kf1'] = F1Metric.compute(ref, [doc])
            for key, val in score.items():
                results[key] = results.get(key, 0) + val
        for key in results:
            results[key] /= len(posts)
        results['ppl'] = e ** (total_loss/len(dl))
        # log
        info = {}
        nested_log(results, info, f'{name}/gen-metrics')
        for key, val in rewards_dict.items():
            if isinstance(val, list):
                rewards_dict[key] = torch.tensor(val)
        nested_log(rewards_dict, info, f'{name}/rewards')
        if self.tensorboard_writer:
            for key, val in info.items():
                self.tensorboard_writer.add_scalar(key, val, global_step=steps)
        info[f'{name}/infer_file'] = save_path
        return info

    def register_and_early_stop(self, res, metric, epoch):
        """Internally save the model if the model is the currently best one,
        and return whether early stop.
        """
        scores = [d[metric] for d in self.history]
        path = (f'reward-{res["val/rewards/reward/mean"]}-'
               f'll-{res["val/rewards/generator_reward/mean"]}-'
               f'bleu2-{res["val/rewards/Bleu_2/mean"]}-'
               f'epoch-{epoch}.pt')
        if metric == 'ppl' or 'loss' in metric:
            best_score = min(scores)
            self.best_num = scores.index(best_score)
            if  res[metric] <= best_score:
                self.cnt = 0
                if 'local_rank' not in dir(self.args) or self.args.local_rank == 0:
                    self.save_checkpoint(path)
                self.log(model=path, epoch=epoch)
        else:
            best_score = max(scores)
            self.best_num = scores.index(best_score)
            if res[metric] >= best_score:
                self.cnt = 0
                if 'local_rank' not in dir(self.args) or self.args.local_rank == 0:
                    self.save_checkpoint(path)
                self.log(model=path, epoch=epoch)
        self.cnt += 1
        if self.cnt >= self.early_stopping:
            return True
        return False

    def save_checkpoint(self, name):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        path = os.path.join(self.save_dir, name)
        if self.args.fp16:
            checkpoint = {
                'model': self.model.state_dict(),
                't_model': self.t_model.state_dict(),
                'opt': self.opt.state_dict(),
                'scaler': self.scaler.state_dict(),
                'history': self.history
            }
        else:
            checkpoint = {
                'model': self.model.state_dict(),
                't_model': self.t_model.state_dict(),
                'opt': self.opt.state_dict(),
                'history': self.history
            }
        torch.save(checkpoint, path)

    def load_model_from_checkpoint(self, path, load_history=False, 
            from_distributed=False, load_target_model=False):
        print(f'Loading model from {path}')
        checkpoint = torch.load(path)
        if from_distributed:
            self.model.load_state_dict(checkpoint['module'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        if load_target_model:
            print('Loading target model ...')
            self.t_model.load_state_dict(checkpoint['t_model'])
        self.opt.load_state_dict(checkpoint['opt'])
        if load_history:
            self.history = checkpoint['history']
        if self.args.fp16 and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

    def train(self, eval_batches):
        print(f'Start training for maximum {self.max_epoch} epochs, '
              f'early-stopping on {self.early_stopping_metric} with '
              f'patient {self.early_stopping}')
        for epoch in range(self.max_epoch):
            avg_loss = self.train_one_epoch(epoch, eval_batches)
            self.log(new=True, epoch=epoch+1)
            self.log(train=avg_loss)
            dev_result = self.evaluate(self.valid_dl, epoch=epoch+1)
            self.log(dev=dev_result)
            print('Saving checkpoint ...')
            path = (f'reward-{dev_result["val/rewards/reward/mean"]}-'
                    f'll-{dev_result["val/rewards/generator_reward/mean"]}-'
                    f'bleu2-{dev_result["val/rewards/Bleu_2/mean"]}-'
                    f'epoch-{epoch}.pt')
            self.save_checkpoint(path)
            self.log(model=path)
            if self.register_and_early_stop(dev_result, self.early_stopping_metric, epoch+1):
                print('Early stopping ...')
                break

        # evaluate on test set
        self.load_model_from_checkpoint(os.path.join(self.save_dir, self.history[self.best_num]['model_path']))
        test_result = self.evaluate(self.test_dl)
        self.log(test=test_result, new=True, epoch=self.history[self.best_num]['epoch'])
        self.save_results()

    def train_tests(self, eval_batches, test_dl_list, test_names):
        if self.args.evaluate_only:
            print('No training, direct evaluation:')
            result = self.evaluate_testsets([self.valid_dl] + test_dl_list, ['val'] + test_names)
            self.log(test=result, new=True, epoch=0)
            return
        
        print(f'Start training for maximum {self.max_epoch} epochs, '
              f'early-stopping on {self.early_stopping_metric} with '
              f'patient {self.early_stopping}')
        for epoch in range(self.max_epoch):
            avg_loss = self.train_one_epoch(epoch, eval_batches, test_dl_list, test_names)
            self.log(new=True, epoch=epoch+1)
            self.log(train=avg_loss)
            global_steps = len(self.train_dl)*(epoch+1)
            dev_result = self.evaluate_testsets([self.valid_dl] + test_dl_list, ['val'] + test_names, steps=global_steps)
            self.log(dev=dev_result)
            print('Saving checkpoint ...')
            path = (f'reward-{dev_result["val/rewards/reward/mean"]}-'
                    f'll-{dev_result["val/rewards/generator_reward/mean"]}-'
                    f'bleu2-{dev_result["val/rewards/Bleu_2/mean"]}-'
                    f'steps-{global_steps}.pt')
            self.save_checkpoint(path)
            self.log(model=path)
            if self.register_and_early_stop(dev_result, self.early_stopping_metric, epoch+1):
                print('Early stopping ...')
                break

        # evaluate on test set
        self.load_model_from_checkpoint(os.path.join(self.save_dir, self.history[self.best_num]['model_path']))
        test_result = self.evaluate(self.test_dl, 'test')
        self.log(test=test_result, new=True, epoch=self.history[self.best_num]['epoch'])
        self.save_results()


class EncoderDecoderTrainer(Trainer):

    def __init__(self, model, t_model, opt, tokenizer, reward_tokenizer, 
            train_dl, valid_dl, test_dl, args):
        super().__init__(model, t_model, opt, tokenizer, 
            train_dl, valid_dl, test_dl, args)
        self.r_tokenizer = reward_tokenizer

    def train_step_sql(self, batch: dict, mode: str) -> Tuple[torch.Tensor, dict]:
        batch = cast_to_device(batch, self.device)
        if mode == 'SQL_OFF':
            # import pdb; pdb.set_trace()
            r_comment_ids = batch['r_comment']
            r_comment_mask = batch['r_comment_mask']
            comment_ids = batch['comment']
            comment_mask = batch['comment_mask']
            comment_texts = batch['comment_text']
            seq_len = comment_mask.sum(dim=1)
            post_doc_ids = batch['pseudo_post_doc_ids']
            post_doc_mask = batch['pseudo_post_doc_mask']
            doc_text = batch['pseudo_doc_text']
            r_post_doc_ids = batch['r_pseudo_post_doc_ids']
            r_post_doc_mask = batch['r_pseudo_post_doc_mask']
        else:
            # import pdb; pdb.set_trace()
            post_doc_ids = batch['pseudo_post_doc_ids']
            post_doc_mask = batch['pseudo_post_doc_mask']
            doc_text = batch['pseudo_doc_text']
            r_post_doc_ids = batch['r_pseudo_post_doc_ids']
            r_post_doc_mask = batch['r_pseudo_post_doc_mask']
            # eos_ids = torch.zeros((post_doc_ids.size(0), 1)).fill_(self.tokenizer.eos_token_id).to(self.device).long()
            # input_ids = torch.cat([post_doc_ids, eos_ids], dim=1)
            # input_mask = torch.cat([post_doc_mask, torch.ones_like(eos_ids)], dim=1)
            with torch.no_grad():
                comment_ids = self.model.generate(
                    post_doc_ids,
                    attention_mask=post_doc_mask,
                    do_sample=True,
                    top_p=0.4,
                    max_length=self.args.max_len,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            # comment_ids = comment_ids[:, input_ids.size(1)-1:]
            # remove [CLS], [SEP], ([PAD])
            comment_texts = self.tokenizer.batch_decode(comment_ids, skip_special_tokens=True)
            # seq_len = torch.tensor(get_seq_len(comment_ids[:, 1:], self.tokenizer.eos_token_id)).to(self.device) + 1
            seq_len = torch.tensor(get_seq_len(comment_ids[:, 1:], self.tokenizer.eos_token_id)).to(self.device) + 1
            comment_mask = torch.arange(comment_ids.size(1)).expand(
                comment_ids.size(0), comment_ids.size(1)).to(self.device) < seq_len.unsqueeze(-1)
            # comment_ids = self.batch_token_ids(comment_ids).to(self.device)
            r_comment_ids, r_comment_mask = tokenize_comment_batch(comment_texts, self.r_tokenizer)
            r_comment_ids = r_comment_ids.to(self.device)
            r_comment_mask = r_comment_mask.to(self.device)
        # rewards
        post_ids = batch['post_ids']
        post_mask = batch['post_mask']
        r_post_ids = batch['r_post_ids']
        r_post_mask = batch['r_post_mask']
        rewards, shaped_reward, rewards_info = self.calculate_reward(
            locals()[f'r_{self.args.nll_condition}_ids'],
            locals()[f'r_{self.args.nll_condition}_mask'],
            doc_text,
            r_comment_ids,
            comment_texts,
            r_comment_mask)
        # SQL
        decoder_input_ids = comment_ids[:, :-1]
        decoder_input_mask = comment_mask[:, :-1]
        target_ids = comment_ids[:, 1:]
        # input_ids = torch.cat([post_doc_ids, decoder_input_ids], dim=1)
        # input_mask = torch.cat([post_doc_mask, comment_mask[:, :-1]], dim=1)
        with torch.cuda.amp.autocast(self.args.fp16):
            logits = self.model(post_doc_ids, attention_mask=post_doc_mask,
                                decoder_input_ids=decoder_input_ids,
                                decoder_attention_mask=decoder_input_mask).logits
            # logits = logits_full[:, post_doc_ids.size(1):]
            with torch.no_grad():
                t_logits = self.t_model(post_doc_ids, attention_mask=post_doc_mask,
                                        decoder_input_ids=decoder_input_ids,
                                        decoder_attention_mask=decoder_input_mask).logits
                # t_logits = t_logits_full[:, post_doc_ids.size(1):]
        target_len = seq_len - 1
        # sql loss
        sql_loss, sql_loss_log = soft_q_loss_with_sparse_rewards(
            implementation=self.args.sql_implementation,
            logits=logits.to(shaped_reward),
            logits_=t_logits.to(shaped_reward),
            logits_pi=None,
            actions=target_ids,
            sampled_actions=None,
            rewards=shaped_reward,
            sequence_length=target_len,
            coefficient=None,
            # Do not add margin losses unless the
            # actions are ground truth actions.
            margin_constant=None,
            margin_coefficient=None)
        # log info
        log_info = {}
        nested_log(rewards_info, log_info, 'reward')
        nested_log(sql_loss_log, log_info, '')
        return sql_loss, log_info

    @torch.no_grad()
    def evaluate(self, dl, name, model=None, steps=0):
        if model is None:
            model = self.model
        model.eval()
        # model.float()
        print('Evaluating ...')
        total_loss = 0
        t0 = time()
        posts = []
        rewards_dict = {}
        for idx, batch in enumerate(dl):
            # if idx == 15: break
            cast_to_device(batch, self.device)
            post_doc_ids = batch['post_doc_ids']
            post_ids = batch['post_ids']
            comment_ids = batch['comment']
            decoder_input = comment_ids[:, :-1]
            target_ids = torch.where(batch['comment_mask'] > 0, comment_ids, 
                                     -100)[:, 1:]
            target_ids = target_ids.reshape(-1)
            # input_ids = torch.cat([post_doc_ids, decoder_input], dim=-1)
            # input_mask = torch.cat([batch['post_doc_mask'], batch['comment_mask'][:, :-1]], dim=1)
            # import pdb; pdb.set_trace()
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                logits = model(post_doc_ids, attention_mask=batch['post_doc_mask'],
                               decoder_input_ids=decoder_input,
                               decoder_attention_mask=batch['comment_mask'][:, :-1]).logits
                # logits = logits_full[:, post_doc_ids.size(1):]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), 
                                        target_ids, ignore_index=-100)

            total_loss += loss.item()
            # log
            if (idx+1) % self.log_period == 0:
                print(f'Batch {idx+1:8d}/{len(dl)}, '
                      f'Avg {self.log_period/(time()-t0):2.3f} batch/s, '
                      f'Avg MLE loss {total_loss/(idx+1):.4f}')
                t0 = time()
            # generate (other metrics)
            post_doc_mask = batch['post_doc_mask']
            post_mask = batch['post_mask']
            # eos_ids = torch.zeros((post_doc_ids.size(0), 1)).fill_(self.tokenizer.eos_token_id).long().to(self.device)
            # import pdb; pdb.set_trace()
            response_ids = self.model.generate(
                post_doc_ids,
                attention_mask=post_doc_mask,
                do_sample=True,
                top_p=0.4,
                max_length=self.args.max_len+1,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
            # response_ids = response_ids[:, post_doc_ids.size(1):]
            comment_texts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            # seq_len = torch.tensor(get_seq_len(response_ids[:, 1:], self.tokenizer.eos_token_id)).to(self.device) + 1
            # comment_mask = torch.arange(response_ids.size(1)).expand(
            #     response_ids.size(0), response_ids.size(1)).to(self.device) < seq_len.unsqueeze(-1)
            r_comment_ids, r_comment_mask = tokenize_comment_batch(comment_texts, self.r_tokenizer)
            r_comment_ids = r_comment_ids.to(self.device)
            r_comment_mask = r_comment_mask.to(self.device)
            for i in range(response_ids.size(0)):
                posts.append({
                    'post_text': batch['post_text'][i],
                    'doc_text': batch['doc_text'][i],
                    'res': comment_texts[i],
                    'gts': batch['comment_text'][i],
                })
            # rewards
            # response_ids = self.batch_token_ids(response_ids).to(self.device)
            rewards, shaped_reward, rewards_info = self.calculate_reward(
                batch['r_post_doc_ids'] if self.args.nll_condition == 'post_doc' else batch['r_post_ids'],
                batch['r_post_doc_mask'] if self.args.nll_condition == 'post_doc' else batch['r_post_mask'],
                batch['doc_text'],
                r_comment_ids,
                comment_texts,
                r_comment_mask,
                tolist=True)
            for key, val in rewards_info.items():
                rewards_dict[key] = rewards_dict.get(key, []) + val
        # save results to file
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        save_path = os.path.join(self.save_dir, f'output-{name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(save_path, 'w') as f:
            f.write(json.dumps(posts, ensure_ascii=False))
        # calculate metrics
        results = {}
        # import pdb; pdb.set_trace()
        for i, info in enumerate(posts):
            hyp, ref, doc = info['res'], info['gts'], info['doc_text']
            assert hyp != 'NA'
            score = self.scorers.compute_individual_metrics([ref], hyp)
            score['f1'] = F1Metric.compute(hyp, [ref])
            rouge_score = ROUGE_EVALUATOR.get_scores(hyp, ref)
            score['rouge-1'] = rouge_score['rouge-1']['f']
            score['rouge-2'] = rouge_score['rouge-2']['f']
            score['doc-rouge-l'] = self.scorers.compute_individual_metrics([doc], hyp)['ROUGE_L']
            score['kf1'] = F1Metric.compute(hyp, [doc])
            score['human-doc-rouge-l'] = self.scorers.compute_individual_metrics([doc], ref)['ROUGE_L']
            score['human-kf1'] = F1Metric.compute(ref, [doc])
            for key, val in score.items():
                results[key] = results.get(key, 0) + val
        for key in results:
            results[key] /= len(posts)
        results['ppl'] = e ** (total_loss/len(dl))
        # log
        info = {}
        nested_log(results, info, f'{name}/gen-metrics')
        for key, val in rewards_dict.items():
            if isinstance(val, list):
                rewards_dict[key] = torch.tensor(val)
        nested_log(rewards_dict, info, f'{name}/rewards')
        if self.tensorboard_writer:
            for key, val in info.items():
                self.tensorboard_writer.add_scalar(key, val, global_step=steps)
        info[f'{name}/infer_file'] = save_path
        return info