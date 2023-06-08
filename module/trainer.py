import pickle
import json
from typing import List
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from datetime import datetime
from time import time
from nlgeval import NLGEval
from math import e
import pandas as pd

class Trainer:

    def __init__(self, model, opt, tokenizer,
            train_dl, valid_dl, test_dl, args):
        self.device = torch.device(args.device)
        self.model = model
        self.opt = opt
        self.tokenizer = tokenizer
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        if tokenizer:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.args = args
        self.early_stopping = args.early_stopping
        self.early_stopping_metric = args.early_stopping_metric
        self.save_dir = os.path.join(args.save_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.max_epoch = args.max_epoch
        self.history = []
        self.log_period = args.log_period
        self.scorers = NLGEval(no_skipthoughts=True, no_glove=True)  # loads the models
        # torch amp
        if args.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        if args.checkpoint:
            self.load_model_from_checkpoint(args.checkpoint, load_history=args.load_history,
                                            from_distributed=self.args.from_distributed)
        # for early stopping
        self.cnt = 0
        self.metric_score = None
        self.best_num = -1   # to find the best model

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        with open(os.path.join(args.save_dir, 'structure.txt'), 'wb') as f:
            pickle.dump(args, f)

    def log(self, train=None, dev=None, test=None, new=False, verbose=True,
            model=None, epoch=None):
        if new:
            self.history.append({})
        if train:
            self.history[-1]['train_loss'] = train
            if verbose:
                print(f'Training loss {train:.3f}')
        if dev:
            for key, val in dev.items():
                self.history[-1][key] = val
                if verbose:
                    if isinstance(val, (int, float)):
                        print(f'Valid {key:10s}: {val:3.4f}')
                    else:
                        print(f'Valid {key:10s}: {val}')
            self.history[-1]['type'] = 'valid'
        if test:
            for key, val in test.items():
                self.history[-1][key] = val
                if verbose:
                    if isinstance(val, (int, float)):
                        print(f'Test {key:10s}: {val:3.4f}')
                    else:
                        print(f'Test {key:10s}: {val}')
            self.history[-1]['type'] = 'test'
        if model:
            self.history[-1]['model_path'] = model
        if epoch:
            self.history[-1]['epoch'] = epoch
        self.remove_extra_checkpoints()

    def remove_extra_checkpoints(self):
        pairs = []
        for dirpath, dirname, files in os.walk(self.save_dir):
            for file in files:
                if file.endswith('.pt'):
                    match = re.search('ppl-(\d*?.\d*?)-', file)
                    if match is not None:
                        ppl = match.group(1)
                        pairs.append((
                            os.path.join(dirpath, file),
                            eval(ppl)
                        ))
        if len(pairs) <= self.args.max_checkpoint_num:
            return
        pairs = sorted(pairs, key=lambda t: t[1])
        for i in range(self.args.max_checkpoint_num, len(pairs)):
            os.remove(pairs[i][0])
        
    def save_checkpoint(self, name):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        path = os.path.join(self.save_dir, name)
        if self.args.fp16:
            checkpoint = {
                'model': self.model.state_dict(),
                'opt_v': self.opt_v.state_dict(),
                'opt_t': self.opt_t.state_dict(),
                'scaler': self.scaler.state_dict(),
                'history': self.history
            }
        else:
            checkpoint = {
                'model': self.model.state_dict(),
                'opt_v': self.opt_v.state_dict(),
                'opt_t': self.opt_t.state_dict(),
                'history': self.history
            }
        torch.save(checkpoint, path)

    def load_model_from_checkpoint(self, path, load_history=False, from_distributed=False):
        print(f'Loading model from {path}')
        checkpoint = torch.load(path)
        if from_distributed:
            self.model.load_state_dict(checkpoint['module'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        self.opt_v.load_state_dict(checkpoint['opt_v'])
        self.opt_t.load_state_dict(checkpoint['opt_t'])
        if load_history:
            self.history = checkpoint['history']
        if self.args.fp16 and 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])

    def save_results(self):
        for data in self.history:
            if 'train_loss' not in data:
                data['train_loss'] = None
        df = pd.DataFrame.from_records(self.history)
        df.to_csv(os.path.join(self.save_dir, 'records.csv'))

    def register_and_early_stop(self, res, metric, epoch):
        """Internally save the model if the model is the currently best one,
        and return whether early stop.
        """
        scores = [d[metric] for d in self.history]
        if metric == 'ppl' or 'loss' in metric:
            best_score = min(scores)
            self.best_num = scores.index(best_score)
            if  res[metric] <= best_score:
                self.cnt = 0
                if 'local_rank' not in dir(self.args) or self.args.local_rank == 0:
                    self.save_checkpoint(f'ppl-{res["ppl"]}-epoch-{epoch}.pt')
                self.log(model=f'ppl-{res["ppl"]}-epoch-{epoch}.pt', epoch=epoch)
        else:
            best_score = max(scores)
            self.best_num = scores.index(best_score)
            if res[metric] >= best_score:
                self.cnt = 0
                if 'local_rank' not in dir(self.args) or self.args.local_rank == 0:
                    self.save_checkpoint(f'ppl-{res["ppl"]}-epoch-{epoch}.pt')
                self.log(model=f'ppl-{res["ppl"]}-epoch-{epoch}.pt', epoch=epoch)
        self.cnt += 1
        if self.cnt >= self.early_stopping:
            return True
        return False

    def train(self, eval_batches):
        print(f'Start training for maximum {self.max_epoch} epochs, '
              f'early-stopping on {self.early_stopping_metric} with '
              f'patient {self.early_stopping}')
        for epoch in range(self.max_epoch):
            avg_loss = self.train_one_epoch(epoch, eval_batches)
            self.log(new=True, epoch=epoch+1)
            self.log(train=avg_loss)
            dev_result = self.evaluate(self.valid_dl)
            self.log(dev=dev_result)
            if self.register_and_early_stop(dev_result, self.early_stopping_metric, epoch+1):
                print('Early stopping ...')
                print('Saving checkpoint ...')
                self.save_checkpoint(f'ppl-{dev_result["ppl"]}-epoch-{epoch+1}.pt')
                self.log(model=f'ppl-{dev_result["ppl"]}-epoch-{epoch+1}.pt')
                break

        # evaluate on test set
        self.load_model_from_checkpoint(os.path.join(self.save_dir, self.history[self.best_num]['model_path']))
        test_result = self.evaluate(self.test_dl)
        self.log(test=test_result, new=True, epoch=self.history[self.best_num]['epoch'])
        self.save_results()

    def train_one_epoch(self, epoch, eval_batches):
        self.model.train()
        print(f'Epoch {epoch}')
        total_loss = 0
        t0 = time()
        for i, batch in enumerate(self.train_dl):
            # if i == 150: break
            post_ids = batch['post'].to(self.device)
            comment_ids = batch['comment'].to(self.device)
            image_batch = batch['image_batch'].to(self.device)
            split_list = batch['image_split']
            decoder_input = comment_ids[:, :-1]
            target_ids = comment_ids[:, 1:].reshape(-1)

            with torch.cuda.amp.autocast(self.args.fp16):
                logits = self.model(post_ids, image_batch,
                                    decoder_input, split_list)
                loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), 
                                        target_ids)

            self.opt_v.zero_grad()
            self.opt_t.zero_grad()

            if self.args.fp16:
                self.scaler.scale(loss).backward()
                # import pdb; pdb.set_trace()
                self.scaler.step(self.opt_v)
                self.scaler.step(self.opt_t)
                self.scaler.update()
            else:
                loss.backward()

                self.opt_v.step()
                self.opt_t.step()

            total_loss += loss.item()
            if (i+1) % self.log_period == 0:
                print(f'Batch {i+1:8d}/{len(self.train_dl)}, '
                      f'Avg {self.log_period/(time()-t0):2.3f} batch/s, '
                      f'Avg loss {total_loss/(i+1):.4f}')
                t0 = time()
            if (i+1) % eval_batches == 0:
                eval_results = self.evaluate(self.valid_dl)
                self.log(dev=eval_results, new=True, epoch=epoch+(i+1)/len(self.train_dl))
                self.save_checkpoint(f'ppl-{eval_results["ppl"]}-epoch-{epoch+(i+1)/len(self.train_dl)}.pt')
                self.log(model=f'ppl-{eval_results["ppl"]}-epoch-{epoch+(i+1)/len(self.train_dl)}.pt')
        return total_loss / len(self.train_dl)

    @torch.no_grad()
    def evaluate(self, dl, model=None):
        if model is None:
            model = self.model
        model.eval()
        # model.float()
        print('Evaluating ...')
        total_loss = 0
        t0 = time()
        posts = {}
        for idx, batch in enumerate(dl):
            # if idx == 100: break
            post_ids = batch['post'].to(self.device)
            comment_ids = batch['comment'].to(self.device)
            image_batch = batch['image_batch'].to(self.device)
            split_list = batch['image_split']
            decoder_input = comment_ids[:, :-1]
            target_ids = comment_ids[:, 1:].reshape(-1)

            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                logits = model(post_ids, image_batch,
                                    decoder_input, split_list)
                loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), 
                                        target_ids)

            total_loss += loss.item()
            # log
            if (idx+1) % self.log_period == 0:
                print(f'Batch {idx+1:8d}/{len(dl)}, '
                      f'Avg {self.log_period/(time()-t0):2.3f} batch/s, '
                      f'Avg loss {total_loss/(idx+1):.4f}')
                t0 = time()
            # generate (other metrics)
            # import pdb; pdb.set_trace()
            post_keywords = []
            for post_keywords_ids in batch['post']:
                text = self.tokenizer.decode(post_keywords_ids.tolist())
                text = text.replace(self.tokenizer.pad_token, '').strip()
                post_keywords.append(text)
            for text, comment, image_paths, post_text in zip(post_keywords, 
                    batch['comment_text'], batch['image_paths'], batch['post_text']):
                if text not in posts:
                    posts[text] = {'res': None, 'gts': [], 'images': image_paths,
                                   'post_text': post_text}
                posts[text]['gts'].append(
                    self.tokenizer.decode(self.tokenizer(
                        comment, add_special_tokens=False)['input_ids']))
            indices = []
            for i, text in enumerate(post_keywords):
                if posts[text]['res'] is None:
                    indices.append(i)
                    posts[text]['res'] = 'NA'
            if len(indices) == 0:
                continue
            indices = torch.tensor(indices).to(self.device)
            post_ids = post_ids[indices]
            images = []
            new_split = [0]
            for i in indices:
                images.append(image_batch[split_list[i]:split_list[i+1]])
                new_split.append(new_split[-1] + split_list[i+1] - split_list[i])
            image_batch = torch.cat(images, dim=0)
            split_list = new_split
            response_ids = self.decode_batch(post_ids, image_batch, split_list, model)
            # import pdb; pdb.set_trace()
            for i, j in enumerate(indices):
                posts[post_keywords[j]]['res'] = self.tokenizer.decode(response_ids[i], 
                                                                    skip_special_tokens=True)
        # save results to file
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, f'output-{str(datetime.now())}.json'), 'w') as f:
            f.write(json.dumps(posts, ensure_ascii=False))
        # calculate metrics
        results = {}
        for i, info in enumerate(posts.values()):
            hyp, refs = info['res'], info['gts']
            assert hyp != 'NA'
            res = self.scorers.compute_individual_metrics(refs, hyp)
            for key, val in res.items():
                results[key] = results.get(key, 0) + val
        for key in results:
            results[key] /= len(posts)
        results['ppl'] = e ** (total_loss/len(dl))
        return results

    def decode_batch(self, post_ids, image_batch, split_list, model=None, decode=None) -> List[List[int]]:
        if model is None:
            model = self.model
        if decode is None:
            decode = self.args.decode
        batch_size = post_ids.size(0)
        if 'bos_token_id' in dir(self.tokenizer) and self.tokenizer.bos_token_id:
            bos_token_id = self.tokenizer.bos_token_id
        else:
            bos_token_id = self.tokenizer.cls_token_id
        if 'eos_token_id' in dir(self.tokenizer) and self.tokenizer.eos_token_id:
            eos_token_id = self.tokenizer.eos_token_id
        else:
            eos_token_id = self.tokenizer.sep_token_id
        # import pdb; pdb.set_trace()
        decoder_input_ids = torch.tensor([bos_token_id]*batch_size
            ).reshape(post_ids.size(0), 1).to(self.device)
        done = torch.tensor([False]*batch_size).to(self.device)
        for i in range(self.args.max_len):
            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                logits = model(post_ids, image_batch,
                                    decoder_input_ids, split_list)
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
                
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids], dim=1)
            end = (next_token_ids == eos_token_id).reshape(-1)
            done = done | end
            if not torch.any(~done):
                break
        output = []
        for response in decoder_input_ids.tolist():
            try:
                index = response.index(eos_token_id)
            except ValueError:
                index = len(response) - 1
            output.append(response[:index+1])
        return output

    def decode_batch_with_prob(self, post_ids, image_batch, split_list, model=None) -> List[List[int]]:
        if model is None:
            model = self.model
        batch_size = post_ids.size(0)
        if 'bos_token_id' in dir(self.tokenizer) and self.tokenizer.bos_token_id:
            bos_token_id = self.tokenizer.bos_token_id
        else:
            bos_token_id = self.tokenizer.cls_token_id
        if 'eos_token_id' in dir(self.tokenizer) and self.tokenizer.eos_token_id:
            eos_token_id = self.tokenizer.eos_token_id
        else:
            eos_token_id = self.tokenizer.sep_token_id
        decoder_input_ids = torch.tensor([bos_token_id]*batch_size
            ).reshape(post_ids.size(0), 1).to(self.device)
        done = torch.tensor([False]*batch_size).to(self.device)
        log_prob = torch.tensor([0]*batch_size).to(self.device)
        for i in range(self.args.max_len):
            logits = model(post_ids, image_batch,
                                decoder_input_ids, split_list)
            prob = F.softmax(logits[:, -1, :], dim=-1)
            if self.args.decode == 'top-k':
                values, token_ids = torch.topk(prob, k=self.args.top_k)
                indices = torch.randint(0, self.args.top_k, (batch_size,)
                    ).reshape(-1, 1).to(self.device)
                next_token_ids = torch.gather(token_ids, 1, indices)
                next_token_probs = torch.gather(values, 1, indices)
            elif self.args.decode == 'greedy':
                values, token_ids = torch.topk(prob, k=1)
                next_token_ids = token_ids
                next_token_probs = values
            elif self.args.decode == 'random':
                # not tested
                next_token_ids = torch.multinomial(prob, 1)
                next_token_probs = torch.gather(prob, 1, next_token_ids)
            elif self.args.decode == 'combine':
                if i == 0:
                    # top-k for the first token
                    values, token_ids = torch.topk(prob, k=batch_size)
                    indices = torch.arange(batch_size).reshape(-1, 1).to(self.device)
                    next_token_ids = torch.gather(token_ids, 1, indices)
                    next_token_probs = torch.gather(values, 1, indices)
                else:
                    values, token_ids = torch.topk(prob, k=1)
                    next_token_ids = token_ids
                    next_token_probs = values
                    # greedy for the rest
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
                next_token_probs = torch.gather(prob, 1, next_token_ids)


            decoder_input_ids = torch.cat([decoder_input_ids, next_token_ids], dim=1)
            log_prob = log_prob + torch.log(next_token_probs.reshape(-1)) * (~done)
            end = (next_token_ids == eos_token_id).reshape(-1)
            done = done | end
            if not torch.any(~done):
                break
        output = []
        for response in decoder_input_ids.tolist():
            try:
                index = response.index(eos_token_id)
            except ValueError:
                index = len(response) - 1
            output.append(response[:index+1])
        return output, log_prob

class TrainerUnlikelihood(Trainer):

    def train_one_epoch(self, epoch, eval_batches):
        self.model.train()
        print(f'Epoch {epoch}')
        total_loss = 0
        total_loss_mle, total_loss_un = 0, 0
        t0 = time()
        for i, batch in enumerate(self.train_dl):
            # if i == 150: break
            post_ids = batch['post'].to(self.device)
            comment_ids = batch['comment'].to(self.device)
            image_batch = batch['image_batch'].to(self.device)
            split_list = batch['image_split']
            decoder_input = comment_ids[:, :-1]
            target_ids = comment_ids[:, 1:]
            # import pdb; pdb.set_trace()
            with torch.cuda.amp.autocast(self.args.fp16):
                logits = self.model(post_ids, image_batch,
                                    decoder_input, split_list)
                loss_mle = self.loss_fn(logits.reshape(-1, logits.size(-1)), 
                                        target_ids.reshape(-1))

                # unlikelihood
                one_minus_prob = torch.clamp(1-F.softmax(logits, dim=-1), min=1e-5)
                # Make 'the triangle'.
                ctx_cands = target_ids.unsqueeze(1).expand(
                    target_ids.size(0), target_ids.size(1), target_ids.size(1))
                ctx_cands_ = (torch.zeros_like(ctx_cands) + self.tokenizer.pad_token_id)
                ctx_cands_ = ctx_cands_.triu()
                ctx_cands = ctx_cands.tril(-1) + ctx_cands_

                # Don't include the target for that timestep as a negative target.
                ctx_cands = ctx_cands.masked_fill(ctx_cands == target_ids.unsqueeze(2), 
                    self.tokenizer.pad_token_id)
                negative_targets = torch.zeros_like(one_minus_prob).scatter_(2, ctx_cands, 1)
                # remove pad from negative targets
                negative_targets[:, :, self.tokenizer.pad_token_id] = 0

                # unlikelihood loss
                loss_un = -torch.log(one_minus_prob) * negative_targets
                loss_un = loss_un.sum(dim=-1)
                mask = (target_ids != self.tokenizer.pad_token_id)
                loss_un = loss_un * mask
                loss_un = loss_un.sum()# / mask.sum()

                loss = loss_mle + loss_un * self.args.unlikelihood_ratio


            self.opt_v.zero_grad()
            self.opt_t.zero_grad()

            if self.args.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt_v)
                self.scaler.unscale_(self.opt_t)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
                self.scaler.step(self.opt_v)
                self.scaler.step(self.opt_t)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)

                self.opt_v.step()
                self.opt_t.step()

            total_loss += loss.item()
            total_loss_mle += loss_mle.item()
            total_loss_un += loss_un.item()
            if (i+1) % self.log_period == 0:
                print(f'Batch {i+1:8d}/{len(self.train_dl)}, '
                      f'Avg {self.log_period/(time()-t0):2.3f} batch/s, '
                      f'Avg loss {total_loss/(i+1):.4f}, '
                      f'MLE loss {total_loss_mle/(i+1):.4f}, '
                      f'Unlikelihood loss {total_loss_un/(i+1):.4f}')
                t0 = time()
            if (i+1) % eval_batches == 0:
                eval_results = self.evaluate(self.valid_dl)
                self.log(dev=eval_results, new=True, epoch=epoch+(i+1)/len(self.train_dl))
                self.save_checkpoint(f'ppl-{eval_results["ppl"]}-epoch-{epoch+(i+1)/len(self.train_dl)}.pt')
                self.log(model=f'ppl-{eval_results["ppl"]}-epoch-{epoch+(i+1)/len(self.train_dl)}.pt')
        return total_loss / len(self.train_dl)

    @torch.no_grad()
    def evaluate(self, dl, model=None):
        if model is None:
            model = self.model
        model.eval()
        # model.float()
        print('Evaluating ...')
        total_loss = 0
        total_loss_mle, total_loss_un = 0, 0
        t0 = time()
        posts = {}
        for idx, batch in enumerate(dl):
            # if idx == 10: break
            post_ids = batch['post'].to(self.device)
            comment_ids = batch['comment'].to(self.device)
            image_batch = batch['image_batch'].to(self.device)
            split_list = batch['image_split']
            decoder_input = comment_ids[:, :-1]
            target_ids = comment_ids[:, 1:]

            with torch.cuda.amp.autocast(enabled=self.args.fp16):
                logits = self.model(post_ids, image_batch,
                                    decoder_input, split_list)
                loss_mle = self.loss_fn(logits.reshape(-1, logits.size(-1)), 
                                        target_ids.reshape(-1))

                # unlikelihood
                one_minus_prob = torch.clamp(1-F.softmax(logits, dim=-1), min=1e-5)
                # Make 'the triangle'.
                ctx_cands = target_ids.unsqueeze(1).expand(
                    target_ids.size(0), target_ids.size(1), target_ids.size(1))
                ctx_cands_ = (torch.zeros_like(ctx_cands) + self.tokenizer.pad_token_id)
                ctx_cands_ = ctx_cands_.triu()
                ctx_cands = ctx_cands.tril(-1) + ctx_cands_

                # Don't include the target for that timestep as a negative target.
                ctx_cands = ctx_cands.masked_fill(ctx_cands == target_ids.unsqueeze(2), 
                    self.tokenizer.pad_token_id)
                negative_targets = torch.zeros_like(one_minus_prob).scatter_(2, ctx_cands, 1)
                # remove pad from negative targets
                negative_targets[:, :, self.tokenizer.pad_token_id] = 0

                # unlikelihood loss
                loss_un = -torch.log(one_minus_prob) * negative_targets
                loss_un = loss_un.sum(dim=-1)
                mask = (target_ids != self.tokenizer.pad_token_id)
                loss_un = loss_un * mask
                loss_un = loss_un.sum()# / mask.sum()

                loss = loss_mle + loss_un * self.args.unlikelihood_ratio

            total_loss += loss.item()
            total_loss_mle += loss_mle.item()
            total_loss_un += loss_un.item()
            # log
            if (idx+1) % self.log_period == 0:
                print(f'Batch {i+1:8d}/{len(dl)}, '
                      f'Avg {self.log_period/(time()-t0):2.3f} batch/s, '
                      f'Avg loss {total_loss/(i+1):.4f}, '
                      f'MLE loss {total_loss_mle/(i+1):.4f}, '
                      f'Unlikelihood loss {total_loss_un/(i+1):.4f}')
                t0 = time()
            # generate (other metrics)
            post_keywords = []
            for post_keywords_ids in batch['post']:
                text = self.tokenizer.decode(post_keywords_ids.tolist())
                text = text.replace(self.tokenizer.pad_token, '').strip()
                post_keywords.append(text)
            for text, comment, image_paths, post_text in zip(post_keywords, 
                    batch['comment_text'], batch['image_paths'], batch['post_text']):
                if text not in posts:
                    posts[text] = {'res': None, 'gts': [], 'images': image_paths,
                                   'post_text': post_text}
                posts[text]['gts'].append(
                    self.tokenizer.decode(self.tokenizer(
                        comment, add_special_tokens=False)['input_ids']))
            indices = []
            for i, text in enumerate(post_keywords):
                if posts[text]['res'] is None:
                    indices.append(i)
                    posts[text]['res'] = 'NA'
            if len(indices) == 0:
                continue
            indices = torch.tensor(indices).to(self.device)
            post_ids = post_ids[indices]
            images = []
            new_split = [0]
            for i in indices:
                images.append(image_batch[split_list[i]:split_list[i+1]])
                new_split.append(new_split[-1] + split_list[i+1] - split_list[i])
            image_batch = torch.cat(images, dim=0)
            split_list = new_split
            response_ids = self.decode_batch(post_ids, image_batch, split_list, model)
            # import pdb; pdb.set_trace()
            for i, j in enumerate(indices):
                posts[post_keywords[j]]['res'] = self.tokenizer.decode(response_ids[i], 
                                                                    skip_special_tokens=True)
        # save results to file
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, f'output-{str(datetime.now())}.json'), 'w') as f:
            f.write(json.dumps(posts, ensure_ascii=False))
        # calculate metrics
        results = {}
        for i, info in enumerate(posts.values()):
            hyp, refs = info['res'], info['gts']
            assert hyp != 'NA'
            res = self.scorers.compute_individual_metrics(refs, hyp)
            for key, val in res.items():
                results[key] = results.get(key, 0) + val
        for key in results:
            results[key] /= len(posts)
        results['ppl'] = e ** (total_loss_mle/len(dl))
        results['loss'] = total_loss/len(dl)
        return results