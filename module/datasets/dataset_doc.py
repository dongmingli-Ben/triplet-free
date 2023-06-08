import json
import os
import pickle
import time
import torch
import mmap
from random import random, choice
from torch.utils.data import DataLoader

class DocDatasetLineIndexed:
    """This dataset loads the text data into memory lazily, 
    therefore should be used when the dataset is large"""

    def __init__(self, data_dir, type, tokenizer, reward_tokenizer, args):
        self.dir = data_dir
        self.type = type # train, dev, test
        self.tokenizer = tokenizer
        self.r_tokenizer = reward_tokenizer
        self.args = args
        # load data
        data_file, data_index, doc_index, cum_doc_count = self.build_index(self.dir)
        self.data_file = data_file
        self.line_index = data_index  # currently the posts are not grouped by len
        self.index = doc_index
        self.doc_count = cum_doc_count
        # self.sample_idx = sample_idx
        self.len = self.doc_count[-1]

    def build_index(self, directory):
        name = f'{self.type}.txt'
        path = os.path.join(directory, name)
        cache_path = os.path.join(self.args.cache_dir, f'{name}.index')
        if os.path.exists(cache_path):
            print(f'Index file for {path} found, skip building index')
            with open(cache_path, 'rb') as f:
                data_index = pickle.load(f)
                doc_index = pickle.load(f)
                cum_doc_count = pickle.load(f)
            with open(path, 'r') as f:
                mmap_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            return mmap_file, data_index, doc_index, cum_doc_count
        else:
            print('No cached index file found.')
            print(f'Building index for {path} ...')
            t0 = time.time()
            data_index = [0]
            index = []
            doc_cnt = [0]
            with open(path, 'r') as f:
                mmap_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                for i, line in enumerate(f):
                    data_index.append(data_index[-1] + len(line.encode()))
                    num = 1
                    doc_cnt.append(doc_cnt[-1] + num)
                    index += [i] * num
            assert doc_cnt[-1] == len(index)
            if not os.path.exists(self.args.cache_dir):
                os.makedirs(self.args.cache_dir)
            with open(cache_path, 'wb') as f:
                pickle.dump(data_index, f)
                pickle.dump(index, f)
                pickle.dump(doc_cnt, f)
            print(f'Index built in {time.time()-t0:.2f}s')
            return mmap_file, data_index, index, doc_cnt

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        post_num = self.index[index]
        doc_num = index - self.doc_count[post_num]
        start, length = self.line_index[post_num], self.line_index[post_num+1] - self.line_index[post_num]
        self.data_file.seek(start)
        post = json.loads(self.data_file.read(length))
        doc = post['knowledge']
        doc_token_ids = self.tokenizer(doc, add_special_tokens=False)['input_ids']
        r_doc_token_ids = self.r_tokenizer(doc, add_special_tokens=False)['input_ids']
        pseudo_doc = post['pseudo_knowledge']
        pseudo_doc_token_ids = self.tokenizer(pseudo_doc, add_special_tokens=False)['input_ids']
        r_pseudo_doc_token_ids = self.r_tokenizer(pseudo_doc, add_special_tokens=False)['input_ids']
        # comments = list(filter(lambda s: len(s['plain_text'].strip()) > 0, post['comments']))
        comment = post['text']
        # post text
        post_text = post['dialog_history'].replace('<|endoftext|>', self.tokenizer.eos_token)
        post_text = post_text.split(self.tokenizer.eos_token)[-1] \
            if self.args.last_utterence_only else post_text
        post_token_ids = self.tokenizer(
            'dialogue: ' + post_text, 
            add_special_tokens=False)['input_ids']
        r_post_token_ids = self.r_tokenizer(post_text.replace(self.tokenizer.eos_token, 
                                                              self.r_tokenizer.eos_token), 
                                            add_special_tokens=False)['input_ids']
        # post images
        # print('images', time.time() - t0)
        # comment text
        comment_text = comment
        # import pdb; pdb.set_trace()
        # in the origin code, add_special_tokens=True for dialogpt tokenizers
        # it is fine for dialogpt tokenizer, because dialogpt tokenizer does not add
        # eos_token to the end of sentence
        # t5 generate tokens starting from <pad>
        comment_token_ids = [self.tokenizer.pad_token_id] + self.tokenizer(comment_text, 
                                          add_special_tokens=False)['input_ids'] \
                            + [self.tokenizer.eos_token_id]
        r_comment_token_ids = [self.r_tokenizer.eos_token_id] + self.r_tokenizer(comment_text, 
                                          add_special_tokens=False)['input_ids'] \
                            + [self.r_tokenizer.eos_token_id]

        data = {
            'post_text': post_text,
            'post_token_ids': post_token_ids,
            'r_post_token_ids': r_post_token_ids,
            'comment_text': comment_text,
            'comment_token_ids': comment_token_ids,
            'r_comment_token_ids': r_comment_token_ids,
            'doc_text': doc,
            'doc_token_ids': doc_token_ids,
            'r_doc_token_ids': r_doc_token_ids,
            'pseudo_doc_text': pseudo_doc,
            'pseudo_doc_token_ids': pseudo_doc_token_ids,
            'r_pseudo_doc_token_ids': r_pseudo_doc_token_ids,
        }

        return data

    def merge(self, id_list, pad_idx=None):
        pad_idx = pad_idx if pad_idx else self.tokenizer.eos_token_id
        num = len(id_list)
        max_len = max(map(len, id_list))
        out = torch.zeros((num, max_len)
            ).fill_(pad_idx).long()
        mask = torch.zeros_like(out)
        for i, ids in enumerate(id_list):
            out[i, :len(ids)] = torch.LongTensor(ids)
            mask[i, :len(ids)] = 1
        return out, mask

    def collate_fn(self, data):
        # t0 = time.time()
        info = {}
        for key in data[0].keys():
            info[key] = []
        for d in data:
            for key, value in d.items():
                info[key].append(value)
        # import pdb; pdb.set_trace()
        post_ids, post_mask = self.merge(info['post_token_ids'], self.tokenizer.eos_token_id)
        doc_ids, doc_mask = self.merge(info['doc_token_ids'], self.tokenizer.eos_token_id)
        pseudo_doc_ids, pseudo_doc_mask = self.merge(info['pseudo_doc_token_ids'], self.tokenizer.eos_token_id)
        comment_ids, comment_mask = self.merge(info['comment_token_ids'], self.tokenizer.eos_token_id)
        # for reward model
        r_post_ids, r_post_mask = self.merge(info['r_post_token_ids'], self.r_tokenizer.eos_token_id)
        r_doc_ids, r_doc_mask = self.merge(info['r_doc_token_ids'], self.r_tokenizer.eos_token_id)
        r_pseudo_doc_ids, r_pseudo_doc_mask = self.merge(info['r_pseudo_doc_token_ids'], self.r_tokenizer.eos_token_id)
        r_comment_ids, r_comment_mask = self.merge(info['r_comment_token_ids'], self.r_tokenizer.eos_token_id)
        eos_ids = torch.zeros((len(data), 1)).fill_(self.tokenizer.eos_token_id).long()
        r_eos_ids = torch.zeros((len(data), 1)).fill_(self.r_tokenizer.eos_token_id).long()
        eos_mask = torch.ones_like(eos_ids)
        # post [eos] doc
        post_doc_ids = torch.cat([
            post_ids, eos_ids, doc_ids], dim=1)
        post_doc_mask = torch.cat([
            post_mask, eos_mask, doc_mask], dim=1)
        pseudo_post_doc_ids = torch.cat([
            post_ids, eos_ids, pseudo_doc_ids], dim=1)
        pseudo_post_doc_mask = torch.cat([
            post_mask, eos_mask, pseudo_doc_mask], dim=1)
        # for reward model
        r_post_doc_ids = torch.cat([
            r_post_ids, r_eos_ids, r_doc_ids], dim=1)
        r_post_doc_mask = torch.cat([
            r_post_mask, eos_mask, r_doc_mask], dim=1)
        r_pseudo_post_doc_ids = torch.cat([
            r_post_ids, r_eos_ids, r_pseudo_doc_ids], dim=1)
        r_pseudo_post_doc_mask = torch.cat([
            r_post_mask, eos_mask, r_pseudo_doc_mask], dim=1)
        # cut sequence that is too long
        post_doc_ids = post_doc_ids[:, -300:]
        post_doc_mask = post_doc_mask[:, -300:]
        pseudo_post_doc_ids = pseudo_post_doc_ids[:, -300:]
        pseudo_post_doc_mask = pseudo_post_doc_mask[:, -300:]
        # for reward model
        r_post_doc_ids = r_post_doc_ids[:, -300:]
        r_post_doc_mask = r_post_doc_mask[:, -300:]
        r_pseudo_post_doc_ids = r_pseudo_post_doc_ids[:, -300:]
        r_pseudo_post_doc_mask = r_pseudo_post_doc_mask[:, -300:]
        # post ids
        # images (merge into a tensor)
        # import pdb; pdb.set_trace()

        out = {
            'post_doc_ids': post_doc_ids,
            'post_doc_mask': post_doc_mask,
            'pseudo_post_doc_ids': pseudo_post_doc_ids,
            'pseudo_post_doc_mask': pseudo_post_doc_mask,
            'post_ids': post_ids,
            'post_mask': post_mask,
            'comment': comment_ids,
            'comment_mask': comment_mask,
            # for reward model
            'r_post_doc_ids': r_post_doc_ids,
            'r_post_doc_mask': r_post_doc_mask,
            'r_pseudo_post_doc_ids': r_pseudo_post_doc_ids,
            'r_pseudo_post_doc_mask': r_pseudo_post_doc_mask,
            'r_post_ids': r_post_ids,
            'r_post_mask': r_post_mask,
            'r_comment': r_comment_ids,
            'r_comment_mask': r_comment_mask,
        }
            
        # other data
        preserve_list = ['post_text', 'comment_text', 'doc_text', 'pseudo_doc_text']
        for key in preserve_list:
            if key in info:
                out[key] = info[key]
        # print(time.time() - t0)
        return out

def build_dataloader(args, tokenizer, reward_tokenizer, split='train'):
    print(f'Building {split} dataloader ...')
    dataset = DocDatasetLineIndexed(args.data_dir, split, tokenizer, reward_tokenizer, args)

    dataloader = DataLoader(dataset, args.bs, True if split == 'train' else False,
                            num_workers=args.workers, collate_fn=dataset.collate_fn)
    return dataloader

def build_dataloader_from_file(args, path, tokenizer, reward_tokenizer, shuffle):
    print(f'Building dataloader for {path} ...')
    split = path.split('/')[-1].split('.')[0]
    dataset = DocDatasetLineIndexed(os.path.dirname(path), split, tokenizer, reward_tokenizer, args)

    dataloader = DataLoader(dataset, args.bs, shuffle=shuffle,
                            num_workers=args.workers, collate_fn=dataset.collate_fn)
    return dataloader

if __name__ == '__main__':
    from argparse import Namespace
    from transformers import AutoTokenizer
    
    args = Namespace(**{
        'data_dir': 'data/wow',
        'bs': 4,
        'workers': 0,
        'max_images': 9,
        'cache_dir': 'test',
        'last_utterence_only': False
    })
    print(args)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    r_tokenizer = AutoTokenizer.from_pretrained('dialogpt-medium')

    # dataloader = build_dataloader(args, tokenizer, 'train')
    dataloader = build_dataloader_from_file(args, 'data/wizint/test-short-knowledge.txt', tokenizer, r_tokenizer, False)

    for i, batch in enumerate(dataloader):
        print(batch)
        import pdb; pdb.set_trace()