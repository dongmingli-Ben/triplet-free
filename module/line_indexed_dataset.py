import json
import os
import pickle
import time
import torch
import mmap
from PIL import Image, UnidentifiedImageError

class DatasetLineIndexed:
    """This dataset loads the text data into memory lazily, 
    therefore should be used when the dataset is large"""

    def __init__(self, data_dir, image_dir, keep_sticker, 
            type, tokenizer, transform, args):
        self.dir = data_dir
        self.image_dir = image_dir
        self.keep_sticker = keep_sticker
        self.type = type # train, dev, test
        self.tokenizer = tokenizer
        self.transform = transform
        self.args = args
        # load data
        data_file, data_index, comment_index, cum_comment_count = self.build_index(self.dir)
        self.data_file = data_file
        self.line_index = data_index  # currently the posts are not grouped by len
        self.index = comment_index
        self.comment_cnt = cum_comment_count
        # self.sample_idx = sample_idx
        self.len = self.comment_cnt[-1]

    def build_index(self, directory):
        name = f'{self.type}.txt'
        path = os.path.join(directory, name)
        cache_path = os.path.join(self.args.cache_dir, f'{name}.index')
        if os.path.exists(cache_path):
            print(f'Index file for {path} found, skip building index')
            with open(cache_path, 'rb') as f:
                data_index = pickle.load(f)
                comment_index = pickle.load(f)
                cum_comment_count = pickle.load(f)
            with open(path, 'r') as f:
                mmap_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            return mmap_file, data_index, comment_index, cum_comment_count
        else:
            print('No cached index file found.')
            print(f'Building index for {path} ...')
            t0 = time.time()
            data_index = [0]
            index = []
            comment_cnt = [0]
            with open(path, 'r') as f:
                mmap_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                for i, line in enumerate(f):
                    item = json.loads(line)
                    data_index.append(data_index[-1] + len(line.encode()))
                    num = len(item['comments'])
                    comment_cnt.append(comment_cnt[-1] + num)
                    index += [i] * num
            assert comment_cnt[-1] == len(index)
            if not os.path.exists(self.args.cache_dir):
                os.makedirs(self.args.cache_dir)
            with open(cache_path, 'wb') as f:
                pickle.dump(data_index, f)
                pickle.dump(index, f)
                pickle.dump(comment_cnt, f)
            print(f'Index built in {time.time()-t0:.2f}s')
            return mmap_file, data_index, index, comment_cnt

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # import pdb; pdb.set_trace()
        post_num = self.index[index]
        comment_num = index - self.comment_cnt[post_num]
        start, length = self.line_index[post_num], self.line_index[post_num+1] - self.line_index[post_num]
        self.data_file.seek(start)
        post = json.loads(self.data_file.read(length))
        comment = post['comments'][comment_num]
        # post text
        post_text = post['text'] if self.keep_sticker else post['plain_text']
        post_token_ids = self.tokenizer(post_text, 
                                        add_special_tokens=False)['input_ids']
        # post images
        image_paths = post['images']
        images = []
        # t0 = time.time()
        for i, path in enumerate(image_paths):
            # throw some images
            if i == self.args.max_images:
                break
            try:
                image = Image.open(os.path.join(self.image_dir, path))
                # import pdb; pdb.set_trace()
                image = self.transform(image)
                images.append(image)
            except UnidentifiedImageError:
                print(path, 'broken, skipped')
                continue
            except FileNotFoundError:
                print(path, 'not found, skipped')
                continue
        if len(images) > 0:
            images = torch.stack(images)
        else:
            images = torch.tensor([])
        # print('images', time.time() - t0)
        # comment text
        comment_text = comment['text'] if self.keep_sticker else comment['plain_text']
        comment_token_ids = self.tokenizer(comment_text, 
                                          add_special_tokens=True)['input_ids']

        data = {
            'post_token_ids': post_token_ids,
            'post_text': post_text,
            'images': images,
            'image_paths': image_paths,
            'comment_token_ids': comment_token_ids,
            'comment_text': comment_text,
        }

        return data

    def merge(self, id_list, pad_idx=None):
        pad_idx = pad_idx if pad_idx else self.tokenizer.pad_token_id
        num = len(id_list)
        max_len = max(map(len, id_list))
        out = torch.zeros((num, max_len)
            ).fill_(pad_idx).long()
        for i, ids in enumerate(id_list):
            out[i, :len(ids)] = torch.tensor(ids)
        return out

    def collate_fn(self, data):
        # t0 = time.time()
        info = {}
        for key in data[0].keys():
            info[key] = []
        for d in data:
            for key, value in d.items():
                info[key].append(value)
        # post 
        post_ids = info['post_token_ids']
        # add padding
        post_ids = self.merge(post_ids)
        comment_ids = self.merge(info['comment_token_ids'])
        # images (merge into a tensor)
        image_split = [0]
        for image in info['images']:
            image_split.append(image_split[-1] + image.size(0))
        image_batch = torch.cat(info['images'], dim=0)
        # import pdb; pdb.set_trace()

        out = {
            'post': post_ids,
            'comment': comment_ids,
            'image_batch': image_batch,
            'image_split': image_split,
        }
            
        # other data
        preserve_list = ['post_text', 'comment_text', 'image_paths']
        for key in preserve_list:
            if key in info:
                out[key] = info[key]
        return out