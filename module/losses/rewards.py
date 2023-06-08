from transformers import AutoModelForCausalLM
import torch
import torch.nn.functional as F
from nlgeval import NLGEval


class RewardHelper:

    def __init__(self, args):
        self.model = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print('Comment generator reward model using', self.device)
        self.scorer = NLGEval(no_skipthoughts=True, no_glove=True,
                              metrics_to_omit=['METEOR', 'ROUGE_L', 'CIDEr'])
        self.args = args

    def load_comment_generator(self):
        print(f'Loading comment generator from {self.args.reward_model} ...')
        self.model = AutoModelForCausalLM.from_pretrained(self.args.reward_model).to(self.device)
        
    @torch.no_grad()
    def comment_generator_reward(self, post_token_ids: torch.Tensor, 
            images: torch.Tensor, split_list: list,
            comment_token_ids: torch.Tensor,
            pad_token_id=0, mode='token'):
        if self.model is None:
            self.load_comment_generator()
        decoder_input_ids = comment_token_ids[..., :-1]
        target_ids = comment_token_ids[..., 1:]
        bs = target_ids.size(0)
        seq_len = target_ids.size(1)
        # import pdb; pdb.set_trace()
        logits = self.model(post_token_ids, images,
                    decoder_input_ids, split_list)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                            target_ids.reshape(-1),
                            ignore_index=pad_token_id,
                            reduction='none')
        nll = nll.reshape(bs, seq_len)
        # import pdb; pdb.set_trace()
        nll = nll.sum(dim=1)
        if mode == 'token':
            lengths = (target_ids != 0).sum(dim=1)
            return -nll / lengths
        elif mode == 'sentence':
            return -nll
        else:
            raise ValueError('mode should be token or sentence')
        
    @torch.no_grad()
    def comment_generator_reward_eng(self, 
            post_token_ids: torch.Tensor, 
            post_mask: torch.Tensor,
            comment_token_ids: torch.Tensor,
            comment_mask: torch.Tensor,
            pad_token_id=0, mode='token'):
        if self.model is None:
            self.load_comment_generator()
        decoder_input_ids = comment_token_ids[..., :-1]
        target_ids = comment_token_ids[..., 1:]
        target_mask = comment_mask[..., 1:]
        bs = target_ids.size(0)
        seq_len = target_ids.size(1)
        # import pdb; pdb.set_trace()
        input_ids = torch.cat([post_token_ids, decoder_input_ids], dim=-1)
        input_mask = torch.cat([post_mask, comment_mask[..., :-1]], dim=-1)
        logits = self.model(input_ids, attention_mask=input_mask).logits[:, post_token_ids.size(1):]
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                            target_ids.reshape(-1),
                            reduction='none')
        nll = nll.reshape(bs, seq_len) * target_mask.float()
        # import pdb; pdb.set_trace()
        nll = nll.sum(dim=1)
        if mode == 'token':
            lengths = target_mask.sum(dim=1)
            return -nll / lengths
        elif mode == 'sentence':
            return -nll
        else:
            raise ValueError('mode should be token or sentence')

    def knowledge_reward(self, docs: list, hyps: list, hyp_is_spaced=False,
            doc_id_spaced=False, n='2'):
        # bp = exp(1 - len(doc)/len(hyp)) if len(doc) > len(hyp) else 1
        if not hyp_is_spaced:
            hyps = [' '.join(list(hyp)) for hyp in hyps]
        if not doc_id_spaced:
            docs = [' '.join(list(doc)) for doc in docs]
        scores = [self.scorer.compute_individual_metrics([doc], hyp) for doc, hyp in zip(docs, hyps)]
        scores = self.merge_dicts(scores, to_tensor=True)
        key = f'Bleu_{n}'
        return scores[key], scores

    def merge_dicts(self, dicts, to_tensor=False):
        info = {}
        for d in dicts:
            for key, val in d.items():
                info[key] = info.get(key, []) + [val]
        if not to_tensor:
            return info
        for key, val in info.items():
            if isinstance(val, list):
                info[key] = torch.tensor(val).to(self.device)
        return info

# helper = RewardHelper()
# comment_generator_reward = helper.comment_generator_reward_eng
# knowledge_reward = helper.knowledge_reward