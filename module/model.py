import torch
import torch.nn as nn
from module.models.modeling_bart import BartForConditionalGeneration
from module.visual import build_visual_model
from module.module import PositionalEncoding, TextEmbedding, expand_positional_encoding

PREIRAINED_MODELS = {
    'fnlp/bart-base-chinese': BartForConditionalGeneration,
    'fnlp/bart-large-chinese': BartForConditionalGeneration,
}

def merge_image_text(images, texts, image_mask, text_mask):
    assert (images.size(0) == texts.size(0) == image_mask.size(0) == text_mask.size(0)) \
        or (images.size(0) == image_mask.size(0) == 0 and texts.size(0) == text_mask.size(0)) \
        or (images.size(0) == image_mask.size(0) and texts.size(0) == text_mask.size(0) == 0)
    bs = max(images.size(0), texts.size(0))
    dim = max(images.size(-1), texts.size(-1))
    image_length = torch.sum(image_mask, dim=-1).long()
    text_length = torch.sum(text_mask, dim=-1).long()
    length = image_length+text_length
    max_len = max(length).item()
    embed = torch.zeros((bs, max_len, dim)).to(images)
    mask = torch.zeros((bs, max_len)).to(images)
    for i in range(bs):
        _length = length[i]
        if torch.numel(images) > 0:
            embed[i, :image_length[i]] = images[i, :image_length[i]]
        if torch.numel(texts) > 0:
            embed[i, _length-text_length[i]:_length] = texts[i, :text_length[i]]
        mask[i, :_length] = 1
    return embed, mask

class EncoderDecoderModel(nn.Module):

    def __init__(self, tokenizer, transformer: nn.Module,
            visual_embedding: nn.Module, text_embedding: nn.Module):
        super(EncoderDecoderModel, self).__init__()
        self.transformer = transformer
        self.visual_embedding = visual_embedding
        self.text_embedding = text_embedding
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, input_images=None, 
            decoder_input_ids=None, image_split_list=None,
            add_cls_to_input=False, return_logits_only=True):
        # import pdb; pdb.set_trace()
        text_embed = self.text_embedding(input_ids)
        images, image_mask = self.visual_embedding(input_images, image_split_list)

        if add_cls_to_input:
            cls_tokens = torch.tensor(
                [[self.tokenizer.cls_token_id]]*(len(image_split_list)-1)).to(input_ids)
            cls_embed = self.text_embedding(cls_tokens)
            cls_mask = torch.ones_like(cls_tokens)
            # add cls to the front
            images, image_mask = merge_image_text(cls_embed, images, cls_mask, image_mask)

        text_mask = (input_ids != self.tokenizer.pad_token_id)
        # encoder_mask = torch.cat([image_mask, text_mask], dim=1)
        # input_embed = torch.cat([images, text_embed], dim=1)
        input_embed, encoder_mask = merge_image_text(images, text_embed, image_mask, text_mask)
        out = self.transformer(
            inputs_embeds=input_embed,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input_ids
        )
        if return_logits_only:
            logits = out['logits']
            return logits
        return out


def build_model(args, tokenizer):
    # import pdb; pdb.set_trace()
    if args.backbone != '':
        print(f'Building model from pretrained model {args.backbone}')
        backbone = PREIRAINED_MODELS[args.backbone].from_pretrained(args.backbone)
        if args.backbone in ['fnlp/bart-base-chinese', 'fnlp/bart-large-chinese']:
            args.dim = backbone.get_encoder().get_input_embeddings().weight.size(-1)
            text_position_embed = backbone.get_encoder().embed_positions
            text_position_embed = expand_positional_encoding(text_position_embed, 
                args.max_text_features, copy=False)
            if args.share_positional_embedding:
                text_position_embed = expand_positional_encoding(text_position_embed, 
                    args.max_image_features, copy=False)
            text_embed = TextEmbedding(
                backbone.get_encoder().get_input_embeddings(),
                text_position_embed, 
                embed_scale=backbone.get_encoder().embed_scale)
    print(f'Building visual model from {args.visual_extractor}')
    # positional encoding for visual
    if args.share_positional_embedding or args.init_from_text_pe:
        visual_position_embed = text_position_embed
        if args.init_from_text_pe:
            visual_position_embed = expand_positional_encoding(text_position_embed, 
                args.max_image_features, copy=True)
    else:
        # not tested yet
        visual_position_embed = PositionalEncoding(args.max_image_features, args.dim)
    visual_model = build_visual_model(args.visual_extractor, visual_position_embed,
                                      args)
    model = EncoderDecoderModel(tokenizer, backbone, visual_model, text_embed)
    return model