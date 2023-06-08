from typing import List, Tuple
import torch
import torch.nn as nn
import torchvision.models as models
from transformers.models.vit.modeling_vit import PatchEmbeddings

class VisualExtractorResnet(nn.Module):

    def __init__(self, visual_extractor, args):
        super(VisualExtractorResnet, self).__init__()
        model = getattr(models, visual_extractor)(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.proj = nn.Linear(2048, args.dim)

    def forward(self, images):
        patch_feats = self.model(images)
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        patch_feats = self.proj(patch_feats)
        return patch_feats

class VisualExtractorViT(nn.Module):
    """A wrapper around the ```PatchEmbeddings```"""

    def __init__(self, name, args):
        super(VisualExtractorViT, self).__init__()
        self.args = args
        self.embed = PatchEmbeddings(patch_size=self.args.patch_size, embed_dim=args.dim)

    def forward(self, x):
        return self.embed.forward(x)

class VisualEmbed(nn.Module):

    def __init__(self, visual_extractor, position_embed, id_embed):
        super(VisualEmbed, self).__init__()
        self.model = visual_extractor
        self.position_embed = position_embed
        self.id_embed = id_embed
        # self.device = device
        self.dim = self.id_embed.weight.size(-1)

    def merge_tensors(self, tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        num = len(tensors)
        dim = self.dim
        max_len = max(map(lambda x: x.size(0), tensors))
        device = self.id_embed.weight.device
        mask = torch.zeros((num, max_len)).to(device)
        out = torch.zeros((num, max_len, dim)).to(device)
        for i, tensor in enumerate(tensors):
            # check for no images
            if tensor.size(0) == 0:
                continue
            mask[i, :tensor.size(0)] = 1
            out[i, :tensor.size(0)] = tensor
        return out, mask

    def forward(self, images, split_list=None):
        """return features, mask (0 for masked).
        If split_list is None, it is assumed that the images are from the
        same post."""
        device = self.id_embed.weight.device
        if torch.numel(images) == 0:
            # no images in a batch
            # import pdb; pdb.set_trace()
            return torch.tensor([]).to(device), torch.tensor([]).to(device)
        image_feats = self.model(images)
        # id embedding + positional encoding
        if split_list is not None:
            feats = []
            # id embedding
            if self.id_embed:
                ids = []
                for i in range(len(split_list)-1):
                    ids += list(range(split_list[i+1]-split_list[i]))
                image_feats += self.id_embed(torch.tensor(ids).to(device)
                    ).reshape(image_feats.size(0), 1, image_feats.size(-1))
            # positional encoding
            for i in range(len(split_list)-1):
                feat_list = [image_feats[j] for j in range(split_list[i], split_list[i+1])]
                if len(feat_list) > 0:
                    feat = torch.cat(feat_list, dim=0)
                else:
                    # no images for a post
                    # import pdb; pdb.set_trace()
                    feat = torch.tensor([]).to(device)
                feats.append(feat)
            # merge into a batch and then positional encoding
            # import pdb; pdb.set_trace()
            feats, mask = self.merge_tensors(feats)
            feats += self.position_embed(feats.shape)
            return feats, mask
        else:
            # not tested yet
            num = image_feats.size(0)
            feats = image_feats
            if self.id_embed is not None:
                feats = feats + self.id_embed(torch.tensor(range(num)).to(self.device)
                    ).reshape(num, 1, feats.size(-1))
            feat_list = [feats[i] for  i in range(num)]
            feats = torch.cat(feat_list, dim=0)
            feats += self.position_embed((1, feats.size(0))
                ).reshape(feats.shape)
            return feats, torch.ones((feats.size(0),)).to(self.device)

VISUAL_EXTRACTORS = {
    'resnet18': VisualExtractorResnet,
    'resnet34': VisualExtractorResnet,
    'resnet50': VisualExtractorResnet,
    'resnet101': VisualExtractorResnet,
    'resnet152': VisualExtractorResnet,
    'vit': VisualExtractorViT,
}


def build_visual_model(visual_extractor, position_embed=None, args=None):
    extractor = VISUAL_EXTRACTORS[visual_extractor](visual_extractor, args)
    # if position_embed is None:
    #     position_embed = nn.Embedding(position_len, args.dim)
    # else:
    #     position_embed = expand_positional_encoding(position_embed, position_len, copy=True)
    id_embed = nn.Embedding(args.max_images, args.dim)
    # ddp
    # device = torch.device(args.local_rank) if 'local_rank' in dir(args) else torch.device(args.device)
    visual_embed = VisualEmbed(extractor, position_embed, id_embed)
    return visual_embed