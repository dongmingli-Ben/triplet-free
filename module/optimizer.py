import torch
from module.model import EncoderDecoderModel

def build_optimizer(model: EncoderDecoderModel, args):
    visual_optimizer = getattr(torch.optim, args.optimizer)(
        model.visual_embedding.parameters(), args.lr_visual)
    visual_param_ids = [id(param) for param in model.visual_embedding.parameters()]
    text_param = filter(lambda p: not (id(p) in visual_param_ids), model.parameters())
    text_optimizer = getattr(torch.optim, args.optimizer)(text_param, args.lr_text)
    return visual_optimizer, text_optimizer