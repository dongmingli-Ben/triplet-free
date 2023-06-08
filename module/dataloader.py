from torch.utils.data import DataLoader
from torchvision import transforms

from module.line_indexed_dataset import DatasetLineIndexed


def build_dataloader(args, tokenizer, split='train'):
    print(f'Building {split} dataloader ...')
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = DatasetLineIndexed(args.data_dir, args.image_dir, args.keep_sticker,
                                split, tokenizer, transform, args)

    dataloader = DataLoader(dataset, args.bs, True if split == 'train' else False,
                            num_workers=args.workers, collate_fn=dataset.collate_fn)
    return dataloader