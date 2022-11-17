import pytorch_lightning as pl
import torch
import argparse

from transformers import CLIPTokenizer
from data.engines import ImageStore, AspectBucket, AspectBucketSampler, AspectDataset
from lib.model import load_model
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description="Waifu Diffusion Finetuner ported to Lightning")
parser.add_argument('-c', '--config', type=str, required=True, help="Path to the configuration file")
args = parser.parse_args()

pathToConf = args.config
config = OmegaConf.load(pathToConf)

def main():
    torch.manual_seed(config.trainer.seed)
    pathToModelDiffuser = config.checkpoint.input
    resolution = config.dataset.resolution

    tokenizer = CLIPTokenizer.from_pretrained(pathToModelDiffuser, subfolder="tokenizer")

    #do as haru's rather than naifus
    #load dataset
    store = ImageStore(config.dataset.path)
    dataset = AspectDataset(store, tokenizer)
    bucket = AspectBucket(
        store=store,
        num_buckets=config.dataset.buckets.num_buckets,
        batch_size=config.trainer.batch_size,
        bucket_side_min=config.dataset.buckets.bucket_side.min,
        bucket_side_max=config.dataset.buckets.bucket_side.max,
        bucket_side_increment=64,
        max_image_area=int(resolution * resolution),
        max_ratio=2.0
    )
    sampler = AspectBucketSampler(
        bucket=bucket,
        num_replicas=1, #because we are not doing distributed and thats the default
        rank=0, #same reason as above
    )

    print(f'STORE_LEN: {len(store)}')

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    trainer = pl.Trainer(
        logger = None,
        strategy = None,
        **config.lightning
    )

    model = load_model(config, len(train_dataloader), tokenizer)

    trainer.tune(model=model)
    trainer.fit(
        model=model,
        ckpt_path=None,
        train_dataloaders=train_dataloader
    )

if __name__ == "__main__":
    main()


