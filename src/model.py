import torch
import os
import pytorch_lightning
from x_unet import XUnet
from matplotlib import pyplot as plt
import matplotlib

class RiverModel(pytorch_lightning.LightningModule):
    def __init__(
        self,
        *,
        lr,
        log_images_every_train,
        log_images_every_validation,
        dim,
        dim_mults,
        num_blocks_per_stage,
        num_self_attn_per_stage,
        nested_unet_depths,
        nested_unet_dim,
        channels,
        use_convnext,
        resnet_groups,
        consolidate_upsample_fmaps,
        weight_standardize,
        attn_heads,
        attn_dim_head,
        bce_weights
    ):
        super(RiverModel, self).__init__()
        
        self.lr = lr
        self.log_images_every = dict(train=log_images_every_train, validation=log_images_every_validation)
        self.bce_weights = bce_weights
        
        dim_mults=list(dim_mults)
        num_blocks_per_stage=list(num_blocks_per_stage)
        num_self_attn_per_stage=list(num_self_attn_per_stage)
        nested_unet_depths=list(nested_unet_depths)

        self.save_hyperparameters()

        self.unet = XUnet(
            dim=dim,
            dim_mults=dim_mults,
            num_blocks_per_stage=num_blocks_per_stage,
            num_self_attn_per_stage=num_self_attn_per_stage,
            nested_unet_depths=nested_unet_depths,
            nested_unet_dim=nested_unet_dim,
            channels=channels,
            use_convnext=use_convnext,
            resnet_groups=resnet_groups,
            consolidate_upsample_fmaps=consolidate_upsample_fmaps,
            weight_standardize=weight_standardize,
            attn_heads=attn_heads,
            attn_dim_head=attn_dim_head
        )

        self.step_count = dict(train=0, validation=0)

    def training_step(self, batch):
        return self.step("train", batch)
    
    def validation_step(self, batch, it):
        return self.step("validation", batch)

    def forward(self, x):
        predicted_mask = self.unet(x)
        return predicted_mask

    def step(self, step, batch):
        raster, mask, river = batch
        predicted_mask = self(raster)

        if self.bce_weights:
            pos_weight = torch.ones_like(mask)
            water_count = (mask == 1).sum()
            nowater_count = (mask == 0).sum()
            water_coef = nowater_count / water_count
            pos_weight[mask == 1] = water_coef
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        
        loss = loss_fn(predicted_mask, mask)

        with torch.no_grad():        
            predicted_classes = torch.sigmoid(predicted_mask)
            predicted_classes[predicted_classes < 0.5] = 0
            predicted_classes[predicted_classes >= 0.5] = 1
            
            actual_classes = mask.int()

            error = (predicted_mask - mask).abs()
            accuracy = (predicted_classes == actual_classes).float()
            correct_water_mask = ((predicted_classes == 1) * (actual_classes == 1))
            water_precision = ((predicted_classes == 1) * (actual_classes == 1)).sum() / (actual_classes == 1).sum()

            correct_nowater_mask = ((predicted_classes == 0) * (actual_classes == 0))
            nowater_precision = ((predicted_classes == 0) * (actual_classes == 0)).sum() / (actual_classes == 0).sum()

            self.log(f"{step}/metrics/error", error.mean().item())
            self.log(f"{step}/metrics/accuracy", accuracy.mean().item())
            self.log(f"{step}/metrics/water_precision", water_precision.item())
            self.log(f"{step}/metrics/nowater_precision", water_precision.item())
            self.log(f"{step}/loss", loss.item(), prog_bar=True)

            if (self.step_count[step] % self.log_images_every[step]) == 0 and self.logger is not None:
                for i in range(0, 3):
                    matplotlib.use("Agg")
                    fig, axes = plt.subplots(3, 2, figsize=(12, 8))

                    axes[0, 0].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
                    axes[0, 0].matshow(raster.cpu().numpy()[i, 0])
                    axes[0, 0].title.set_text("raster")

                    axes[0, 1].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
                    axes[0, 1].matshow(river.cpu().numpy()[i, 0])
                    axes[0, 1].title.set_text("river_mask")

                    axes[1, 0].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
                    axes[1, 0].matshow(mask.cpu().numpy()[i, 0])
                    axes[1, 0].title.set_text("target")

                    axes[1, 1].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
                    axes[1, 1].matshow(torch.sigmoid(predicted_mask).cpu().numpy()[i, 0])
                    axes[1, 1].title.set_text("predicted")

                    axes[2, 0].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
                    axes[2, 0].matshow(correct_water_mask.cpu().numpy()[i, 0])
                    axes[2, 0].title.set_text("correct_water_mask")

                    axes[2, 1].tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
                    axes[2, 1].matshow(correct_nowater_mask.cpu().numpy()[i, 0])
                    axes[2, 1].title.set_text("correct_nowater_mask")

                    fig.tight_layout()
                    
                    self.logger.log_image(key=f"{step}/segmentation/{i}", images=[fig])

                    plt.close(fig)

            self.step_count[step] += 1
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

import hydra
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    from src.dataset import RiverSegmentationDataset
    ds = RiverSegmentationDataset(cfg)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=os.cpu_count())
    model = RiverModel(**cfg.model, lr=.001, log_images_every=1)

    for elem in dl:
        loss = model.training_step(elem)
        print(loss)

if __name__ == "__main__":
    main()