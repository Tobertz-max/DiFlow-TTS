from lightning import LightningModule
from abc import ABC
import torch
from data.dataset_libritts import TTSDataModule
from transformers import get_cosine_schedule_with_warmup
import wandb
import librosa
 
class BaseLightningClass(LightningModule, ABC):
    def setup_optimizer(self, config):
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=config.num_training_steps
        )
 
    def setup_dataset(self, config):
        self.dataset = TTSDataModule(config)

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate"
            }
        }
    
    def get_dataloader(self):
        self.dataset.setup()
        return self.dataset.train_dataloader(), self.dataset.val_dataloader()
    
    def get_losses(self, batch):
        audio_codes = batch["audio_codes"]
        prompt_codes = batch["prompt_codes"]
        texts = batch["texts"]
        spk_embs = batch["spk_embs"]
        durs = batch["durs"]
        text_lens = batch["text_lens"]
        code_lens = batch["code_lens"]
        
        return self(audio_codes=audio_codes,
                    prompt_codes=prompt_codes,
                    texts=texts,
                    spk_embs=spk_embs,
                    durs=durs,
                    text_lens=text_lens,
                    code_lens=code_lens)

    def training_step(self, batch, batch_idx):
        loss_dict = self.get_losses(batch)
 
        self.log(
            "Step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
 
        self.log(
            "Sub_Loss/Train_FDFD_Loss",
            loss_dict["FDFD_Loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        self.log(
            "Sub_Loss/Train_Content_Loss",
            loss_dict["Content_Loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        self.log(
            "Sub_Loss/Train_Duration_Loss",
            loss_dict["Duration_Loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        total_loss = loss_dict["FDFD_Loss"] + loss_dict["Content_Loss"] + 0.5 * loss_dict["Duration_Loss"]

        self.log(
            "Loss/Train",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
 
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        loss_dict = self.get_losses(batch)
 
        self.log(
            "Step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
 
        self.log(
            "Sub_Loss/Val_DFDF_Loss",
            loss_dict["FDFD_Loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        self.log(
            "Sub_Loss/Val_Content_Loss",
            loss_dict["Content_Loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        self.log(
            "Sub_Loss/Val_Duration_Loss",
            loss_dict["Duration_Loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )

        total_loss = loss_dict["FDFD_Loss"] + loss_dict["Content_Loss"] + 0.5 * loss_dict["Duration_Loss"]

        self.log(
            "Loss/Val",
            total_loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
 
        return total_loss
    
    def on_validation_end(self):
        if self.global_rank == 0 and self.logger.__class__.__name__ == "WandbLogger":
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                audio_path = one_batch["audio_path"][0]
                gt_audio = librosa.load(audio_path, sr=16000)[0]
                wandb.log({
                    "Synthesize/Val_GT_Speech": wandb.Audio(gt_audio, sample_rate=16000)
                }, step=self.global_step)

            text = one_batch["texts"][0].unsqueeze(0).to(self.device)
            audio_path = one_batch["audio_path"][0]
            ref_audio_path = one_batch["ref_audio_path"][0]

            audio_out = self.synthesize(text=text,
                                        n_timesteps=128,
                                        ref_audio_path=ref_audio_path)["wav"]

            wandb.log({
                "Synthesize/Val_Synth": wandb.Audio(audio_out, sample_rate=16000)
            }, step=self.global_step)