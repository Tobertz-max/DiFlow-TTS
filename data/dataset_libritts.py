import json
import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from text import text_to_sequence
from data.utils import pad_1D, pad_2D

class TTSDataset(Dataset):
    def __init__(
        self, 
        metadata_path, 
        codes_path,
        cleaners
    ):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.codes_path = codes_path
        self.cleaners = cleaners

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        item = self.metadata[index]
        start_code, end_code = round(item["start"] * 80), round(item["end"] * 80)
    
        audio_path = item["audio_path"]
        ref_audio_path = item["audio_path"]
        durs = torch.tensor(item["audio_text_align"])

        filename = os.path.basename(item["audio_path"])

        texts = torch.tensor(text_to_sequence(item["text"], self.cleaners)).long()

        code_path = os.path.join(self.codes_path, filename.replace(".wav", ".json"))
        with open(code_path, "r") as fin:
            data = json.load(fin)
        
        audio_codes = torch.tensor(data["quantizers"])[:, start_code:end_code].transpose(1, 0)
        spk_embs = torch.tensor(data["spkemb"])

        return {
            "texts": texts,
            "audio_codes": audio_codes,
            "spk_embs": spk_embs,
            "durs": durs,
            "audio_path": audio_path,
            "ref_audio_path": ref_audio_path
        }
    
class DataCollator:
    def __init__(self, vocab_size, prompt_segment_ratio):
        self.vocab_size = vocab_size
        self.pad_token_id = self.vocab_size
        self.sep_token_id = self.vocab_size + 1
        self.prompt_segment_ratio = prompt_segment_ratio

    def get_prompt_codes(self, codes):
        min_length = min([x.shape[0] for x in codes])
        prompt_length = int(self.prompt_segment_ratio * min_length)

        prompts = []
        for x in codes:
            start_idx = random.randint(0, min_length - prompt_length)
            end_idx = start_idx + prompt_length
            prompt_codes = x[start_idx:end_idx]
            sep_token = torch.full((1, x.shape[-1]), self.sep_token_id, dtype=x.dtype, device=x.device)
            prompt_codes = torch.cat([prompt_codes, sep_token], dim=0)
            prompts.append(prompt_codes)
        prompts = torch.stack(prompts).transpose(1, 2)
        return prompts

    def __call__(self, batch):
        texts, audio_codes, spk_embs, durs, audio_path, ref_audio_path = [], [], [], [], [], []
        for item in batch:
            texts.append(item["texts"])
            audio_codes.append(item["audio_codes"])
            spk_embs.append(item["spk_embs"])
            durs.append(item["durs"])
            audio_path.append(item["audio_path"])
            ref_audio_path.append(item["ref_audio_path"])

        text_lens = torch.tensor([len(text) for text in texts]).long()
        code_lens = torch.tensor([x.shape[0] for x in audio_codes]).long()

        prompt_codes = self.get_prompt_codes(audio_codes)

        padded_texts = pad_1D(texts, PAD=0)
        padded_codes = pad_2D(audio_codes, maxlen=code_lens.max().item(), PAD=self.pad_token_id).transpose(1, 2)
        padded_durs = pad_1D(durs, PAD=0)
        spk_embs = torch.stack(spk_embs)

        return {
            "texts": padded_texts,
            "audio_codes": padded_codes,
            "prompt_codes": prompt_codes,
            "spk_embs": spk_embs,
            "durs": padded_durs,
            "audio_path": audio_path,
            "ref_audio_path": ref_audio_path,
            "text_lens": text_lens,
            "code_lens": code_lens
        }

class TTSDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_metadata_path = config.train.metadata_path
        self.val_metadata_path = config.val.metadata_path

        self.train_codes_path = config.train.codes_path
        self.val_codes_path = config.val.codes_path

        self.vocab_size = config.vocab_size
        self.prompt_segment_ratio = config.prompt_segment_ratio
        self.cleaners = config.text_cleaners
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

    def setup(self, stage: str = None):
        self.train_dataset = TTSDataset(self.train_metadata_path,
                                        self.train_codes_path,
                                        self.cleaners)

        self.val_dataset = TTSDataset(self.val_metadata_path,
                                      self.val_codes_path,
                                      self.cleaners)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          collate_fn=DataCollator(self.vocab_size, self.prompt_segment_ratio))
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          collate_fn=DataCollator(self.vocab_size, self.prompt_segment_ratio))