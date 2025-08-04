import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
import time
import torch
from models.flow_matching import FDFD
from models.pcm import PCM
from models.modules.audio_tokenizer import FACodec
from baselightningmodule import BaseLightningClass
from text.tokenizer import Tokenizer

class DiFlowTTS(BaseLightningClass):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_quantizers = config.codec_model.n_quantizers
        self.vocab_size = config.codec_model.vocab_size

        config.pcm.n_quantizers = 2 # content codes
        config.pcm.vocab_size = self.vocab_size
        self.PCM = PCM(config.pcm)

        config.flow_matching.n_quantizers = self.n_quantizers - 2 # exclude content codes
        config.flow_matching.vocab_size = self.vocab_size
        self.FDFD = FDFD(config.flow_matching)

        self.codec_model = FACodec(config.codec_model, config.device).eval()
        self.tokenizer = Tokenizer()

    def forward(
        self, 
        audio_codes,
        prompt_codes, 
        texts,
        spk_embs,
        text_lens,
        code_lens,
        durs
    ):
        (
            logits,
            content_emb, # (B, n_quantizes, L, D)
            log_duration_predictions,
            duration_rounded,
            src_masks,
            tgt_masks
        ) = self.PCM(
            texts=texts,
            src_lens=text_lens,
            max_src_len=texts.shape[-1],
            tgt_lens=code_lens,
            max_tgt_len=audio_codes.shape[-1],
            d_targets=durs,
        )

        content_codes = audio_codes[:, 1:3, :]
        codes_gen_loss_dict = self.PCM.compute_loss(
            codes_pred=logits,
            codes=content_codes,
            log_durations_pred=log_duration_predictions,
            durations=durs,
            src_masks=src_masks,
        )
        
        prosody_codes, acoustic_codes = audio_codes[:, :1, :], audio_codes[:, 3:, :]
        x1 = torch.cat([prosody_codes, acoustic_codes], dim=1)

        fm_loss_dict = self.FDFD.compute_loss(x1=x1,
                                              prompts=prompt_codes,
                                              content_emb=content_emb,
                                              spk_embs=spk_embs,
                                              audio_masks=tgt_masks)
        
        loss_dict = codes_gen_loss_dict | fm_loss_dict
        return loss_dict

    @torch.inference_mode()
    def synthesize(
        self, 
        text, 
        n_timesteps, 
        ref_audio_path,
        prompt_duration=None
    ):
        start_time = time.time() # for RTF computation
        prompt_codes, spk_emb = self.codec_model.tokenize(ref_audio_path)

        if prompt_duration is not None:
            prompt_length = int(80 * prompt_duration)
            if prompt_length < prompt_codes.shape[-1]:
                prompt_codes = prompt_codes[:, :prompt_length]
        prompt_codes = prompt_codes.transpose(0, 1)

        if isinstance(text, str):
            text = self.tokenizer.tokenize(text).to(self.config.device)
        else:
            text = text.to(self.config.device)
        
        text = text[text != 0].unsqueeze(0) # [1, L_text]
        content_logits, content_emb = self.PCM(texts=text, 
                                               src_lens=torch.tensor([text.shape[-1]]).to(text.device), 
                                               max_src_len=text.shape[-1])[:2] # (B, 2, L, D)

        pred_codes = self.FDFD(content_emb=content_emb,
                               prompt_codes=prompt_codes,
                               spk_emb=spk_emb,
                               n_timesteps=n_timesteps)
        
        posody_codes, acoustic_codes = torch.split(pred_codes, [1, 3], dim=0)
        content_codes = content_logits.softmax(1).argmax(1).permute(1, 0, 2)

        pred_codes = torch.cat([posody_codes, content_codes, acoustic_codes], dim=0)  # (n_quantizers, B, T)
        audio_out = self.codec_model.detokenize(pred_codes, spk_emb)

        end_time = time.time()

        time_used = (end_time - start_time)
        rtf = (time_used * 16000) / audio_out.shape[-1]

        return {
            "wav": audio_out,
            "rtf": rtf
        }