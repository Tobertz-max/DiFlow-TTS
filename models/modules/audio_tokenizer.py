import torch.nn as nn
import torch
import librosa
from models.facodec import FACodecEncoder, FACodecDecoder

class FACodec(nn.Module):
    def __init__(self, config, device="cpu"):
        super().__init__()
        config.encoder.device = device
        config.decoder.device = device

        self.device = device
        self.fa_encoder = FACodecEncoder.from_pretrained(cfg=config.encoder).eval().to(self.device)
        self.fa_decoder = FACodecDecoder.from_pretrained(cfg=config.decoder).eval().to(self.device)

    def load_audio(self, audio_path):
        wav = librosa.load(audio_path, sr=16000)[0]
        wav = torch.from_numpy(wav).float()
        wav = wav.unsqueeze(0).unsqueeze(0).to(self.device)
        return wav

    @torch.no_grad()
    def tokenize(self, audio_path):
        wav = self.load_audio(audio_path) # B, 1, L
        enc_out = self.fa_encoder(wav)
        vq_post_emb, vq_id, _, quantized, spk_embs = self.fa_decoder(enc_out, eval_vq=False, vq=True)
        return vq_id, spk_embs
    
    @torch.no_grad()
    def detokenize(self, codes, timbre):
        # codes: [num_quantizer, B, L]
        embs = self.fa_decoder.vq2emb(codes)
        audio_out = self.fa_decoder.inference(embs, timbre)
        audio_out = audio_out[0][0].cpu().numpy()
        return audio_out