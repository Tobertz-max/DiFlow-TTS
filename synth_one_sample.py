from omegaconf import OmegaConf
from diflow_tts import DiFlowTTS
import soundfile as sf
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize one sample")
    parser.add_argument("--config-path", required=True, help="path to configuration file.")
    parser.add_argument("--ckpt-path", required=True, help="path to checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--prompt-path", type=str, default=None, help="Path to reference audio for prompt")
    parser.add_argument("--prompt-duration", type=float, default=None, help="Duration of the prompt in seconds")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    parser.add_argument("--n-timesteps", type=int, default=128, help="Number of timesteps for synthesis")
    args = parser.parse_args()
    return args

def load_model(config, ckpt_path, device=None):
    if device is None:
        device = config.model.device
    else:
        config.model.device = device
    model = DiFlowTTS.load_from_checkpoint(ckpt_path, config=config.model, map_location=device)
    return model

def main():
    args = parse_args()
    config = OmegaConf.load(args.config_path)
    model = load_model(config, args.ckpt_path, device=args.device)
    out = model.synthesize(
        text=args.text,
        n_timesteps=args.n_timesteps,
        ref_audio_path=args.prompt_path,
        prompt_duration=args.prompt_duration
    )
    audio_out = out["wav"]
    sf.write("synthesized_output.wav", audio_out, samplerate=16000)

if __name__ == "__main__":
    main()