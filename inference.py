import os
from omegaconf import OmegaConf
from diflow_tts import DiFlowTTS
import soundfile as sf
import torch.multiprocessing as mp
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for TTS model")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--prompt-dir", type=str, required=True, help="Path to the root of prompt audio files.")
    parser.add_argument("--metadata-path", type=str, required=True, help="Path to the metadata file.")
    parser.add_argument("--run-name", type=str, default="default_run", help="Name of the run.")
    parser.add_argument("--output-path", type=str, required=True, help="Directory to save the output audio files.")
    parser.add_argument("--n-timesteps", type=int, default=16, help="Number of timesteps for synthesis.")
    parser.add_argument("--prompt-duration", type=int, default=None, help="Duration of the prompt audio in seconds.")
    parser.add_argument("--num-processes", type=int, default=1, help="Number of processes to use for inference.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume inference from a previous run.")
    return parser.parse_args()
 
def load_model(config_path, ckpt_path, device=None):
    config = OmegaConf.load(config_path)
    if device is None:
        device = config.model.device
    else:
        config.model.device = device
    model = DiFlowTTS.load_from_checkpoint(ckpt_path, config=config.model, map_location=device)
    return model

def infer_one_sample(sample, model, n_timesteps=16, prompt_duration=None):
    target_path = sample["target_path"]
    prompt_path = sample["prompt_path"]
    text = sample["text"]

    out = model.synthesize(
        text=text,
        n_timesteps=n_timesteps,
        ref_audio_path=prompt_path,
        prompt_duration=prompt_duration
    )

    audio_out = out["wav"]
    sf.write(target_path, audio_out, samplerate=16000)

def run_worker(samples, args, lock, position):
    description = f"Process #{position}"
    with lock:
        progress = tqdm(total=len(samples), desc=description, position=position)

    model = load_model(args.config_path, args.ckpt_path, device=args.device)
    for sample in samples:
        infer_one_sample(
            sample, 
            model, 
            n_timesteps=args.n_timesteps, 
            prompt_duration=args.prompt_duration, 
        )

        with lock:
            progress.update(1)
            progress.set_description(f"{description} - Processed {progress.n}/{progress.total} samples")
    
    with lock:
        progress.close()
 
def main():
    args = parse_args()

    run_name = args.run_name
    name_save = "{}-steps-{}-{}s".format(run_name, args.n_timesteps, args.prompt_duration)
    save_path = os.path.join(args.output_path, name_save)
    os.makedirs(save_path, exist_ok=True)

    with open(args.metadata_path, "r") as fin:
        samples = []
        for line in fin:
            target_name, prompt_name, target_transcript, _, _, _  = line.rstrip().split('|')
            target_path = os.path.join(save_path, target_name)

            if args.resume and os.path.exists(target_path):
                continue

            prompt_path = os.path.join(args.prompt_dir, prompt_name)

            samples.append({
                "target_path": target_path,
                "prompt_path": prompt_path,
                "text": target_transcript,
            })

    chunk_size = len(samples) // args.num_processes
    lock = mp.Lock()
    processes = []

    for i in range(args.num_processes):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < args.num_processes - 1 else len(samples)
        process_samples = samples[start_idx:end_idx]
        
        p = mp.Process(target=run_worker, args=(process_samples, args, lock, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"Inference completed. Outputs saved to {save_path}")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()