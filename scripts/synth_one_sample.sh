python synth_one_sample.py \
    --config-path configs/model.yaml \
    --ckpt-path /path/to/diflow-tts.ckpt \
    --text "<put your text here>" \
    --prompt-path /path/to/prompt.wav \
    --prompt-duration 3 \
    --n-timesteps 128 \
    --device cuda