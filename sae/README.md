# ace-step-sae

Sparse Autoencoder (SAE) interpretability and steering for [ACE-Step](https://github.com/ivillar/ace-step-1.5) music generation models.

Implements the approach from ["Discovering and Steering Interpretable Concepts in Large Generative Music Models"](https://arxiv.org/abs/2505.18186) (Singh, Cherep & Maes, 2025), adapted for ACE-Step's Diffusion Transformer (DiT) architecture.

## What this does

1. **Discover** what musical concepts ACE-Step's DiT has learned internally, by training sparse autoencoders on its residual-stream activations
2. **Label** discovered features automatically using CLAP audio-text alignment
3. **Steer** generation by amplifying or suppressing specific features (e.g. "more piano", "less drums", "darker mood") without retraining

## Architecture

```
ACE-Step DiT Layer (hidden_size=2048)
          │
    residual stream  ──►  SAE Encoder  ──►  TopK(h, k=32)  ──►  SAE Decoder  ──►  reconstruction
          │                                       │
          │                                 sparse features
          │                                 (16,384 dims)
          │
    + steering vector  ◄──  α · W_dec[feature_idx]
          │
    steered residual stream
```

## Setup

```bash
git clone --recurse-submodules <this-repo>
cd ace-step-sae
pip install -e .

# For CLAP-based auto-labeling:
pip install -e ".[clap]"
```

## Pipeline

### 1. Collect activations

Run ACE-Step generations and capture DiT residual-stream activations:

```bash
python scripts/collect_activations.py \
    --model_dir ./ACE-Step \
    --output_dir ./activations \
    --layers 16 20 24 28 \
    --num_generations 200
```

### 2. Train SAE

Train a TopK sparse autoencoder on the collected activations:

```bash
python scripts/train_sae.py \
    --activations_dir ./activations \
    --layer 24 \
    --expansion_factor 8 \
    --k 32 \
    --num_steps 100000 \
    --output_dir ./sae_output/layer_24
```

### 3. Discover features

Analyze the trained SAE to find interpretable features:

```bash
python scripts/discover_features.py \
    --sae_path ./sae_output/layer_24/sae_final.pt \
    --activations_dir ./activations \
    --layer 24 \
    --output features_layer_24.json
```

### 4. Steer generation

Generate music while amplifying or suppressing discovered features:

```bash
python scripts/generate_with_steering.py \
    --model_dir ./ACE-Step \
    --sae_path ./sae_output/layer_24/sae_final.pt \
    --sae_layer 24 \
    --steer 42:10.0 100:-5.0 \
    --caption "A calm ambient track" \
    --output steered.wav --compare
```

## Python API

```python
from ace_step_sae import (
    SAEConfig, TopKSparseAutoencoder, ActivationCollector,
    SAETrainer, FeatureAnalyzer, SteeringHook,
)
from ace_step_sae.steering import SteeringSpec

# Train
cfg = SAEConfig(d_model=2048, expansion_factor=8, k=32)
sae = TopKSparseAutoencoder(cfg)
trainer = SAETrainer(cfg, activations_tensor)
sae = trainer.train()

# Analyze
analyzer = FeatureAnalyzer(sae)
records = analyzer.analyze(activations_tensor, top_k_examples=10)

# Steer
hook = SteeringHook(dit_model, sae, default_layer=24)
hook.add(SteeringSpec(feature_idx=42, coefficient=10.0))
with hook:
    result = model.generate_audio(...)
```

## Key design decisions

- **TopK over L1**: Uses hard TopK sparsity (Gao et al. 2024) instead of L1 penalty. This eliminates activation shrinkage and the need to tune an L1 coefficient.
- **Auxiliary dead-feature loss**: Dead latents are recycled by training them to reconstruct the residual error from alive features.
- **Diffusion-aware collection**: Activations are tagged with their diffusion timestep, enabling analysis of which features are active at early vs. late denoising steps.
- **Multi-layer steering**: Supports steering the same concept across multiple DiT layers for stronger effect.

## References

- Singh, Cherep & Maes (2025). "Discovering and Steering Interpretable Concepts in Large Generative Music Models." arXiv:2505.18186.
- Gao et al. (2024). "Scaling and evaluating sparse autoencoders." OpenAI.
- "Steering Diffusion Transformers with Sparse Autoencoders." OpenReview.
