### Environment Setup

Download the lm_eval library from https://github.com/EleutherAI/lm-evaluation-harness

```bash
conda create -yn streaming python=3.8
conda activate streaming

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```
