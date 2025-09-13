set -euo pipefail

apt update && apt install ffmpeg -y 

cd /workspace/VibeVoice

uv venv
source .venv/bin/activate


uv pip install flash-attn --no-build-isolation
uv pip install -e .


uv pip install --upgrade "huggingface_hub[cli]"


hf download aoi-ot/VibeVoice-Large --local-dir aoi-ot/VibeVoice-Large
hf download aoi-ot/VibeVoice-1.5B --local-dir aoi-ot/VibeVoice-1.5B

python demo/inference_from_file.py --model_path aoi-ot/VibeVoice-Large --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice
