import os
import json
from huggingface_hub import snapshot_download

model_id = "uctnlp/mzansilm-125m"
local_model_path = "./mzansilm_local"

print(f"🔄 Downloading/Updating MzanziLM to {local_model_path}...")

# 1. Download the model files to a local folder
snapshot_download(repo_id=model_id, local_dir=local_model_path, local_dir_use_symlinks=False)

print("✅ Download complete!")

# 2. Read the config.json file
config_path = os.path.join(local_model_path, "config.json")
with open(config_path, 'r') as f:
    config = json.load(f)

# 3. Patch BOTH typos to match the actual weight files
print("🔧 Patching config.json to match actual model weights...")
config["hidden_size"] = 504          # The weights are actually 504, not 512
config["num_attention_heads"] = 8    # 512 / 8 = 64 (valid head dimension)
config["num_key_value_heads"] = 8    # Match the attention heads

# 4. Save the fixed config back to the file
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print("✅ MzanziLM is now fully patched and ready to use locally!")
print(f"📁 You can find it at: {os.path.abspath(local_model_path)}")