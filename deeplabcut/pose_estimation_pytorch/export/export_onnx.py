"""
@author:    Yu-Chanqging
@brief:     use to export dlc_pytorch to onnx 

tips: you need to complete the paths below
"""

import torch
import yaml
import sys
import os
from deeplabcut.pose_estimation_pytorch.models.model import PoseModel

# Create a wrapper to handle DLC's dictionary output
class ExportablePoseModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        out_dict = outputs["bodypart"]
        
        # Return only the tensors for ONNX
        return out_dict["heatmap"], out_dict["locref"]

# file paths
cfg_path = r"pytorch_config.yaml"  #your pytorch config path.
checkpoint_path = r"snapshot-best-030.pt" #your best pt path
onnx_path = r"E:dlc-vision\model_onnx.onnx" # a path you want to save onnx (file name)

# output directory
output_dir = os.path.dirname(onnx_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load config
print(f"Loading config: {cfg_path}")
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# Build model architecture from config
print("Building model from config...")
model = PoseModel.build(cfg['model'])
model.eval()

# Load trained weights
print(f"Loading weights: {checkpoint_path}")
snapshot = torch.load(checkpoint_path, map_location="cpu")

try:
    if 'model_state' in snapshot:
        model.load_state_dict(snapshot['model_state'])
    elif 'model' in snapshot:
        model.load_state_dict(snapshot['model'])
    else:
        model.load_state_dict(snapshot)
    print("Weights loaded successfully.")
except Exception as e:
    print(f"Error loading weights: {e}")
    sys.exit(1)
    
# Wrap the model for export
export_model = ExportablePoseModel(model)

# (448x448 confirmed from your config's crop_sampling)
dummy_input = torch.randn(1, 3, 448, 448) 
print(f"Creating dummy input with size: {dummy_input.shape}")

# Export to ONNX
print(f"Exporting ONNX model to: {onnx_path}")
try:
    torch.onnx.export(
        export_model,         
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["heatmap", "locref"], 
        dynamic_axes={            
            'input': {0: 'batch_size'},   
            'heatmap': {0: 'batch_size'},
            'locref': {0: 'batch_size'}
        }
    )
    
    print("-" * 30)
    print(f"Success! ONNX model saved to:")
    print(f"{onnx_path}")
    print("-" * 30)
    
except Exception as e:
    print(f"--- ONNX EXPORT FAILED ---")
    print(f"Error: {e}")
    print("\n(Hint: If error is 'Module onnx is not installed!', run 'pip install onnx onnxruntime')")