import argparse
import torch
import coremltools as ct
from model.edsr import EDSR   # ✅ adjust if your EDSR file lives elsewhere

def convert_model_to_coreml(model_path, scale):
    """
    Convert a trained PyTorch EDSR checkpoint to Core ML (.mlmodel).
    """

    print("🔹 Loading trained PyTorch model…")

    # 1. Build the same EDSR architecture you trained
    model = EDSR(scale=scale)

    # 2. Load trained weights (note the correct key)
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)   # fallback if already a plain state_dict
    model.load_state_dict(state_dict)
    model.eval()

    print("✅ Model loaded. Starting Core ML conversion…")

    # 3. Dummy input with dynamic height/width allowed
    # (batch=1, channels=3, HxW flexible)
    example_input = torch.rand(1, 3, 48, 48)

    coreml_model = ct.convert(
        model,
        inputs=[ct.ImageType(
            name="input",
            shape=(1, 3, ct.RangeDim(), ct.RangeDim()),  # dynamic H/W
            scale=1.0,          # model expects [0,1] if trained with ToTensor()
            bias=[0.0, 0.0, 0.0]
        )],
        convert_to="mlprogram",
        source="pytorch"
    )

    # 4. Save Core ML model
    out_name = f"SuperResolution_x{scale}.mlmodel"
    coreml_model.save(out_name)
    print(f"🎉 Conversion complete! Saved as {out_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert EDSR PyTorch model to Core ML")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument("--scale", type=int, default=4,
                        help="Upscale factor (e.g. 2, 3, or 4)")
    args = parser.parse_args()

    convert_model_to_coreml(args.checkpoint, args.scale)
