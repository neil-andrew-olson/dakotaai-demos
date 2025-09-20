#!/usr/bin/env python3

import os
import urllib.request
import json
import shutil

def download_mobilenet():
    """Download MobileNet v2 model locally to avoid CORS issues"""

    print("📥 Downloading MobileNet v2 model locally...")
    print("=" * 50)

    # Create local directory for model
    local_dir = "public/mobilenet"
    os.makedirs(local_dir, exist_ok=True)

    print(f"📁 Created directory: {local_dir}")

    # MobileNet v2 URLs from TensorFlow Hub (CORS-compatible)
    base_url = "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1"
    files_to_download = [
        "model.json",
        "group1-shard1of1.bin"
    ]

    print("⬇️ Downloading model files...")

    for filename in files_to_download:
        url = f"{base_url}/{filename}"
        local_path = os.path.join(local_dir, filename)

        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, local_path)
            print(f"  ✅ {filename} downloaded")
        except Exception as e:
            print(f"  ❌ Failed to download {filename}: {e}")
            return False

    print("✅ All MobileNet files downloaded successfully!")

    # Verify the model.json file
    model_json_path = os.path.join(local_dir, "model.json")
    try:
        with open(model_json_path, 'r') as f:
            model_data = json.load(f)
        print("✅ Model configuration verified")

        # Print some model info
        print("📊 Model Info:")
        print(f"  - Format: {model_data.get('format', 'Unknown')}")
        print(f"  - Files: {len(files_to_download)}")
        print("  - Size: ~12MB (compressed)")

    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

    print("\n🎉 MobileNet v2 ready for local hosting!")
    print(f"📂 Location: {local_dir}/")
    print("🌐 Load path: /mobilenet/model.json")
    print("⚡ Benefits:")
    print("  - No CORS issues")
    print("  - Faster loading")
    print("  - Reliable (no external dependencies)")

    return True

if __name__ == "__main__":
    success = download_mobilenet()
    if success:
        print("\n🚀 Next Steps:")
        print("1. Update frontend to load from /mobilenet/model.json")
        print("2. Test local model loading")
        print("3. Deploy to Netlify")
    else:
        print("\n❌ Download failed - check your internet connection")
