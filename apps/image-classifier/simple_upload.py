#!/usr/bin/env python3
"""
Simple script to manually upload model files to Hugging Face
"""

import os
from huggingface_hub import HfApi

def upload_files():
    """Upload the prepared model files to Hugging Face Hub"""

    # Get token from environment
    HF_TOKEN = os.getenv('HF_TOKEN')
    if not HF_TOKEN:
        print("❌ Please set HF_TOKEN environment variable")
        print("Run: export HF_TOKEN=your_huggingface_token_here")
        return

    # Repository details
    repo_id = "TrashHobbit/dakota-ai-cifar10-classifier"

    api = HfApi()
    api.set_access_token(HF_TOKEN)

    # Check if repository exists
    try:
        api.repo_info(repo_id)
        print(f"✅ Repository found: {repo_id}")
    except Exception as e:
        print(f"❌ Repository not found: {e}")
        print("The repository may not be fully created yet.")
        print("Please check: https://huggingface.co/TrashHobbit/dakota-ai-cifar10-classifier")
        return

    # Files to upload
    files_to_upload = [
        "README.md",  # Model card
        "best_cifar10_model.keras",  # Original Keras model
        "saved_model/saved_model.pb",  # SavedModel format
    ]

    print("🚀 Uploading files to Hugging Face Hub...")
    print(f"Repository: {repo_id}")

    for filepath in files_to_upload:
        if os.path.exists(filepath):
            print(f"📤 Uploading: {filepath}")
            try:
                api.upload_file(
                    path_or_fileobj=filepath,
                    path_in_repo=filepath,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"✅ Uploaded: {filepath}")
            except Exception as e:
                print(f"❌ Failed to upload {filepath}: {e}")
        else:
            print(f"⚠️ File not found: {filepath}")
            print("Make sure the file exists and try again.")

    # Upload SavedModel variables directory
    variables_dir = "saved_model/variables"
    if os.path.exists(variables_dir):
        print(f"📤 Uploading variables directory: {variables_dir}")
        try:
            api.upload_folder(
                folder_path=variables_dir,
                path_in_repo="saved_model/variables",
                repo_id=repo_id,
                repo_type="model"
            )
            print("✅ Uploaded variables directory")
        except Exception as e:
            print(f"❌ Failed to upload variables: {e}")

    print(f"\n🎉 Upload complete!")
    print(f"📦 Model available at: https://huggingface.co/{repo_id}")
    print("🚀 Ready to use your custom CIFAR-10 model in the classifier!")

if __name__ == "__main__":
    print("🐍 Simple Hugging Face Model Upload Script")
    print("=" * 50)
    upload_files()
