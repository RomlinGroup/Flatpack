import argparse
from huggingface_hub import HfApi


def main(args):
    api = HfApi()
    api.upload_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        ignore_patterns=args.ignore_patterns
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace Hub")

    parser.add_argument("--folder_path", type=str, required=True,
                        help="Path to the folder containing the model files.")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="Repository ID for the HuggingFace Hub.")
    parser.add_argument("--repo_type", type=str, default="model",
                        help="Type of repository. Default is 'model'.")
    parser.add_argument("--ignore_patterns", type=str, default="**/*.pth",
                        help="Pattern for files/folders to ignore. Default is '**/*.pth'.")

    args = parser.parse_args()
    main(args)
