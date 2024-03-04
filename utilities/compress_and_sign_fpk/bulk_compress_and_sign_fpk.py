import argparse
import logging
import os
import compress_and_sign_fpk as compress_and_sign


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_file_path(file_path, purpose="input"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The {purpose} file/folder path does not exist.")


def find_folders_to_compress(base_dir):
    folders = []
    # Starting from base_dir, find all folders directly within it, excluding the 'template' folder
    for entry in os.listdir(base_dir):
        if entry == "template":  # Skip the 'template' directory
            continue
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            folders.append(full_path)
    return folders


def main():
    parser = argparse.ArgumentParser(
        description='Automatically compress, sign, and then only keep the signed folders found directly within /warehouse, excluding the "template" directory, which is two levels up from the script location.')
    parser.add_argument('-p', '--private_key', type=str, required=True, help='Path to the private key for signing.')
    parser.add_argument('--hash_size', type=int, default=256, help='Hash size for signing.')
    parser.add_argument('--passphrase', type=str, default=None, help='Passphrase for the private key.')
    args = parser.parse_args()

    setup_logging()
    logging.info("Starting the compression and signing process...")

    # Validate the private key path
    validate_file_path(args.private_key, purpose="private key")

    # Assuming the script is run from a directory two levels down from /warehouse
    base_dir = os.path.join(os.getcwd(), '..', '..', 'warehouse')
    folders = find_folders_to_compress(base_dir)

    for folder in folders:
        folder_name = os.path.basename(folder)
        output_path = os.path.join(folder, f"{folder_name}-temp.fpk")
        signed_path = os.path.join(folder, f"{folder_name}.fpk")

        # Compress the folder
        compress_and_sign.compress_data(folder, output_path)
        logging.info(f"Compression complete for {folder}. Compressed file saved at {output_path}")

        # Sign the compressed file
        compress_and_sign.sign_data(output_path, signed_path, args.private_key, hash_size=args.hash_size,
                                    passphrase=args.passphrase)
        logging.info(f"Signing complete for {output_path}. Signed file saved at {signed_path}")

        # Delete the uncompressed .fpk file, leaving only the signed version
        os.remove(output_path)
        logging.info(f"Uncompressed file {output_path} deleted, only the signed version is kept.")


if __name__ == "__main__":
    main()
