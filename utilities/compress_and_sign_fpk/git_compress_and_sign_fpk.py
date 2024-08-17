import argparse
import compress_and_sign_fpk as compress_and_sign
import logging
import os


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


def validate_file_path(file_path, purpose="input"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The {purpose} file/folder path does not exist.")


def find_folders_to_compress(base_dir):
    folders = []

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            folders.append(full_path)
    return folders


def delete_existing_fpk_files(folder_path, logger):
    for file in os.listdir(folder_path):
        if file.endswith(".fpk"):
            os.remove(os.path.join(folder_path, file))
            logger.info("Deleted existing .fpk file: %s.", file)


def main():
    parser = argparse.ArgumentParser(
        description='Automatically compress, sign, and then only keep the signed folders found directly within /warehouse, excluding the "template" directory.')

    parser.add_argument(
        '-p', '--private_key',
        type=str,
        required=True,
        help='Path to the private key for signing.'
    )

    parser.add_argument(
        '--hash_size',
        type=int,
        default=256,
        help='Hash size for signing.'
    )

    parser.add_argument(
        '--passphrase',
        type=str,
        default=None,
        help='Passphrase for the private key.'
    )

    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting the compression and signing process...")

    validate_file_path(args.private_key, purpose="private key")

    base_dir = os.path.join(os.getcwd(), 'warehouse')
    validate_file_path(base_dir, purpose="base directory")

    folders = find_folders_to_compress(base_dir)

    for folder in folders:
        delete_existing_fpk_files(folder, logger)

        folder_name = os.path.basename(folder)
        output_path = os.path.join(folder, f"{folder_name}-temp.fpk")
        signed_path = os.path.join(folder, f"{folder_name}.fpk")

        compress_and_sign.compress_data(folder, output_path)
        logger.info("Compressed file saved at %s.", output_path)

        compress_and_sign.sign_data(output_path, signed_path, args.private_key, hash_size=args.hash_size,
                                    passphrase=args.passphrase)
        logger.info("Signed file saved at %s.", signed_path)

        os.remove(output_path)
        logger.info("Uncompressed file %s deleted.", output_path)


if __name__ == "__main__":
    main()
