import argparse
import logging
import os
import compress_and_sign_fpk as compress_and_sign


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_file_path(file_path, purpose="input"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The {purpose} file/folder path does not exist.")


def main():
    parser = argparse.ArgumentParser(description='Test the compress and sign functionalities.')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the input file or folder to be compressed and signed.')
    parser.add_argument('-o', '--output', type=str, required=True, help='Path where the compressed file will be saved.')
    parser.add_argument('-s', '--signed', type=str, required=True, help='Path where the signed file will be saved.')
    parser.add_argument('-p', '--private_key', type=str, required=True, help='Path to the private key for signing.')
    parser.add_argument('--hash_size', type=int, default=256, help='Hash size for signing.')
    parser.add_argument('--passphrase', type=str, default=None, help='Passphrase for the private key.')
    args = parser.parse_args()

    setup_logging()
    logging.info("Starting the test...")

    # Validate the paths
    validate_file_path(args.input)
    validate_file_path(args.private_key, purpose="private key")

    # Compress the input file or folder
    compress_and_sign.compress_data(args.input, args.output)
    logging.info(f"Compression complete. Compressed file saved at {args.output}")

    # Sign the compressed file
    compress_and_sign.sign_data(args.output, args.signed, args.private_key, hash_size=args.hash_size,
                                passphrase=args.passphrase)
    logging.info(f"Signing complete. Signed file saved at {args.signed}")

    # Decompress the signed file
    decompressed_output_path = "decompressed_output"
    compress_and_sign.decompress_data(args.signed, decompressed_output_path)
    logging.info(f"Decompression complete. Decompressed file saved at {decompressed_output_path}")


if __name__ == "__main__":
    main()
