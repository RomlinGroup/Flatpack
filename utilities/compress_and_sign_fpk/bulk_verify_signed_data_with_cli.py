import argparse
import os

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


def validate_file_path(path, is_input=True, allowed_dir=None):
    """
    Validate the file path to prevent directory traversal attacks.

    Parameters:
        path (str): The path to validate.
        is_input (bool): Flag indicating if the path is for input. Defaults to True.
        allowed_dir (str): The allowed directory for the path. Defaults to None.

    Returns:
        str: The absolute path if valid.

    Raises:
        ValueError: If the path is outside the allowed directory or invalid.
        FileNotFoundError: If the path does not exist.
    """
    absolute_path = os.path.abspath(path)

    if allowed_dir:
        allowed_dir_absolute = os.path.abspath(allowed_dir)
        if not absolute_path.startswith(allowed_dir_absolute):
            raise ValueError(
                f"Path '{path}' is outside the allowed directory '{allowed_dir}'."
            )

    if is_input:
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"The path '{absolute_path}' does not exist.")
        if not (os.path.isfile(absolute_path) or os.path.isdir(absolute_path)):
            raise ValueError(
                f"The path '{absolute_path}' is neither a file nor a directory."
            )
    else:
        output_dir = os.path.dirname(absolute_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    return absolute_path


def verify_signed_data(signed_file_path, public_pem_path):
    """
    Verify the digital signature of a signed file.

    Parameters:
        signed_file_path (str): Path to the file that contains both the original data and the signature.
        public_pem_path (str): Path to the public key in PEM format.

    Returns:
        bool: True if the signature is valid, False otherwise.
    """
    try:
        abs_signed_file_path = validate_file_path(signed_file_path)
        abs_public_pem_path = validate_file_path(public_pem_path)

        with open(abs_public_pem_path, 'rb') as f:
            public_pem = f.read()

        with open(abs_signed_file_path, 'rb') as f:
            signed_data = f.read()

        separator = b"---SIGNATURE_SEPARATOR---"
        original_data, signature = signed_data.split(separator)

        public_key = serialization.load_pem_public_key(public_pem, backend=default_backend())

        public_key.verify(
            signature,
            original_data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except (InvalidSignature, ValueError):
        return False


def verify_bulk_signed_data(validate_directory_path, public_pem_path):
    """
    Verify the digital signatures of .fpk files in each subdirectory of the given directory.

    Parameters:
        directory_path (str): Path to the directory containing subdirectories with signed .fpk files.
        public_pem_path (str): Path to the public key in PEM format.
    """
    valid_signatures = 0
    invalid_signatures = 0

    abs_directory_path = validate_file_path(validate_directory_path, allowed_dir=validate_directory_path)

    for subdir in os.listdir(abs_directory_path):
        subdir_path = os.path.join(abs_directory_path, subdir)
        if os.path.isdir(subdir_path):
            fpk_files = [f for f in os.listdir(subdir_path) if f.endswith('.fpk')]
            if fpk_files:
                fpk_file_path = os.path.join(subdir_path, fpk_files[0])
                result = verify_signed_data(fpk_file_path, public_pem_path)
                if result:
                    valid_signatures += 1
                    print(f"The signature is valid for file: {fpk_files[0]} in {subdir}")
                else:
                    invalid_signatures += 1
                    print(f"The signature is NOT valid for file: {fpk_files[0]} in {subdir}")

    print(f"\nTotal valid signatures: {valid_signatures}")
    print(f"Total invalid signatures: {invalid_signatures}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify the digital signatures of .fpk files in the /warehouse directory.")
    parser.add_argument("-p", "--public_key", required=True, help="Path to the public key in PEM format.")

    args = parser.parse_args()

    directory_path = os.path.join(os.getcwd(), '..', '..', 'warehouse')
    verify_bulk_signed_data(directory_path, args.public_key)
