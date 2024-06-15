import logging
import os
import tarfile
import tempfile
import zstandard as zstd

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

logging.basicConfig(filename='debug.log', filemode='w', level=logging.INFO)


def validate_file_path(path, is_input=True, allowed_dir=None):
    # Resolve absolute path to avoid false positives on legitimate paths
    absolute_path = os.path.abspath(path)

    if allowed_dir:
        allowed_dir_absolute = os.path.abspath(allowed_dir)
        if not absolute_path.startswith(allowed_dir_absolute):
            raise ValueError(
                f"Path '{path}' is outside the allowed directory '{allowed_dir}'."
            )

    if is_input:
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"The path '{path}' does not exist.")
        if not (os.path.isfile(absolute_path) or os.path.isdir(absolute_path)):
            raise ValueError(f"The path '{path}' is neither a file nor a directory.")
    else:
        output_dir = os.path.dirname(absolute_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


def compress_data(input_path, output_path, allowed_dir=None):
    try:
        validate_file_path(input_path, allowed_dir=allowed_dir)
        validate_file_path(output_path, is_input=False, allowed_dir=allowed_dir)

        if os.path.isfile(input_path):
            with open(input_path, 'rb') as f:
                data = f.read()
            compression_level = 22
            compressed_data = zstd.compress(data, compression_level)
            with open(output_path, 'wb') as f:
                f.write(compressed_data)
        elif os.path.isdir(input_path):
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_tar_file_path = os.path.join(temp_dir, "temp.tar")
                with tarfile.open(temp_tar_file_path, 'w') as tar:
                    tar.add(input_path, arcname=os.path.basename(input_path))
                with open(temp_tar_file_path, 'rb') as f:
                    data = f.read()
                compressed_data = zstd.compress(data)
                with open(output_path, 'wb') as f:
                    f.write(compressed_data)
        else:
            print("The specified input path is neither a file nor a directory.")
    except Exception as e:
        logging.error(f"An error occurred while compressing: {e}")
        print(f"An error occurred while compressing: {e}")


def sign_data(
        output_path,
        signed_path,
        private_key_path,
        hash_size=256,
        passphrase=None,
        allowed_dir=None
):
    try:
        validate_file_path(output_path, allowed_dir=allowed_dir)
        validate_file_path(signed_path, is_input=False, allowed_dir=allowed_dir)
        validate_file_path(private_key_path, allowed_dir=allowed_dir)

        if hash_size not in [256, 384, 512]:
            raise ValueError("Invalid hash size. Supported sizes are 256, 384, and 512.")

        with open(private_key_path, 'rb') as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=passphrase.encode('utf-8') if passphrase else None,
                backend=default_backend()
            )

        hash_algorithm = {
            256: hashes.SHA256(),
            384: hashes.SHA384(),
            512: hashes.SHA512()
        }[hash_size]

        with open(output_path, 'rb') as f:
            data = f.read()

        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hash_algorithm),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hash_algorithm
        )

        separator = b"---SIGNATURE_SEPARATOR---"
        combined_data = data + separator + signature

        with open(signed_path, 'wb') as f:
            f.write(combined_data)
    except Exception as e:
        logging.error(f"An error occurred while signing: {e}")
        print(f"An error occurred while signing: {e}")


def decompress_data(input_path, output_path, allowed_dir=None):
    try:
        validate_file_path(input_path, allowed_dir=allowed_dir)
        validate_file_path(output_path, is_input=False, allowed_dir=allowed_dir)

        with open(input_path, 'rb') as f:
            compressed_data = f.read()

        decompressed_data = zstd.decompress(compressed_data)

        try:
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(decompressed_data)
                tmp_file.seek(0)

                with tarfile.open(fileobj=tmp_file, mode='r:') as tar:
                    tar.extractall(path=output_path)
        except tarfile.ReadError:
            with open(output_path, 'wb') as f:
                f.write(decompressed_data)

    except Exception as e:
        logging.error(f"An error occurred while decompressing: {e}")
        print(f"An error occurred while decompressing: {e}")
