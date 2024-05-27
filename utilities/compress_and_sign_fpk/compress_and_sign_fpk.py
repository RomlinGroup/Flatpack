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


def compress_data(input_path, output_path):
    try:
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
        print(f"An error occurred while compressing: {e}")


def sign_data(output_path, signed_path, private_key_path, hash_size=256, passphrase=None):
    try:
        if hash_size not in [256, 384, 512]:
            raise ValueError("Invalid hash size. Supported sizes are 256, 384, and 512.")

        with open(private_key_path, 'rb') as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=passphrase.encode('utf-8') if passphrase else None,
                backend=default_backend()
            )

        if hash_size == 256:
            hash_algorithm = hashes.SHA256()
        elif hash_size == 384:
            hash_algorithm = hashes.SHA384()
        elif hash_size == 512:
            hash_algorithm = hashes.SHA512()
        else:
            raise ValueError("Unsupported hash size. Supported sizes are 256, 384, and 512.")

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
        print(f"An error occurred while signing: {e}")


def decompress_data(input_path, output_path):
    try:
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
        print(f"An error occurred while decompressing: {e}")
