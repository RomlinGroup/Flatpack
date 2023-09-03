import os
import tarfile
import tempfile
import zstandard as zstd
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import logging

# Setup logging
logging.basicConfig(filename='debug.log', filemode='w', level=logging.INFO)


# Function to compress data
def compress_data(input_path, output_path):
    try:
        if os.path.isfile(input_path):
            # Handling individual files
            with open(input_path, 'rb') as f:
                data = f.read()
            compressed_data = zstd.compress(data)
            with open(output_path, 'wb') as f:
                f.write(compressed_data)
        elif os.path.isdir(input_path):
            # Create a secure temporary directory for temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_tar_file_path = os.path.join(temp_dir, "temp.tar")
                with tarfile.open(temp_tar_file_path, 'w') as tar:
                    tar.add(input_path, arcname=os.path.basename(input_path))
                # Compress the tar file with Zstandard
                with open(temp_tar_file_path, 'rb') as f:
                    data = f.read()
                compressed_data = zstd.compress(data)
                with open(output_path, 'wb') as f:
                    f.write(compressed_data)
        else:
            print("The specified input path is neither a file nor a directory.")
    except Exception as e:
        print(f"An error occurred while compressing: {e}")


# Function to sign data
def sign_data(output_path, signed_path, private_key_path, hash_size=256, passphrase=None):
    try:
        # Validate hash size
        if hash_size not in [256, 384, 512]:
            raise ValueError("Invalid hash size. Supported sizes are 256, 384, and 512.")
        # Load the private key for signing
        with open(private_key_path, 'rb') as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=passphrase.encode('utf-8') if passphrase else None,
                backend=default_backend()
            )
        # Select hash algorithm based on hash_size
        if hash_size == 256:
            hash_algorithm = hashes.SHA256()
        elif hash_size == 384:
            hash_algorithm = hashes.SHA384()
        elif hash_size == 512:
            hash_algorithm = hashes.SHA512()
        else:
            raise ValueError("Unsupported hash size. Supported sizes are 256, 384, and 512.")
        # Read the compressed data
        with open(output_path, 'rb') as f:
            data = f.read()
        # Sign the data
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hash_algorithm),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hash_algorithm
        )
        # Combine the data and signature with a separator for robustness
        separator = b"---SIGNATURE_SEPARATOR---"
        combined_data = data + separator + signature
        # Write the signed data
        with open(signed_path, 'wb') as f:
            f.write(combined_data)
    except Exception as e:
        print(f"An error occurred while signing: {e}")


# Function to decompress data
def decompress_data(input_path, output_path):
    try:
        with open(input_path, 'rb') as f:
            compressed_data = f.read()

        decompressed_data = zstd.decompress(compressed_data)

        # Determine if the decompressed data is a tar archive (folder) or a regular file
        try:
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(decompressed_data)
                tmp_file.seek(0)

                # Attempt to open as a tar file to check if it's an archive
                with tarfile.open(fileobj=tmp_file, mode='r:') as tar:
                    # Successfully opened, so it's an archive; extract it
                    tar.extractall(path=output_path)
        except tarfile.ReadError:
            # Not a tar archive; assume it's a regular file and write it out
            with open(output_path, 'wb') as f:
                f.write(decompressed_data)

    except Exception as e:
        print(f"An error occurred while decompressing: {e}")
