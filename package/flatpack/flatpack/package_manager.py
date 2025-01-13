import os
import tarfile
import tempfile
import zstandard as zstd

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class PackageManager:
    def _validate_flatpack_path(self, path, is_input=True, operation=None):
        absolute_path = os.path.abspath(path)

        if is_input:
            if not os.path.exists(absolute_path):
                raise FileNotFoundError(
                    f"The path '{absolute_path}' does not exist."
                )

            if operation == 'pack':
                if not os.path.isdir(absolute_path):
                    raise ValueError(
                        f"Input for pack operation must be a directory. Got: {absolute_path}"
                    )

                toml_path = os.path.join(absolute_path, "flatpack.toml")

                if not os.path.exists(toml_path):
                    raise ValueError(
                        f"Directory '{absolute_path}' is not a valid flatpack. Missing flatpack.toml file."
                    )

            if operation in ('unpack', 'sign'):
                if not os.path.isfile(absolute_path):
                    raise ValueError(
                        f"Input for {operation} operation must be a .fpk file."
                    )

                if not path.endswith('.fpk'):
                    raise ValueError(
                        f"Input must be .fpk file for {operation} operation"
                    )

        return absolute_path

    def pack(self, input_path, overwrite=False):
        try:
            print(f"Starting pack operation for {input_path}")

            abs_input_path = self._validate_flatpack_path(
                input_path,
                is_input=True,
                operation='pack'
            )

            print(f"Validated input path: {abs_input_path}")

            output_path = abs_input_path + '.fpk'
            print(f"Output path will be: {output_path}")

            if os.path.exists(output_path):
                if overwrite:
                    print(
                        f"Overwrite option is True. Attempting to remove existing file: {output_path}"
                    )

                    try:
                        os.remove(output_path)
                        print(
                            f"Successfully removed existing file: {output_path}"
                        )
                    except Exception as e:
                        print(
                            f"Failed to remove existing file: {output_path}. Error: {str(e)}"
                        )
                        raise
                else:
                    raise FileExistsError(
                        f"The package '{output_path}' already exists."
                    )

            if os.path.isfile(abs_input_path):
                print("Input is a file. Compressing single file.")

                with open(abs_input_path, 'rb') as f:
                    data = f.read()

                compression_level = 22
                compressed_data = zstd.compress(data, compression_level)

                with open(output_path, 'wb') as f:
                    f.write(compressed_data)
            elif os.path.isdir(abs_input_path):
                print(
                    "Input is a directory. Creating archive and compressing."
                )

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_tar_path = os.path.join(temp_dir, "temp.tar")

                    with tarfile.open(temp_tar_path, 'w') as tar:
                        for root, dirs, files in os.walk(abs_input_path):
                            if 'build' in dirs:
                                dirs.remove('build')

                            if 'web' in dirs:
                                dirs.remove('web')

                            for file in files:
                                file_path = os.path.join(root, file)

                                if (
                                        '/build/' in file_path or
                                        '/web/' in file_path or
                                        file_path.endswith('/build') or
                                        file_path.endswith('/web')
                                ):
                                    continue

                                arcname = os.path.relpath(
                                    file_path,
                                    start=abs_input_path
                                )

                                tar.add(file_path, arcname=arcname)

                                print(f"Added to FPK: {arcname}")

                    print("Compressing archive")

                    with open(temp_tar_path, 'rb') as f:
                        data = f.read()

                    compression_level = 22
                    compressed_data = zstd.compress(data, compression_level)

                    print(f"Writing compressed data to {output_path}")

                    with open(output_path, 'wb') as f:
                        f.write(compressed_data)

            print("Pack operation completed successfully")
        except Exception as e:
            print(f"Pack operation failed: {str(e)}")
            raise

    def unpack(self, input_path, output_path=None):
        try:
            abs_input_path = self._validate_flatpack_path(
                input_path,
                is_input=True,
                operation='unpack'
            )

            final_output_path = output_path if output_path else \
                os.path.splitext(abs_input_path)[0]

            if os.path.exists(final_output_path):
                raise FileExistsError(
                    f"The directory '{final_output_path}' already exists."
                )

            os.makedirs(final_output_path)

            with open(abs_input_path, 'rb') as f:
                compressed_data = f.read()

            decompressed_data = zstd.decompress(compressed_data)

            try:
                with tempfile.NamedTemporaryFile() as tmp_file:
                    tmp_file.write(decompressed_data)
                    tmp_file.seek(0)

                    with tarfile.open(fileobj=tmp_file, mode='r:') as tar:
                        tar.extractall(path=final_output_path)
            except tarfile.ReadError:
                with open(final_output_path, 'wb') as f:
                    f.write(decompressed_data)
        except Exception as e:
            raise

    def sign(
            self,
            input_path,
            output_path,
            private_key_path,
            hash_size=256,
            passphrase=None
    ):
        try:
            abs_input_path = self._validate_flatpack_path(
                input_path,
                is_input=True,
                operation='sign'
            )

            abs_output_path = self._validate_flatpack_path(
                output_path,
                is_input=False,
                operation='sign'
            )

            abs_key_path = self._validate_flatpack_path(
                private_key_path,
                is_input=True
            )

            if hash_size not in [256, 384, 512]:
                raise ValueError(
                    f"Invalid hash size {hash_size}. Must be 256, 384, or 512."
                )

            try:
                with open(abs_key_path, 'rb') as key_file:
                    key_data = key_file.read()
            except PermissionError:
                raise PermissionError(
                    f"Permission denied reading private key at {abs_key_path}"
                )

            try:
                private_key = serialization.load_pem_private_key(
                    key_data,
                    password=passphrase.encode(
                        'utf-8'
                    ) if passphrase else None,
                    backend=default_backend()
                )
            except TypeError:
                raise ValueError(
                    "Private key is encrypted. Please provide the correct passphrase."
                )
            except ValueError:
                raise ValueError(
                    "Invalid private key format or incorrect passphrase."
                )

            hash_algorithm = {
                256: hashes.SHA256(),
                384: hashes.SHA384(),
                512: hashes.SHA512()
            }[hash_size]

            try:
                with open(abs_input_path, 'rb') as f:
                    data = f.read()
            except PermissionError:
                raise PermissionError(
                    f"Permission denied reading package at {abs_input_path}"
                )

            try:
                signature = private_key.sign(
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hash_algorithm),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hash_algorithm
                )
            except Exception as e:
                raise ValueError(f"Signature generation failed: {str(e)}")

            separator = b"---SIGNATURE_SEPARATOR---"
            combined_data = data + separator + signature

            try:
                with open(abs_output_path, 'wb') as f:
                    f.write(combined_data)
            except PermissionError:
                raise PermissionError(
                    f"Permission denied writing signed package to {abs_output_path}"
                )

        except Exception as e:
            raise
