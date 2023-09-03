import argparse
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


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
        # Read the public key
        with open(public_pem_path, 'rb') as f:
            public_pem = f.read()

        # Read the signed file
        with open(signed_file_path, 'rb') as f:
            signed_data = f.read()

        # Separate the original data and the signature
        separator = b"---SIGNATURE_SEPARATOR---"
        original_data, signature = signed_data.split(separator)

        # Load the public key from PEM format
        public_key = serialization.load_pem_public_key(public_pem, backend=default_backend())

        # Verify the signature
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the digital signature of a signed file.")
    parser.add_argument("-s", "--signed_file", required=True, help="Path to the signed file.")
    parser.add_argument("-p", "--public_key", required=True, help="Path to the public key in PEM format.")

    args = parser.parse_args()

    result = verify_signed_data(args.signed_file, args.public_key)

    if result:
        print("The signature is valid.")
    else:
        print("The signature is NOT valid.")
