"""Routines for handling (non-public) ground truth labels.
"""

import glob
import io
import pickle
import random

import cryptography
from cryptography.fernet import Fernet


def encrypt(file: str, key: bytes):
    """Encrypt the given file (passed as filename)."""
    f = Fernet(key)
    with open(file, "rb") as fh:
        data = fh.read()
    encrypted_data = f.encrypt(data)
    with open(file + ".secret", "wb") as fh:
        fh.write(encrypted_data)


class EncryptedFile:
    """Contextmanager for opening encrypted files"""

    def __init__(self, filename: str, key: bytes):
        if not isinstance(key, bytes):
            raise ValueError(
                "Pass a bytes object as the key. If key "
                "is supplied as a string, make sure to call "
                "encode() before passing the key to this "
                "function."
            )
        self.filename = filename
        self.key = key

    def __enter__(self):
        crypt = Fernet(self.key)
        with open(self.filename, "rb") as fh:
            data = crypt.decrypt(fh.read())
        self._stream = io.BytesIO(data)
        self._stream.seek(0)
        return self._stream

    def __exit__(self, type, value, traceback):
        self._stream.close()
