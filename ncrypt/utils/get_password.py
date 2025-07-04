import os

import keyring

from ncrypt.utils import SERVICE_NAME, USER_NAME


def get_password() -> bytes | None:
    """
    Retrieves a password (AES-256 key) from the keyring, prompting the user to set one if it
    doesn't exist.

    returns: bytes: The retrieved password, or None if the user chooses not to set it.
    """
    password = keyring.get_password(SERVICE_NAME, USER_NAME)

    if not password:
        print(f"No encryption key found for user '{USER_NAME}' under the '{SERVICE_NAME}' service.")

        set_password = input("Do you want to create an encryption key? (y/n): ").lower()

        if set_password == "y":
            kek: bytes = os.urandom(32)
            keyring.set_password(SERVICE_NAME, USER_NAME, kek.hex())

            return kek

        else:
            return None

    else:
        return bytes.fromhex(password)
