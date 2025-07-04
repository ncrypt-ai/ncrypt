class NcryptError(Exception):
    """
    Base class for all custom exceptions in this application.
    """
    pass


class PasswordError(NcryptError):
    """
    Raised when the user fails to set a password.
    """
    pass


class FilesystemCreationError(NcryptError):
    """
    Raised when the filesystem fails to be initialized.
    """
    pass


class DeletionError(NcryptError):
    """
    Raised when a file fails to be deleted from the filesystem while in remote mode.
    """
    pass


class UploadError(NcryptError):
    """
    Raised when a file fails to be uploaded while in remote mode.
    """
    pass


class DownloadError(NcryptError):
    """
    Raised when a file fails to be downloaded while in remote mode.
    """
    pass


class UnsupportedExtensionError(NcryptError):
    """
    Raised when a file has an extension that is either unsupported or of a different type
    than the metadata class that was provided.
    """
    pass


class ProcessingError(NcryptError):
    """
    Raised when there is an error attempting to process the raw content of a file.
    """
    pass


class SearchError(NcryptError):
    """
    Raised when there is an error attempting to search a files metadata.
    """
    pass
