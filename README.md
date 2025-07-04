<h1 align="center">&#128274; ncrypt</h1>
<hr/>

<p align="center">
  <a href="https://github.com/ncryptai/ncrypt/releases"><img src="https://img.shields.io/github/v/release/ncryptai/ncrypt?style=flat-square"></a>
  <a href="https://github.com/ncryptai/ncrypt/blob/main/LICENSE.txt"><img src="https://img.shields.io/badge/License-BSD--3--Clause--Clear-%23ffb243?style=flat-square"></a>
</p>

## About

---

### What is ncrypt

Ncrypt is an end-to-end secure file manager that offers encrypted local and remote storage, secure metadata management, 
and privacy-preserving search over encrypted data using fully homomorphic encryption (FHE). It is designed to give users
complete control over their data, even in environments where the underlying storage or server infrastructure is not 
trusted. 

Search functionality in ncrypt allows encrypted search queries to be evaluated directly over encrypted content without
ever revealing the underlying plaintext. This ensures that even if a remote server is compromised, no sensitive 
information can be leaked. To protect the integrity and confidentiality of file contents, all files are encrypted using 
AES-256 before storage or transfer. For encrypted search and metadata processing, ncrypt uses 
[concrete](https://github.com/zama-ai/concrete/blob/main/README.md), a fully homomorphic encryption framework by Zama. 
Direct encryption and decryption of raw files is handled using the Python [cryptography](https://cryptography.io/en/latest/)
library.

Ncrypt follows the principle that users ***not platforms*** should control access to their data.

---

### Key Features

* **End-to-end encryption** of data at rest, in transit, and during computation ensuring compliance with privacy laws (GDPR, HIPAA, etc)
* **Fully Homomorphic Encryption (FHE)** powered search over encrypted metadata
* **Quantum-resistant cryptography** using AES-256 and secure key management
* **Confidential cloud storage** on untrusted infrastructure such as AWS S3
* **Seamless local and remote modes** with a unified command-line interface
* **Integration with Huggingface models** for embedding generation and intelligent metadata extraction
* **Granular key wrapping and rotation** per file to limit key exposure, with the ability to easily rotate keys
* **Built-in shell interface** with SFTP-inspired commands and tab completion

---

## Table Of Contents

- [Installation](#installation)
  - [External Dependencies](#external-dependencies)
- [Usage](#usage)
  - [Commands](#commands)
  - [Security](#security)
  - [Performance](#performance)
  - [A Simple Example](#a-simple-example)
- [Resources](#resources)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

You can install `ncrypt` from PyPI using the following commands:

```commandline
pip install -U pip wheel setuptools
pip install ncrypt
```

For developers, you can install `ncrypt` from source using the following commands:

```commandline
pip install -U pip wheel setuptools
git clone https://github.com/ncrypt-ai/ncrypt.git
```

**With pip**

```commandline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-doc.txt
```

**With conda**

```commandline
conda create -f environment.yaml
conda activate base
```

You can verify that ncrypt was installed successfully by running `which ncrypt`.

### External Dependencies

In order to run `ncrypt`, you must have:

* `python >= 3.11`
* `glibc >= 2.28` On Linux, you can check your glibc version by running `ldd --version`.
* Currently, `ncrypt` only supports macOS 11+ and Linux

---

## Usage

In its current state, `ncrypt` is intended to be used as a command line tool, and it supports a superset of the commands
utilized by the Secure File Transfer Protocol (SFTP). Ncrypt can be used both locally and remotely, with the remote mode
leveraging AWS S3 as the file storage backend. All configuration files are stored in the `~/.ncrypt` directory.

To start `ncrypt`, run:

```commandline
ncrypt [--remote]
```

After the `ncrypt` shell is running you can type `help` at any time to view a list of commands, or type `quit` to exit.

### Commands

The following commands are currently supported by `ncrypt`. This is intended to be an overview of what each command does,
but does not specify the specific flags that each command supports. To view details about how to use a command, simply
add `--help` or `-h` after the command.

#### Local Commands

* `lls` - List files in local directory.
* `lcd <dir>` - Change the local directory.
* `lpwd` - Print the current local working directory.
* `lmkdir <name>` - Create a directory in local file system.

#### File Management

* `ls [dir]` - List all files in the virtual current working directory or the specified virtual directory
* `cd <dir>` - Change the current working directory in the virtual file system
* `mkdir <dir>` - Create a virtual directory
* `rmdir <dir>` - Remove a virtual directory
* `rm <file>` - Remove a file and any of its associated metadata
* `mv <src> <dst>` - Move or rename a file or virtual directory
* `put <file> [dst]` - Encrypt a local file and upload it to the virtual directory
* `get <file> [dst]` - Download a file from the virtual file system to a local directory and decrypt it

#### Metadata and Encryption

* `rot` - Rotate the Key Encryption Key (KEK) for all files in the virtual file system or rotate the Data Encryption Key (DEK) for a single file
* `meta <file>` - Extract encrypted metadata from a file and upload it to the virtual file system. Run this command to make contents of a file searchable without decryption
* `search <dir>` - Search for files in a given virtual directory by multiple available attributes such as name, date created/modified, and extension. This operation is not recursive. Also allows you to search over encrypted metadata using fully homomorphic encryption

***Note: The `meta` and `search` commands can only be run in remote mode.*** When running in local mode there is no 
significant performance cost associated with accessing the entirety of a files contents as it just requires decryption
of the file. As such, it does not make sense from a performance perspective to perform FHE operations directly on the
encrypted files.

#### Misc

* `pwd` - Print the virtual current working directory
* `clear` - Clear the terminal screen
* `help` - Show help for commands
* `quit` - Exit the terminal

### Security

The `ncrypt` tool operates under the assumption that the host machine it is running on is safe and trusted. If that is 
not the case, no piece of software will be secure, and you will have much bigger problems to worry about. However, we
make no such assumption about any of the cloud machines used by our remote file management backend (AWS S3) and `ncrypt` 
is designed to be able to store and manage files securely on untrusted devices. The plain-text contents of your files
will *never* leave your machine.

### Performance

While `ncrypt` was developed with performance in mind, Fully Homomorphic Encryption (FHE) is often orders of magnitude
slower than performing the same operations on plintext. The cryptographic keys necessary to perform operations directly
on encrypted data are also typically very large (on the order of hundreds of MB). As such, you will notice a significant
delay when you first start `ncrypt` as those keys are generated and the shared keys are uploaded. However, this is a 
one-time operation and it will only occur when running in remote mode.

Some tips to improve FHE search performance:
* The search command runs only on the files that are direct descendants of the specified directory, and the run time of FHE operations will increase proportionally with the number and size of files in that directory. For faster searches, avoid having a flat directory structure by creating more folders 
* Minimize search query length for faster encrypted evaluation
* Apply preprocessing to the files during metadata extraction. For example, rather than extracting all of the unique words in a document, you could summarize the document first, shrinking the search space

### A simple example

In the example below, we will open `ncrypt` in remote mode, create a directory, place two files in it, extract metadata
from each of those files, and perform a search over that metadata.

**Directory creation and file placement**

```commandline
ncrypt --remote
(/) ncrypt> mkdir test_dir
(/) ncrypt> put /Users/me/Documents/a.docx test_dir/a.docx
(/) ncrypt> put /Users/me/Documents/b.docx test_dir/b.docx
(/) ncrypt> cd test_dir
```

**Contents of a.docx**

```text
Hello world, how are you today?
```

**Contents of b.docx**

```text
Lorem ipsum!
```

***Metadata extraction***

```commandline
(/test_dir) ncrypt> meta a.docx --type text -sx
(/test_dir) ncrypt> meta b.docx --type text -sx
```

The above commands extract the raw contents of the specified file, indicating that they should be treated as text files.
The `-x` flag causes the files contents to be summarized, and the `-s` flag extracts each of the unique words from the
file, enabling encrypted keyword seach. If the `-s` flag were not provided, an embedding would be generated instead. The
output of this command is an encrypted array which is uploaded to the file management backend.

***Encrypted search***

```commandline
(/test_dir) ncrypt> search ./ text "world" -s
(/test_dir) ncrypt>
(/test_dir) ncrypt> Contents of test_dir/ matching search parameters:
(/test_dir) ncrypt>     a.docx
```

---

## Contributing

We firmly believe that products that claim to preserve privacy should be open-source, and that the community should be 
able to verify those claims. We welcome contributions to the codebase as well as bug reports and security audits that 
will help to ensure ncrypt remains secure.

For issues, questions, or feature requests, please open an issue on the [GitHub repository](https://github.com/ncryptai/ncrypt/issues) or send us an [email](mailto:ncrypt.ai.dev@gmail.com).

---

## License

This software is distributed under the BSD-3-Clause-Clear license. See the [LICENSE.txt](https://github.com/ncryptai/ncrypt/blob/main/LICENSE.txt) file for details.
