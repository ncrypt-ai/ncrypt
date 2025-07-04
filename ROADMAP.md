# Ncrypt Roadmap

This document outlines the development roadmap for `ncrypt`, an end-to-end secure, CLI-based file manager with local and remote 
storage capabilities and privacy preserving search functionality enabled by fully homomorphic encryption (FHE).

---

## MVP Tasks (v1.0.0)

- [x] Command shell using `cmd2`
- [x] Persistent command history
- [x] Basic error handling and screen clearing
- [x] Local and remote modes with SQLite backend
- [x] Autocomplete for all commands
- [x] File operations: `ls`, `cd`, `mkdir`, `rmdir`, `rm`, `mv`
- [x] Local equivalents: `lls`, `lcd`, `lpwd`, `lmkdir`
- [x] Upload and download with progress indicators: `get`, `put`
- [x] File encryption at rest and in transit (AES-256)
- [x] Key encryption key (KEK) and data encryption key (DEK) rotation: `rot`
- [x] Huggingface integration for embedding generation and feature extraction
- [x] Encrypted metadata extraction and search using fully homomorphic encryption (FHE): `meta`, `search`
- [x] Installation script
- [x] Create a README

---

## Short Term Tasks

- [ ] Automate building documentation
- [ ] Add authentication and API keys
- [ ] Transition from an open connection to long-polling for searches
- [ ] Refactor command registration to avoid manual assignment
- [ ] Unit tests for each command (`do_*`)
- [ ] Integration tests (local + remote)
- [ ] Test on Windows, macOS, Linux
- [ ] Implement an audit log for all file operations

---

## Long-Term Tasks

- [ ] Add support for use as a library
- [ ] Web interface wrapper for `ncrypt`
- [ ] File sharing: generate secure, time-limited tokens
- [ ] Plugin architecture for new backends (e.g., Azure, GCS)
- [ ] Replace local DB with cloud-based relational DB for remote mode
- [ ] Rewrite the CLI in Rust

---

## Major Release Tracker

| Version | Target Date | Highlights                         |
|---------|-------------|------------------------------------|
| v1.0.0  | ✅ Released  | MVP, CLI, local/remote, FHE search |

---

## How to Contribute

1. Check open issues or TODOs in code.
2. Fork and branch off of `dev`.
3. Add test coverage.
4. Submit a PR linked to a roadmap item or issue.

---

> ✨ This roadmap evolves — feel free to suggest changes in [Discussions](https://github.com/ncrypt-ai/ncrypt/discussions) or by opening an issue.
