# Secret handling

Never print or commit private keys, tokens, passwords, or `.env` contents. Prefer paths, key IDs, fingerprints, and redacted diagnostics; commands must not echo secret values. `/etc/jax/.env` and private key material remain external. Public roots remain integrity-sensitive.
