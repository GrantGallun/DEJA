# Security policy

Deja runs locally. Its settings app and demo server bind loopback and reject requests from other
origins. They read model credentials from the environment or the checkout's `.env`. The settings app
shows only whether a key exists, never the key.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through GitHub's
[private vulnerability reporting](https://github.com/GrantGallun/DEJA/security/advisories/new) for this
repository. Include what is affected, how to reproduce it, and what an attacker could do with it.
You should get a reply within a week.

Only the latest commit on `main` is supported.
