# Security Policy

## Reporting a Vulnerability

Report security issues privately to **maintainers@vectorweight.com**. Do not open public
issues for undisclosed vulnerabilities.

Include: affected component, reproduction steps, impact assessment, and suggested fix if
available. We aim to acknowledge reports within 5 business days.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | Yes       |

## Known Vulnerabilities

### CVE-2026-45829 — ChromaDB pre-authentication RCE (unpatched upstream)

| Field | Value |
| ----- | ----- |
| **CVE** | [CVE-2026-45829](https://nvd.nist.gov/vuln/detail/CVE-2026-45829) |
| **GHSA** | [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c) |
| **Severity** | CVSS 4.0 **10.0 CRITICAL** (HiddenLayer); CVSS 3.1 **10.0 CRITICAL** (Red Hat) |
| **Affected package** | `chromadb` ≥ 1.0.0, ≤ 1.5.9 (all published PyPI releases as of 2026-07-15) |
| **Pinned version** | `chromadb==1.5.9` (latest PyPI; **still vulnerable**) |
| **Upstream tracker** | [chroma-core/chroma#6717](https://github.com/chroma-core/chroma/issues/6717) |
| **Vendor patch** | **None** — no patched PyPI release available |

#### Description

A pre-authentication code-injection flaw in the ChromaDB Python server allows remote
attackers to achieve arbitrary code execution by crafting a malicious embedding model
configuration with `trust_remote_code=true` on the collection-creation API endpoint
(`/api/v2/tenants/{tenant}/databases/{db}/collections`).

A related client-SDK exposure exists when a Python client connects to a remote Chroma
server and uses a collection whose server-side embedding configuration was poisoned
(see [HiddenLayer research](https://www.hiddenlayer.com/research/chromatoast-served-pre-auth)).

> **Status: unpatched upstream.** Memory-gate cannot eliminate this CVE via dependency
> upgrade alone until Chroma publishes a fixed release.

#### Memory-gate exposure assessment

| Deployment mode | Exposure | Notes |
| --------------- | -------- | ----- |
| **Embedded client (default)** — `PersistentClient` / in-process `Client` | **Mitigated** | No Chroma HTTP server is started; RCE requires network-reachable Chroma server |
| **Remote `HttpClient`** — `CHROMA_SERVER_HOST` set | **High** | Client may execute poisoned server-side embedding configs |
| **Standalone Chroma server** — `chroma run` / Docker `chromadb/chroma` | **Critical** | Do **not** expose Python-backend Chroma to untrusted networks |

Memory-gate's production code path uses **embedded client-only** storage via
`VectorMemoryStore` (`src/memory_gate/storage/vector_store.py`), supplying its own
`SentenceTransformer` embeddings and never calling `chromadb.HttpClient`.

#### Implemented mitigations (defense-in-depth)

1. **Dependency pin** — `chromadb` pinned to latest PyPI (`1.5.9`) in `pyproject.toml`
   and `uv.lock`; `tool.uv.override-dependencies` prevents transitive downgrades.
2. **Client-only architecture** — default deployment uses `PersistentClient`; remote
   server mode is unsupported and must not be enabled without explicit risk acceptance.
3. **Network binding guidance** — see [docs/security/chromadb-cve-2026-45829.md](docs/security/chromadb-cve-2026-45829.md) for bind-address, firewall, and Kubernetes `NetworkPolicy` requirements when operating any Chroma server alongside memory-gate.
4. **Startup guard (optional)** — run `docs/security/chromadb_startup_guard.py` before
   application start, or set `MEMORY_GATE_CHROMA_SERVER_WARN=1` to emit a critical log
   when remote Chroma server environment variables are detected. Set
   `MEMORY_GATE_ENFORCE_CLIENT_ONLY=1` to fail fast in CI/staging.
5. **Metrics binding** — memory-gate binds Prometheus metrics to `127.0.0.1` by default
   (`METRICS_HOST`); do not expose Chroma or metrics ports on `0.0.0.0` without auth.

#### Residual risk (accepted)

| Risk | Likelihood | Impact | Decision |
| ---- | ---------- | ------ | -------- |
| Embedded-client deployments (default Helm chart) | Low | Low | **Mitigated** — no network Chroma surface |
| Operator enables `HttpClient` / external Chroma server | Medium | Critical | **Accepted with controls** — documented guard + network isolation required |
| Future Chroma server bundled in chart | Low | Critical | **Blocked** — not supported until upstream patch ships |

**Risk acceptance owner:** project maintainers  
**Review cadence:** weekly until patched PyPI release  
**Acceptance date:** 2026-07-15  
**Expiry:** first business day after `chromadb` patched release ≥ 1.5.10 (or vendor advisory)

#### Remediation timeline and monitoring

| Milestone | Action |
| --------- | ------ |
| **Now** | Pin `chromadb==1.5.9`; enforce client-only defaults; document controls |
| **Weekly** | Monitor [chroma#6717](https://github.com/chroma-core/chroma/issues/6717), [PyPI chromadb](https://pypi.org/project/chromadb/#history), [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c) |
| **On patch release** | Bump pin, run full regression suite, remove formal risk acceptance |
| **Quarterly** | Re-audit deployment docs and Helm values for accidental server exposure |

Subscribe to GitHub Dependabot alerts on this repository for automated `chromadb` bump PRs.

## Dependency Security

- Python 3.13 required (see `.python-version`).
- Transitive dependency floors enforced via `[tool.uv.override-dependencies]` in
  `pyproject.toml`.
- Run `uv run safety check` (dev group) for local vulnerability scans.

## Related Documentation

- [ChromaDB CVE mitigation guide](docs/security/chromadb-cve-2026-45829.md)
- [Optional startup guard script](docs/security/chromadb_startup_guard.py)