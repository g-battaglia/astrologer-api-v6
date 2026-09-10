# Astrologer API v6

FastAPI service for astrological calculations, SVG charts, and MCP tools, built on Kerykeion and libephemeris.

## Installation and startup

Requires Python 3.12 or later. From the project directory:

```bash
python -m pip install .
export PRIVATE_ASTROLOGER_API_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
uvicorn app.main:app --host 127.0.0.1 --port 8080
```

With the built-in defaults, protected requests use the `X-API-Key` header and the key configured above. Without a key, startup refuses an unauthenticated configuration. `GET /health` and `GET /ready` are public probes.

## Configuration

Environment variables override TOML values. `ASTROLOGER_CONFIG_FILE` selects an external TOML file; a missing or invalid file prevents startup. `ASTROLOGER_ENV_FILE` selects a dotenv file (default: `.env`). Never commit secrets.

Example non-secret configuration:

```toml
admin_email = "admin@example.com"
debug = false
enable_tracing = false
docs_url = "/docs"
redoc_url = "/redoc"
secret_key_names = ["X-API-Key"]
allowed_hosts = []
allowed_cors_origins = []
```

`ASTROLOGER_DEBUG` enables debug mode and disables the authentication middleware; use it only for local development. `ASTROLOGER_ENABLE_TRACING` controls tracing. The service retains its existing CORS policy; `allowed_hosts` and `allowed_cors_origins` are not currently enforced as restrictions.

REST endpoints live under `/api/v6`. MCP Streamable HTTP is available at `/api/v6/mcp/`. Interactive API documentation is available at `/docs`.

## Ephemeris data

Ephemeris data is not included. Configure libephemeris and provision its inventory according to the library documentation. A runtime managed through `LIBEPHEMERIS_DATA_DIR` requires a validated inventory and the `.initialized` marker; calculation routes remain unavailable until readiness validation succeeds. A running HTTP server alone does not mean that ephemeris data is ready.

## License

AGPL-3.0. See `LICENSE` and the source file notices.
