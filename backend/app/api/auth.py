from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import jwt
from fastapi import HTTPException, Request, status
from jwt import PyJWKClient

from app.core.config import Settings, get_settings


@dataclass(frozen=True)
class CurrentUser:
    subject: str
    email: str | None
    claims: dict[str, Any]

    def is_admin(self, group: str) -> bool:
        groups = self.claims.get("cognito:groups", [])
        return isinstance(groups, list) and group in groups


@lru_cache(maxsize=8)
def _jwks_client(uri: str) -> PyJWKClient:
    return PyJWKClient(uri)


def _auth_error(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def validate_access_token(token: str, settings: Settings | None = None) -> CurrentUser:
    settings = settings or get_settings()
    if not settings.cognito_issuer or not settings.cognito_client_id:
        raise _auth_error("Cognito authentication is not configured for this API.")
    try:
        key = _jwks_client(settings.cognito_jwks_uri).get_signing_key_from_jwt(token)
        claims = jwt.decode(token, key.key, algorithms=["RS256"], issuer=settings.cognito_issuer, options={"require": ["sub", "iss", "exp", "token_use"]})
    except (jwt.PyJWTError, ValueError) as exc:
        raise _auth_error("The Cognito access token is invalid or expired.") from exc
    if claims.get("token_use") != "access" or claims.get("client_id") != settings.cognito_client_id:
        raise _auth_error("The access token is not valid for this API.")
    return CurrentUser(subject=str(claims["sub"]), email=claims.get("email"), claims=claims)


def get_current_user(request: Request) -> CurrentUser | None:
    settings = get_settings()
    if not settings.auth_required:
        return None
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise _auth_error("A Cognito bearer access token is required.")
    user = validate_access_token(token, settings)
    request.state.user = user
    return user
