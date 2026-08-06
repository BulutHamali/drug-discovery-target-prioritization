from fastapi import Depends, HTTPException, Request, status

from app.api.auth import CurrentUser
from app.core.config import Settings, get_settings


def require_admin(request: Request, settings: Settings = Depends(get_settings)) -> CurrentUser | None:
    user = getattr(request.state, "user", None)
    if settings.auth_required and (user is None or not user.is_admin(settings.admin_group)):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required.")
    if settings.auth_required:
        return user
    return None


def require_public_read(request: Request) -> None:
    # Public result reads are intentionally separate from admin mutations.
    return None
