# -*- coding: utf-8 -*-
"""
API key storage.

Keys are kept in the QGIS authentication database (encrypted with the user's
master password) rather than in plain QSettings. The QSettings entry is only
used to remember which auth configuration holds the key, and any key found in
the old plaintext entry is migrated on first read and then removed.
"""

from .log import log, log_exception

SERVICES = {
    "gemini": {
        "legacy_key": "ArcheoGlyph/gemini_api_key",
        "config_key": "ArcheoGlyph/gemini_auth_cfg",
        "name": "ArchaeoGlyph Gemini",
    },
    "huggingface": {
        "legacy_key": "ArcheoGlyph/huggingface_api_key",
        "config_key": "ArcheoGlyph/huggingface_auth_cfg",
        "name": "ArchaeoGlyph Hugging Face",
    },
}


def _auth_manager():
    """Return the QGIS auth manager when it is usable, else None."""
    try:
        from qgis.core import QgsApplication

        manager = QgsApplication.authManager()
        if manager is None or manager.isDisabled():
            return None
        return manager
    except Exception:
        return None


def _read_config(manager, config_id):
    from qgis.core import QgsAuthMethodConfig

    config = QgsAuthMethodConfig()
    if not manager.loadAuthenticationConfig(config_id, config, True):
        return None
    return config


def read_api_key(service, settings):
    """
    Return ``(key, readable)`` for ``service`` ("gemini" / "huggingface").

    ``readable`` is False when a key *is* stored but the authentication
    database would not hand it over - normally because the user dismissed the
    QGIS master-password prompt. The distinction matters: without it a locked
    database and an empty slot both looked like "no key", and saving the
    settings then wrote that emptiness over the real one.

    Migrates a plaintext key from the legacy QSettings entry on first use.
    """
    spec = SERVICES[service]
    manager = _auth_manager()
    if manager is not None:
        config_id = str(settings.value(spec["config_key"], "") or "").strip()
        if config_id:
            try:
                config = _read_config(manager, config_id)
            except Exception as e:
                log_exception(f"Could not read the stored {service} key", e)
                return "", False
            if config is None:
                log(f"A {service} key is stored but the authentication database "
                    f"would not open it; leaving it alone.", level="warning")
                return "", False
            stored = config.configMap().get("password", "")
            if stored:
                return stored, True

    legacy = str(settings.value(spec["legacy_key"], "") or "").strip()
    if legacy and manager is not None:
        # Move it into the auth database, then drop the plaintext copy.
        if set_api_key(service, legacy, settings):
            return legacy, True
    return legacy, True


def get_api_key(service, settings):
    """The stored key for ``service``, or "" when there is none to be had."""
    return read_api_key(service, settings)[0]


def set_api_key(service, api_key, settings):
    """
    Store ``api_key``. Returns True when it went into the auth database.

    Falls back to QSettings when the authentication system is unavailable
    (for example when the user has not set a master password).
    """
    spec = SERVICES[service]
    api_key = str(api_key or "").strip()
    manager = _auth_manager()

    if manager is None:
        settings.setValue(spec["legacy_key"], api_key)
        return False

    try:
        from qgis.core import QgsAuthMethodConfig

        config_id = str(settings.value(spec["config_key"], "") or "").strip()
        config = _read_config(manager, config_id) if config_id else None
        if config is None and config_id:
            # A key is stored and the database will not open it. Storing a new
            # one here would point the settings at a fresh empty config and
            # orphan the real key, so refuse and say why.
            log(f"Not overwriting the stored {service} key: the authentication "
                f"database did not open. Unlock it and try again.",
                level="warning")
            return False
        if config is None:
            config = QgsAuthMethodConfig()
            config.setMethod("Basic")
            config_id = ""
        config.setName(spec["name"])
        config.setConfig("username", "archeoglyph")
        config.setConfig("password", api_key)

        if config_id:
            stored = manager.updateAuthenticationConfig(config)
        else:
            stored = manager.storeAuthenticationConfig(config)
            config_id = config.id()
        if not stored or not config_id:
            settings.setValue(spec["legacy_key"], api_key)
            return False

        settings.setValue(spec["config_key"], config_id)
        settings.remove(spec["legacy_key"])
        return True
    except Exception as e:
        log_exception(f"Could not store the {service} key in the QGIS auth database", e)
        settings.setValue(spec["legacy_key"], api_key)
        return False


def clear_api_key(service, settings):
    """Remove a stored key from both the auth database and QSettings."""
    spec = SERVICES[service]
    manager = _auth_manager()
    config_id = str(settings.value(spec["config_key"], "") or "").strip()
    if manager is not None and config_id:
        try:
            manager.removeAuthenticationConfig(config_id)
        except Exception as e:
            log_exception(f"Could not remove the stored {service} key", e)
    settings.remove(spec["config_key"])
    settings.remove(spec["legacy_key"])


def storage_description(settings):
    """Short text for the settings dialog describing where keys are kept.

    Translated here rather than at the call site: the second branch is a
    security warning, and it went to the dialog as raw English prose with no
    catalogue entry, so the readers who most needed it could not read it.
    """
    from .i18n import tr

    if _auth_manager() is not None:
        return tr("Keys are stored in the QGIS authentication database.")
    log("QGIS authentication system unavailable; keys fall back to QSettings.", level="warning")
    return tr(
        "The QGIS authentication database is unavailable, so keys are stored in "
        "QGIS settings in plain text. Set a master password in "
        "Settings > Options > Authentication to protect them."
    )
