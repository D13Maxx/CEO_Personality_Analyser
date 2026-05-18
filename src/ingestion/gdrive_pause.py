"""
Google Drive sync pause utility.

Suspends and resumes GoogleDriveFS.exe to prevent filesystem
contention during heavy I/O (scraping, PDF processing).
Uses pssuspend from Sysinternals; degrades gracefully if unavailable.
"""

import subprocess
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def _gdrive_is_running():
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq GoogleDriveFS.exe"],
            capture_output=True, text=True, timeout=5,
        )
        return "GoogleDriveFS.exe" in result.stdout
    except Exception:
        return False


def _get_gdrive_pids():
    result = subprocess.run(
        ["powershell", "-Command",
         "Get-Process -Name GoogleDriveFS -ErrorAction SilentlyContinue | "
         "ForEach-Object { $_.Id }"],
        capture_output=True, text=True, timeout=5,
    )
    return [int(p.strip()) for p in result.stdout.strip().split('\n') if p.strip()]


def pause_gdrive_sync():
    if not _gdrive_is_running():
        logger.info("Google Drive not running.")
        return False

    try:
        pids = _get_gdrive_pids()
        if not pids:
            return False

        for pid in pids:
            subprocess.run(
                ["powershell", "-Command",
                 f"& pssuspend64 {pid} 2>$null; "
                 f"if ($LASTEXITCODE -ne 0) {{ & pssuspend {pid} 2>$null }}"],
                capture_output=True, text=True, timeout=5,
            )

        logger.info("Paused Drive sync (%d processes).", len(pids))
        return True
    except Exception as e:
        logger.warning("Could not pause Drive: %s", e)
        return False


def resume_gdrive_sync():
    try:
        pids = _get_gdrive_pids()
        for pid in pids:
            subprocess.run(
                ["powershell", "-Command",
                 f"& pssuspend64 -r {pid} 2>$null; "
                 f"if ($LASTEXITCODE -ne 0) {{ & pssuspend -r {pid} 2>$null }}"],
                capture_output=True, text=True, timeout=5,
            )
        logger.info("Resumed Drive sync (%d processes).", len(pids))
        return True
    except Exception as e:
        logger.warning("Could not resume Drive: %s", e)
        return False


@contextmanager
def gdrive_paused():
    """
    Context manager that suspends Drive sync for the block's duration.
    If pausing fails, the block still runs -- just with sync active.
    """
    paused = False
    try:
        paused = pause_gdrive_sync()
        if not paused:
            logger.info("Tip: install Sysinternals PsSuspend for auto-pause, "
                        "or pause manually via the Drive tray icon.")
        yield paused
    finally:
        if paused:
            resume_gdrive_sync()
