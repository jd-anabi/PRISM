"""Fixtures for tests/ (the root conftest.py sets the environment and the marker skips).

`qapp` is the one QApplication a process may hold. The suites historically built it themselves
with ``QApplication.instance() or QApplication([])`` inside a helper; that stays valid, and this
fixture is the same expression for tests that prefer to declare the dependency.
"""
import pytest


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
