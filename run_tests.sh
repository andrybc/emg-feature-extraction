#!/usr/bin/env bash
# Run the test suite with ROS pytest plugins disabled.
# The ROS 2 install at /opt/ros registers pytest plugins via PYTHONPATH
# that crash collection in this project. Disabling plugin autoload avoids
# them; this project's tests need no plugins.
#
# `python -m pytest` rather than bare `pytest`: the latter resolves
# through PATH and can pick up /usr/bin/pytest, which runs against system
# site-packages instead of the project venv.
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest "$@"