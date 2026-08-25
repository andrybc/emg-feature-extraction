#!/usr/bin/env bash
# Run the test suite with ROS pytest plugins disabled.
# The ROS 2 install at /opt/ros registers pytest plugins via PYTHONPATH
# that crash collection in this project. Disabling plugin autoload avoids
# them; this project's tests need no plugins.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest "$@"
