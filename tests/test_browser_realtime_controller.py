from pathlib import Path
import re
import shutil
import subprocess

import pytest

from assistive_writing_pad.display.web_app import HTML


def test_actual_pad_javascript_realtime_controller():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the JavaScript controller unit test')
    script = re.search(r'<script>(.*?)</script>', HTML, re.S).group(1)
    result = subprocess.run(
        [node, str(Path(__file__).with_name('browser_realtime.cjs'))],
        input=script, text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
