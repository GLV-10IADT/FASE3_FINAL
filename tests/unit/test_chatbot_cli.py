import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_chatbot_help_suppresses_known_dependency_deprecations() -> None:
    code = (
        "import runpy, sys\n"
        "sys.argv = ['scripts/chatbot_assistente.py', '--help']\n"
        "try:\n"
        "    runpy.run_path('scripts/chatbot_assistente.py', run_name='__main__')\n"
        "except SystemExit as error:\n"
        "    assert error.code == 0\n"
        "import peft\n"
    )
    result = subprocess.run(
        [sys.executable, "-W", "always", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Chat local acadêmico" in result.stdout
    assert result.stderr == ""
