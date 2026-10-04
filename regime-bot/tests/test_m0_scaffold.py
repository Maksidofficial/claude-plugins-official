from pathlib import Path

from regimebot.ops.redact import redact

ROOT = Path(__file__).resolve().parents[1]
SECRET_KEYS = {
    "ALPACA_KEY_ID",
    "ALPACA_SECRET_KEY",
    "TELEGRAM_BOT_TOKEN",
    "ANTHROPIC_API_KEY",
}


def test_env_is_gitignored() -> None:
    lines = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in lines


def test_env_example_has_keys_with_empty_values() -> None:
    pairs = {}
    for line in (ROOT / ".env.example").read_text().splitlines():
        if line and not line.startswith("#"):
            k, _, v = line.partition("=")
            pairs[k] = v
    assert SECRET_KEYS <= pairs.keys()
    for k in SECRET_KEYS:
        assert pairs[k] == "", f"{k} must be empty in .env.example"


def test_redact_removes_secret_values() -> None:
    env = {"ALPACA_SECRET_KEY": "s3cr3t-abcdef", "TELEGRAM_BOT_TOKEN": "123:XYZ"}
    line = "auth failed with s3cr3t-abcdef and token 123:XYZ"
    out = redact(line, env)
    assert "s3cr3t-abcdef" not in out
    assert "123:XYZ" not in out
    assert "***" in out


def test_redact_ignores_empty_and_nonsecret_values() -> None:
    env = {"ALPACA_SECRET_KEY": "", "ALPACA_PAPER": "true"}
    assert redact("paper=true", env) == "paper=true"
