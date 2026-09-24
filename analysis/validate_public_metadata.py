"""Lightweight structural validation of the citation and artifact CI contract."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
assert citation["cff-version"] == "1.2.0"
assert citation["authors"] == [{"given-names": "Simon P.", "family-names": "Villani"}]
assert citation["preferred-citation"]["authors"] == citation["authors"]
assert citation["preferred-citation"]["title"] == citation["title"]
assert citation["preferred-citation"]["year"] == 2026
assert citation["preferred-citation"]["type"] == "article"
assert citation["type"] == "software"
assert "submit/" not in str(citation)
assert "doi" not in citation and "doi" not in citation["preferred-citation"]
# BaseLoader preserves YAML 1.2's literal 'on' key despite PyYAML's YAML 1.1 defaults.
workflow = yaml.load((ROOT / ".github/workflows/verify.yml").read_text(encoding="utf-8"),
                     Loader=yaml.BaseLoader)
assert set(workflow["on"]) == {"push", "pull_request"}
assert workflow["permissions"] == {"contents": "read"}
steps = workflow["jobs"]["artifact-statistics"]["steps"]
commands = {step.get("run") for step in steps}
for script in ("extract_results", "verify_results", "reproduce_figures", "reproduce_tables"):
    assert f"python analysis/{script}.py" in commands
assert not any("--available-only" in (command or "") for command in commands)
print("CITATION.cff YAML and workflow structure: PASS")
