from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "build" / "multi_agent"

ROLE_PLANNER = "planner"
ROLE_GEOSTAT = "geostat"
ROLE_GEOLOGY = "geology"
ROLE_PYTHON_DEV = "python_dev"
ROLE_WRITER = "writer"
ROLE_EVIDENCE = "evidence_guard"
ROLE_REVIEWER = "jaes_reviewer"

STRICT_GEOSTAT_ARTIFACTS = [
    "validation_baseline_comparison.csv",
    "validation_baseline_summary.csv",
    "cross_validation_blocked_500.json",
    "cross_validation_leave_hole.json",
    "cross_validation_leave_section_100m.json",
    "variogram_model.json",
    "variogram_pair_counts.csv",
]
STRICT_GEOLOGY_ARTIFACTS = [
    "contact_analysis.csv",
    "weathering_summary.csv",
    "domain_uncertainty_summary.json",
    "thickness_geometry_summary.json",
    "supplementary_structural_map.png",
    "supplementary_representative_sections.png",
    "supplementary_anisotropy_orientation.png",
    "supplementary_structural_fabric_diagnostics.png",
]


@dataclass
class AgentTask:
    task_id: str
    role: str
    objective: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    task_id: str
    role: str
    objective: str
    assumptions: list[str]
    findings: list[str]
    citations_or_artifacts: list[str]
    confidence: str
    risks_or_limits: list[str]
    next_actions: list[str]
    status: str
    review_decision: str | None = None
    publishable: bool | None = None


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def notebooklm_auth_ok() -> tuple[bool, str]:
    code, out, err = run_cmd(["notebooklm", "list", "--json"])
    if code == 0:
        return True, out.strip()
    return False, (err or out).strip()


def notebooklm_ask(notebook_id: str, question: str) -> tuple[bool, str]:
    cmd = ["notebooklm", "ask", question, "--json", "--notebook", notebook_id]
    code, out, err = run_cmd(cmd)
    if code == 0:
        return True, out.strip()
    return False, (err or out).strip()


def classify_workstreams(objective: str) -> dict[str, bool]:
    text = objective.lower()
    geo_words = ("lithology", "geology", "structure", "stratiform", "host", "domain")
    geostat_words = ("geostatistics", "geostat", "sgs", "variogram", "variography", "kriging", "decluster", "uncertainty", "risk", "statistical")
    code_words = (
        "script",
        "python",
        "implement",
        "refactor",
        "fix",
        "pipeline",
        "code",
        "format",
        "table",
        "figure",
        "submission",
        "compliance",
        "docx",
        "package",
    )
    paper_words = ("paper", "manuscript", "write", "section", "journal", "draft")
    return {
        ROLE_GEOLOGY: any(w in text for w in geo_words),
        ROLE_GEOSTAT: any(w in text for w in geostat_words),
        ROLE_PYTHON_DEV: any(w in text for w in code_words),
        ROLE_WRITER: any(w in text for w in paper_words),
    }


class MultiAgentOrchestrator:
    def __init__(self, objective: str, notebook_id: str | None, max_cycles: int, out_dir: Path, no_notebooklm: bool) -> None:
        self.objective = objective.strip()
        self.notebook_id = notebook_id
        self.max_cycles = max(1, max_cycles)
        self.out_dir = out_dir
        self.no_notebooklm = no_notebooklm
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.state: dict[str, Any] = {
            "objective": self.objective,
            "started_at": self.started_at,
            "max_cycles": self.max_cycles,
            "notebook_id": self.notebook_id,
            "notebooklm_enabled": not self.no_notebooklm,
            "cycles": [],
        }
        self.must_reach_publishable = False

    def _supplement_dir(self) -> Path:
        primary = ROOT / "build" / "submission_work" / "supplement"
        if primary.exists():
            return primary
        fallback = ROOT / "submission" / "supplement"
        return fallback

    def _paper_path(self) -> Path:
        return ROOT / "build" / "submission_work" / "paper.md"

    def _s2_names(self) -> set[str]:
        s2 = ROOT / "submission" / "Supplementary_Data_S2.zip"
        if not s2.exists():
            return set()
        with zipfile.ZipFile(s2) as zf:
            return {Path(n).name for n in zf.namelist() if not n.endswith("/")}

    def _strict_role_failures(self, role: str) -> list[str]:
        failures: list[str] = []
        sup = self._supplement_dir()
        if role == ROLE_GEOSTAT:
            s2_names = self._s2_names()
            for name in STRICT_GEOSTAT_ARTIFACTS:
                if not ((sup / name).exists() or (name in s2_names)):
                    failures.append(f"missing geostat evidence: supplement/{name}")
            paper = self._paper_path()
            if paper.exists():
                txt = paper.read_text(encoding="utf-8", errors="replace").lower()
                if ("screening-stage" not in txt) and ("screening level" not in txt):
                    failures.append("missing screening-stage limitation language in manuscript")
        elif role == ROLE_GEOLOGY:
            s2_names = self._s2_names()
            for name in STRICT_GEOLOGY_ARTIFACTS:
                if not ((sup / name).exists() or (name in s2_names)):
                    failures.append(f"missing geology evidence: supplement/{name}")
        elif role == ROLE_PYTHON_DEV:
            required_paths = [
                ROOT / "scripts" / "build_paper_from_meta.py",
                ROOT / "scripts" / "build_submission_package.py",
                ROOT / "scripts" / "submission_preflight.py",
            ]
            for p in required_paths:
                if not p.exists():
                    failures.append(f"missing required script: {p.name}")
        return failures

    def _evidence_zip_failures(self) -> list[str]:
        failures: list[str] = []
        table_path = ROOT / "build" / "submission_work" / "tables_final.md"
        zip_path = ROOT / "submission" / "Supplementary_Data_S2.zip"
        if not table_path.exists():
            failures.append("missing tables_final.md for evidence map audit")
            return failures
        if not zip_path.exists():
            failures.append("missing submission/Supplementary_Data_S2.zip for evidence map audit")
            return failures
        table_txt = table_path.read_text(encoding="utf-8", errors="replace")
        mapped = sorted(set(re.findall(r"supplement/([A-Za-z0-9_.-]+)", table_txt)))
        with zipfile.ZipFile(zip_path) as zf:
            names = {Path(n).name for n in zf.namelist() if not n.endswith("/")}
        missing = [name for name in mapped if name not in names]
        if missing:
            failures.append(f"evidence map files missing from S2 zip: {missing}")
        paper = self._paper_path()
        if paper.exists():
            txt = paper.read_text(encoding="utf-8", errors="replace").lower()
            if "first deposit-scale stochastic uncertainty study" in txt:
                failures.append("absolute first-claim novelty wording detected")
        return failures

    def _planner(self, cycle_idx: int) -> list[AgentTask]:
        flags = classify_workstreams(self.objective)
        text = self.objective.lower()
        verdict_mode = any(k in text for k in ("not ready for jaes", "major revision", "reject", "publishable", "reviewer verdict"))
        self.must_reach_publishable = verdict_mode or ("publishable" in text)
        tasks: list[AgentTask] = []
        has_domain_science = flags[ROLE_GEOSTAT] or flags[ROLE_GEOLOGY]
        if flags[ROLE_GEOSTAT] or has_domain_science or verdict_mode or not any(flags.values()):
            tasks.append(
                AgentTask(
                    task_id=f"c{cycle_idx:02d}-{ROLE_GEOSTAT}",
                    role=ROLE_GEOSTAT,
                    objective=f"Produce geostatistics recommendations and validation-fix priorities for: {self.objective}",
                    payload={"strict_gate": "require geostat evidence artifact set + screening-stage limitation wording"},
                )
            )
        if flags[ROLE_GEOLOGY] or has_domain_science or verdict_mode or not any(flags.values()):
            tasks.append(
                AgentTask(
                    task_id=f"c{cycle_idx:02d}-{ROLE_GEOLOGY}",
                    role=ROLE_GEOLOGY,
                    objective=f"Produce geology framing, anisotropy, and structure-justification fixes for: {self.objective}",
                    payload={"strict_gate": "require geology-control evidence artifact set"},
                )
            )
        if flags[ROLE_PYTHON_DEV] or verdict_mode:
            tasks.append(
                AgentTask(
                    task_id=f"c{cycle_idx:02d}-{ROLE_PYTHON_DEV}",
                    role=ROLE_PYTHON_DEV,
                    objective=f"Convert approved reviewer fixes into concrete code/package tasks for: {self.objective}",
                    payload={"strict_gate": "require table-order/caption/equation-format and package checks"},
                )
            )
        tasks.append(
            AgentTask(
                task_id=f"c{cycle_idx:02d}-{ROLE_WRITER}",
                role=ROLE_WRITER,
                objective=f"Synthesize a paper-ready summary for: {self.objective}",
            )
        )
        tasks.append(
            AgentTask(
                task_id=f"c{cycle_idx:02d}-{ROLE_EVIDENCE}",
                role=ROLE_EVIDENCE,
                objective="Validate claim evidence from agent outputs.",
                payload={"strict_gate": "block if evidence map references files missing from S2 zip"},
            )
        )
        tasks.append(
            AgentTask(
                task_id=f"c{cycle_idx:02d}-{ROLE_REVIEWER}",
                role=ROLE_REVIEWER,
                objective="Run JAES-style review on current cycle outputs.",
            )
        )
        return tasks

    def _domain_agent(self, task: AgentTask, role_label: str) -> AgentResult:
        assumptions = [
            "Objective text reflects current user intent.",
            "Repository artifacts remain the source of truth for project-specific values.",
        ]
        findings = [f"{role_label} analysis scoped to objective: {self.objective}"]
        citations_or_artifacts = []
        risks = []
        confidence = "medium"
        status = "pass"

        if self.no_notebooklm:
            risks.append("NotebookLM disabled by CLI flag; findings are provisional.")
            findings.append("Provisional mode enabled: NotebookLM gate skipped by user.")
            citations_or_artifacts.append("provisional://no-notebooklm")
            confidence = "low"
        elif not self.notebook_id:
            risks.append("Notebook ID not provided; NotebookLM evidence gate cannot run.")
            confidence = "low"
            status = "needs_revision"
        else:
            ok, answer = notebooklm_ask(
                notebook_id=self.notebook_id,
                question=f"{role_label}: Provide evidence-backed guidance for this task: {self.objective}",
            )
            if ok:
                findings.append("NotebookLM evidence query succeeded.")
                citations_or_artifacts.append(f"notebooklm://{self.notebook_id}")
                findings.append(answer[:1200])
                confidence = "high"
            else:
                risks.append("NotebookLM query failed; re-authentication may be required.")
                risks.append(answer[:400])
                confidence = "low"
                status = "needs_revision"

        strict_failures = self._strict_role_failures(task.role)
        if strict_failures:
            findings.append("Strict role gate failed.")
            risks.extend(strict_failures)
            status = "needs_revision"

        return AgentResult(
            task_id=task.task_id,
            role=task.role,
            objective=task.objective,
            assumptions=assumptions,
            findings=findings,
            citations_or_artifacts=citations_or_artifacts,
            confidence=confidence,
            risks_or_limits=risks,
            next_actions=[
                "Add or refresh NotebookLM sources for this objective.",
                "Refine question prompts for targeted evidence extraction.",
            ],
            status=status,
        )

    def _python_dev(self, task: AgentTask) -> AgentResult:
        strict_failures = self._strict_role_failures(task.role)
        status = "pass" if not strict_failures else "needs_revision"
        risks = ["Requires explicit execution task for concrete file edits."]
        risks.extend(strict_failures)
        return AgentResult(
            task_id=task.task_id,
            role=task.role,
            objective=task.objective,
            assumptions=["Implementation must follow planner-approved constraints."],
            findings=[
                "Code implementation task prepared.",
                "No automatic file edits are applied by this orchestrator; this stage outputs executable spec notes only.",
            ],
            citations_or_artifacts=["agent.md", "skills.md"],
            confidence="medium",
            risks_or_limits=risks,
            next_actions=["Submit implementation subtasks to coding agent with file ownership."],
            status=status,
        )

    def _writer(self, task: AgentTask, prior_results: list[AgentResult]) -> AgentResult:
        supported = [r for r in prior_results if r.role in (ROLE_GEOSTAT, ROLE_GEOLOGY) and r.citations_or_artifacts]
        status = "pass" if supported else "needs_revision"
        risks = [] if supported else ["No validated domain evidence available for manuscript synthesis."]
        return AgentResult(
            task_id=task.task_id,
            role=task.role,
            objective=task.objective,
            assumptions=["Only validated scientific findings should be used."],
            findings=[
                "Writer stage completed structured synthesis draft.",
                f"Validated domain inputs available: {len(supported)}",
                "Humanizer policy: apply humanizer skill before finalizing writer output.",
            ],
            citations_or_artifacts=[f"task://{r.task_id}" for r in supported],
            confidence="medium" if supported else "low",
            risks_or_limits=risks,
            next_actions=[
                "Run humanizer skill on draft text and keep technical meaning unchanged.",
                "Proceed to evidence guard for claim classification.",
            ],
            status=status,
        )

    def _evidence_guard(self, task: AgentTask, prior_results: list[AgentResult]) -> AgentResult:
        unsupported: list[str] = []
        for result in prior_results:
            if result.role in (ROLE_GEOSTAT, ROLE_GEOLOGY, ROLE_WRITER):
                if not result.citations_or_artifacts:
                    unsupported.append(result.task_id)
        strict_failures = self._evidence_zip_failures()
        status = "pass" if not unsupported else "needs_revision"
        if strict_failures:
            status = "needs_revision"
        findings = ["Evidence classification complete."]
        if unsupported:
            findings.append(f"Unsupported outputs: {', '.join(unsupported)}")
        else:
            findings.append("All critical outputs include source/artifact support.")
        if strict_failures:
            findings.append("Evidence-map strict gate failed.")
            findings.extend(strict_failures)
        return AgentResult(
            task_id=task.task_id,
            role=task.role,
            objective=task.objective,
            assumptions=["Every major claim must be run-backed or citation-backed."],
            findings=findings,
            citations_or_artifacts=[f"task://{r.task_id}" for r in prior_results],
            confidence="high" if not unsupported else "low",
            risks_or_limits=(
                []
                if (not unsupported and not strict_failures)
                else ["Unsupported claims detected and blocked."] + strict_failures
            ),
            next_actions=["Fix unsupported tasks before final reviewer pass."] if (unsupported or strict_failures) else ["Proceed to reviewer gate."],
            status=status,
        )

    def _reviewer(self, task: AgentTask, prior_results: list[AgentResult]) -> AgentResult:
        blockers = [r.task_id for r in prior_results if r.status != "pass"]
        passed_roles = {r.role for r in prior_results if r.status == "pass"}
        required_roles = {ROLE_GEOSTAT, ROLE_GEOLOGY, ROLE_WRITER, ROLE_EVIDENCE}
        has_full_stack = required_roles.issubset(passed_roles)
        publishable = (not blockers) and has_full_stack
        status = "pass" if publishable else ("pass" if not blockers else "needs_revision")
        decision = "accept" if publishable else ("minor" if not blockers else "major")
        return AgentResult(
            task_id=task.task_id,
            role=task.role,
            objective=task.objective,
            assumptions=["Review outcome must prioritize scientific defensibility."],
            findings=[
                "JAES-style review completed.",
                f"Recommendation: {decision}",
                f"Blocking task count: {len(blockers)}",
                f"Publishable now: {'yes' if publishable else 'no'}",
            ],
            citations_or_artifacts=[f"task://{r.task_id}" for r in prior_results],
            confidence="high",
            risks_or_limits=[] if not blockers else [f"Blockers: {', '.join(blockers)}"],
            next_actions=["Finalize package."] if publishable else ["Return to planner with reviewer blockers and revision plan."],
            status=status,
            review_decision=decision,
            publishable=publishable,
        )

    def _run_task(self, task: AgentTask, prior_results: list[AgentResult]) -> AgentResult:
        if task.role == ROLE_GEOSTAT:
            return self._domain_agent(task, role_label="Senior Geostatistician")
        if task.role == ROLE_GEOLOGY:
            return self._domain_agent(task, role_label="Senior Geologist")
        if task.role == ROLE_PYTHON_DEV:
            return self._python_dev(task)
        if task.role == ROLE_WRITER:
            return self._writer(task, prior_results=prior_results)
        if task.role == ROLE_EVIDENCE:
            return self._evidence_guard(task, prior_results=prior_results)
        if task.role == ROLE_REVIEWER:
            return self._reviewer(task, prior_results=prior_results)
        raise ValueError(f"Unknown role: {task.role}")

    def run(self) -> dict[str, Any]:
        if not self.no_notebooklm:
            ok, detail = notebooklm_auth_ok()
            self.state["notebooklm_auth"] = "ok" if ok else "failed"
            self.state["notebooklm_auth_detail"] = detail[:1200]
        else:
            self.state["notebooklm_auth"] = "skipped"
            self.state["notebooklm_auth_detail"] = "NotebookLM disabled by user."

        for cycle_idx in range(1, self.max_cycles + 1):
            cycle: dict[str, Any] = {
                "cycle": cycle_idx,
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "tasks": [],
            }
            tasks = self._planner(cycle_idx=cycle_idx)
            task_results: list[AgentResult] = []
            for task in tasks:
                result = self._run_task(task, prior_results=task_results)
                task_results.append(result)
                cycle["tasks"].append(
                    {
                        "task": asdict(task),
                        "result": asdict(result),
                    }
                )
            blockers = [r.task_id for r in task_results if r.status != "pass"]
            reviewer = next((r for r in task_results if r.role == ROLE_REVIEWER), None)
            publishable = bool(reviewer.publishable) if reviewer is not None else False
            cycle["blockers"] = blockers
            cycle["status"] = "pass" if not blockers else "needs_revision"
            cycle["publishable"] = publishable
            cycle["completed_at"] = datetime.now().isoformat(timespec="seconds")
            self.state["cycles"].append(cycle)

            if publishable:
                self.state["final_status"] = "publishable"
                self.state["publishable"] = True
                break
            if not blockers and not self.must_reach_publishable:
                self.state["final_status"] = "pass"
                self.state["publishable"] = False
                break
        else:
            self.state["final_status"] = "needs_revision"
            self.state["publishable"] = False

        self.state["completed_at"] = datetime.now().isoformat(timespec="seconds")
        return self.state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Planner-first multi-agent orchestrator with NotebookLM evidence gates.",
    )
    parser.add_argument("--task", required=True, help="Top-level objective for the planner.")
    parser.add_argument("--notebook-id", default=None, help="NotebookLM notebook ID used for geostat/geology evidence queries.")
    parser.add_argument("--max-cycles", type=int, default=3, help="Maximum planner-review cycles (default: 3).")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for orchestration artifacts.")
    parser.add_argument("--no-notebooklm", action="store_true", help="Disable NotebookLM calls (provisional mode).")
    parser.add_argument("--until-publishable", action="store_true", help="Require reviewer publishable verdict before success.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = MultiAgentOrchestrator(
        objective=args.task,
        notebook_id=args.notebook_id,
        max_cycles=args.max_cycles,
        out_dir=out_dir,
        no_notebooklm=bool(args.no_notebooklm),
    )
    result = orchestrator.run()
    if args.until_publishable:
        if result.get("final_status") != "publishable":
            result["final_status"] = "needs_revision"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = out_dir / "orchestrator_latest.json"
    stamped_path = out_dir / f"orchestrator_{ts}.json"
    payload = json.dumps(result, indent=2)
    latest_path.write_text(payload, encoding="utf-8")
    stamped_path.write_text(payload, encoding="utf-8")

    print(payload)
    if result.get("final_status") not in {"pass", "publishable"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
