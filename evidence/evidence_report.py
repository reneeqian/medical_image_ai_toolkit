from dataclasses import dataclass, field
from typing import List
from datetime import datetime
from pathlib import Path
import json
import os

@dataclass
class EvidenceIssue:
    level: str                  # ERROR | WARN | INFO
    message: str
    requirement_id: str | None  # <-- ADD
    context: str | None = None


@dataclass
class EvidenceReport:
    subject: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    issues: List[EvidenceIssue] = field(default_factory=list)

    def error(self, message: str, requirement_id: str, context: str | None = None):
        self.issues.append(EvidenceIssue("ERROR", message, requirement_id, context))

    def warn(self, message: str, requirement_id: str, context: str | None = None):
        self.issues.append(EvidenceIssue("WARNING", message, requirement_id, context))

    def info(self, message: str, requirement_id: str, context: str | None = None):
        self.issues.append(EvidenceIssue("INFO", message, requirement_id, context))
    @property
    def has_errors(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)

    def summary(self) -> str:
        lines = [f"Evidence report for {self.subject}"]
        for i in self.issues:
            prefix = f"[{i.level}]"
            ctx = f" ({i.context})" if i.context else ""
            lines.append(f"{prefix} {i.message}{ctx}")
        return "\n".join(lines)
    
    def to_string(self) -> str:
        """
        Human-readable string representation suitable for exceptions.
        """
        lines = [f"Evidence report for {self.subject}"]

        for i in self.issues:
            prefix = f"[{i.level}]"
            req = f" [{i.requirement_id}]" if i.requirement_id else ""
            ctx = f" ({i.context})" if i.context else ""
            lines.append(f"{prefix}{req} {i.message}{ctx}")

        return "\n".join(lines)
    
    def print_summary(self) -> None:
        print("\n=== Evidence Report ===")
        print(f"Subject: {self.subject}")

        errors = [i for i in self.issues if i.level == "ERROR"]
        warnings = [i for i in self.issues if i.level == "WARNING"]
        infos = [i for i in self.issues if i.level == "INFO"]

        print(f"Errors:   {len(errors)}")
        print(f"Warnings: {len(warnings)}")
        print(f"Info:     {len(infos)}")

        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  ❌ {e.message}")
                if e.context:
                    print(f"     ↳ {e.context}")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  ⚠️  {w.message}")
                if w.context:
                    print(f"     ↳ {w.context}")

        if infos:
            print("\nInfo:")
            for i in infos:
                print(f"  ℹ️  {i.message}")
                if i.context:
                    print(f"     ↳ {i.context}")

        print("\n=== End Evidence Report ===\n")
        
        
    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "timestamp": self.timestamp.isoformat(),
            "has_errors": self.has_errors,
            "issues": [
                {
                    "level": i.level,
                    "message": i.message,
                    "context": i.context,
                    "requirement_id": i.requirement_id,
                }
                for i in self.issues
            ],
        }

    def to_markdown(self) -> str:
        lines = [f"# Evidence Report: {self.subject}", ""]
        for i in self.issues:
            ctx = f" ({i.context})" if i.context else ""
            req = f" [Req: {i.requirement_id}]" if i.requirement_id else ""
            lines.append(f"- **{i.level}**: {i.message}{ctx}")
            lines.append(f"- **{i.level}**: {i.message}{ctx}{req}")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps(self.to_dict(), indent=2))
        elif path.suffix in {".md", ".markdown"}:
            path.write_text(self.to_markdown())
        else:
            raise ValueError(f"Unsupported report format: {path}")

    def auto_save(self, name: str):
        root = os.getenv("EVIDENCE_OUTPUT_DIR")
        if not root:
            return

        ts = datetime.now().strftime("%H%M%S_%f")
        path = Path(root) / f"{name}_{ts}.json"
        self.save(path)
