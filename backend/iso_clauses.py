"""ISO 50001:2018 clause catalogue (clauses 4.1 – 10.2) and seeding helpers."""
from sqlalchemy.orm import Session
from models import ComplianceItem

ISO_SECTIONS = {
    "4": "Context of the organization",
    "5": "Leadership",
    "6": "Planning",
    "7": "Support",
    "8": "Operation",
    "9": "Performance evaluation",
    "10": "Improvement",
}

ISO_50001_CLAUSES = [
    {"clause_ref": "4.1", "clause_title": "Understanding the organization and its context"},
    {"clause_ref": "4.2", "clause_title": "Understanding the needs and expectations of interested parties"},
    {"clause_ref": "4.3", "clause_title": "Determining the scope of the energy management system"},
    {"clause_ref": "4.4", "clause_title": "Energy management system"},
    {"clause_ref": "5.1", "clause_title": "Leadership and commitment"},
    {"clause_ref": "5.2", "clause_title": "Energy policy"},
    {"clause_ref": "5.3", "clause_title": "Organization roles, responsibilities and authorities"},
    {"clause_ref": "6.1", "clause_title": "Actions to address risks and opportunities"},
    {"clause_ref": "6.2", "clause_title": "Objectives, energy targets and planning to achieve them"},
    {"clause_ref": "6.3", "clause_title": "Energy review"},
    {"clause_ref": "6.4", "clause_title": "Energy performance indicators"},
    {"clause_ref": "6.5", "clause_title": "Energy baseline"},
    {"clause_ref": "6.6", "clause_title": "Planning for collection of energy data"},
    {"clause_ref": "7.1", "clause_title": "Resources"},
    {"clause_ref": "7.2", "clause_title": "Competence"},
    {"clause_ref": "7.3", "clause_title": "Awareness"},
    {"clause_ref": "7.4", "clause_title": "Communication"},
    {"clause_ref": "7.5", "clause_title": "Documented information"},
    {"clause_ref": "8.1", "clause_title": "Operational planning and control"},
    {"clause_ref": "8.2", "clause_title": "Design"},
    {"clause_ref": "8.3", "clause_title": "Procurement"},
    {"clause_ref": "9.1", "clause_title": "Monitoring, measurement, analysis and evaluation of energy performance and the EnMS"},
    {"clause_ref": "9.2", "clause_title": "Internal audit"},
    {"clause_ref": "9.3", "clause_title": "Management review"},
    {"clause_ref": "10.1", "clause_title": "Nonconformity and corrective action"},
    {"clause_ref": "10.2", "clause_title": "Continual improvement"},
]

CLAUSE_TITLES = {c["clause_ref"]: c["clause_title"] for c in ISO_50001_CLAUSES}


def seed_compliance(db: Session, org_id: int, statuses: dict | None = None, evidence: dict | None = None):
    """Create all 26 clauses for an organization (statuses default to not_started)."""
    statuses = statuses or {}
    evidence = evidence or {}
    for c in ISO_50001_CLAUSES:
        ref = c["clause_ref"]
        db.add(ComplianceItem(
            org_id=org_id, clause_ref=ref, clause_title=c["clause_title"],
            status=statuses.get(ref, "not_started"), evidence=evidence.get(ref),
        ))
