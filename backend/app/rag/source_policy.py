from __future__ import annotations

from urllib.parse import urlparse


def _host(url: str) -> str:
    return (urlparse(url or "").hostname or "").lower()


def _matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith(f".{domain}")


INTERNATIONAL_SOURCES: tuple[tuple[str, str, str], ...] = (
    ("who.int", "international", "official_guideline"),
    ("bnf.nice.org.uk", "UK", "regulator_or_formulary"),
    ("nice.org.uk", "UK", "official_guideline"),
    ("sign.ac.uk", "UK", "official_guideline"),
    ("ema.europa.eu", "EU", "regulator_or_formulary"),
    ("ec.europa.eu", "EU", "regulator_or_formulary"),
    ("fda.gov", "US", "regulator_or_formulary"),
    ("dailymed.nlm.nih.gov", "US", "regulator_or_formulary"),
    ("pubmed.ncbi.nlm.nih.gov", "international", "indexed_literature"),
    ("professional.heart.org", "US", "official_guideline"),
    ("ahajournals.org", "US", "official_guideline"),
    ("heart.org", "US", "official_guideline"),
    ("aan.com", "US", "official_guideline"),
    ("guidelinecentral.com", "US", "secondary_guideline_index"),
    ("cdc.gov", "US", "official_guideline"),
    ("nih.gov", "US", "official_guideline"),
    ("eso-stroke.org", "EU", "official_guideline"),
    ("ean.org", "EU", "official_guideline"),
    ("escardio.org", "EU", "official_guideline"),
    ("eular.org", "EU", "official_guideline"),
    ("ersnet.org", "EU", "official_guideline"),
    ("esmo.org", "EU", "official_guideline"),
    ("cochranelibrary.com", "international", "evidence_synthesis"),
    ("cochrane.org", "international", "evidence_synthesis"),
    ("europepmc.org", "international", "indexed_literature"),
    ("strokebestpractices.ca", "Canada", "official_guideline"),
    ("strokefoundation.org.au", "Australia", "official_guideline"),
    ("braintrauma.org", "international", "official_guideline"),
    ("iscord.org", "international", "official_guideline"),
)


RUSSIAN_OFFICIAL_DOMAINS = (
    "cr.minzdrav.gov.ru",
    "minzdrav.gov.ru",
    "grls.rosminzdrav.ru",
    "roszdravnadzor.gov.ru",
    "publication.pravo.gov.ru",
)


def classify_international_source(url: str) -> dict[str, str | bool]:
    host = _host(url)
    for domain, jurisdiction, tier in INTERNATIONAL_SOURCES:
        if _matches(host, domain):
            return {
                "source_tier": tier,
                "jurisdiction": jurisdiction,
                "official_domain": tier in {
                    "official_guideline",
                    "regulator_or_formulary",
                },
            }
    return {
        "source_tier": "other",
        "jurisdiction": "unknown",
        "official_domain": False,
    }


def is_russian_official(url: str) -> bool:
    host = _host(url)
    return any(_matches(host, domain) for domain in RUSSIAN_OFFICIAL_DOMAINS)


def is_international_evidence_source(url: str) -> bool:
    info = classify_international_source(url)
    return info["source_tier"] in {
        "official_guideline",
        "regulator_or_formulary",
        "evidence_synthesis",
        "indexed_literature",
    }


def is_primary_international_source(url: str) -> bool:
    info = classify_international_source(url)
    return info["source_tier"] in {
        "official_guideline",
        "regulator_or_formulary",
    }
