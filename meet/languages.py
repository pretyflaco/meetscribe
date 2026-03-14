"""Unified language constants for meetscribe.

Single source of truth for language names, section headers, PDF labels,
alignment model registry, and RTL detection.  Eliminates three duplicate
language-name dicts that were spread across transcribe.py, summarize.py,
and cli.py.
"""

from __future__ import annotations


# ─── Language names ─────────────────────────────────────────────────────────

LANG_NAMES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "tr": "Turkish",
    "fa": "Persian (Farsi)",
}
"""Map ISO 639-1 codes to human-readable language names."""


# ─── RTL detection ──────────────────────────────────────────────────────────

RTL_LANGUAGES: frozenset[str] = frozenset({"fa", "ar", "he", "ur"})
"""Language codes that use right-to-left script."""


def is_rtl(lang: str) -> bool:
    """Return True if *lang* uses a right-to-left script."""
    return lang in RTL_LANGUAGES


# ─── Section headers (for AI summary prompts) ──────────────────────────────

SECTION_HEADERS: dict[str, dict[str, str]] = {
    "en": {
        "overview": "Meeting Overview",
        "topics": "Key Topics Discussed",
        "actions": "Action Items",
        "decisions": "Decisions Made",
        "questions": "Open Questions / Follow-ups",
        "none_stated": "None explicitly stated",
    },
    "de": {
        "overview": "Besprechungsübersicht",
        "topics": "Besprochene Hauptthemen",
        "actions": "Aufgaben",
        "decisions": "Getroffene Entscheidungen",
        "questions": "Offene Fragen / Nachverfolgung",
        "none_stated": "Keine ausdrücklich genannt",
    },
    "fr": {
        "overview": "Aperçu de la réunion",
        "topics": "Sujets clés abordés",
        "actions": "Points d'action",
        "decisions": "Décisions prises",
        "questions": "Questions ouvertes / Suivis",
        "none_stated": "Aucun mentionné explicitement",
    },
    "es": {
        "overview": "Resumen de la reunión",
        "topics": "Temas clave discutidos",
        "actions": "Puntos de acción",
        "decisions": "Decisiones tomadas",
        "questions": "Preguntas abiertas / Seguimientos",
        "none_stated": "Ninguno mencionado explícitamente",
    },
    "tr": {
        "overview": "Toplantı Özeti",
        "topics": "Tartışılan Ana Konular",
        "actions": "Eylem Maddeleri",
        "decisions": "Alınan Kararlar",
        "questions": "Açık Sorular / Takip Edilecekler",
        "none_stated": "Açıkça belirtilmedi",
    },
    "fa": {
        "overview": "خلاصه جلسه",
        "topics": "موضوعات کلیدی مورد بحث",
        "actions": "اقدامات لازم",
        "decisions": "تصمیمات اتخاذ شده",
        "questions": "سؤالات باز / پیگیری‌ها",
        "none_stated": "هیچ موردی صریحاً ذکر نشده است",
    },
}


# ─── PDF section labels ────────────────────────────────────────────────────

PDF_SECTIONS: dict[str, dict[str, str]] = {
    "en": {"summary": "AI Meeting Summary", "transcript": "Full Transcript"},
    "de": {"summary": "KI-Besprechungszusammenfassung", "transcript": "Vollständiges Transkript"},
    "fr": {"summary": "Résumé de réunion (IA)", "transcript": "Transcription complète"},
    "es": {"summary": "Resumen de reunión (IA)", "transcript": "Transcripción completa"},
    "tr": {"summary": "Yapay Zeka Toplantı Özeti", "transcript": "Tam Transkript"},
    "fa": {"summary": "خلاصه جلسه (هوش مصنوعی)", "transcript": "رونوشت کامل"},
}
