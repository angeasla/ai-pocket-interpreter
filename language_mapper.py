class LanguageMapper:
    """Maps Whisper ISO 639-1 language codes to NLLB-200 FLORES-200 codes."""

    MAPPING: dict[str, str] = {
        "en": "eng_Latn",
        "el": "ell_Grek",
        "es": "spa_Latn",
        "fr": "fra_Latn",
        "de": "deu_Latn",
        "ar": "arb_Arab",
        "zh": "zho_Hans",
        "ja": "jpn_Jpan",
        "ru": "rus_Cyrl",
        "pt": "por_Latn",
        "it": "ita_Latn",
        "ko": "kor_Hang",
    }

    def map(self, iso_code: str) -> str | None:
        """Return the FLORES-200 code for the given ISO 639-1 code, or None if unmapped."""
        return self.MAPPING.get(iso_code)
