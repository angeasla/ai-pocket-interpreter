"""Translation engine wrapping CTranslate2 NLLB-200."""

import logging

import ctranslate2
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TranslationEngine:
    """Wraps CTranslate2 NLLB-200 for multi-language translation.

    The model and tokenizer are loaded once via :meth:`load` at application
    startup and reused for all subsequent translations.
    """

    TARGET_LANGUAGES: list[str] = ["en", "el", "es"]
    TARGET_FLORES: dict[str, str] = {
        "en": "eng_Latn",
        "el": "ell_Grek",
        "es": "spa_Latn",
    }

    def __init__(
        self,
        model_path: str,
        tokenizer_name: str,
        device: str = "cuda",
        compute_type: str = "int8",
    ) -> None:
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.device = device
        self.compute_type = compute_type
        self._translator: ctranslate2.Translator | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> None:
        """Load the CTranslate2 Translator and tokenizer once at startup."""
        logger.info(
            "Loading NLLB-200 model from '%s' on %s (%s)…",
            self.model_path,
            self.device,
            self.compute_type,
        )
        self._translator = ctranslate2.Translator(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        logger.info("NLLB-200 model and tokenizer loaded successfully.")

    def translate(self, text: str, source_flores: str) -> dict[str, str]:
        """Translate text into all three target languages.

        Args:
            text: The source text to translate.
            source_flores: The FLORES-200 code of the source language.

        Returns:
            A dictionary mapping target language ISO codes to translated
            strings, e.g. ``{"en": "...", "el": "...", "es": "..."}``.
            On per-language failure, the value is ``""``.
        """
        if self._translator is None or self._tokenizer is None:
            raise RuntimeError(
                "TranslationEngine.load() must be called before translate()"
            )

        results: dict[str, str] = {}

        for lang_code in self.TARGET_LANGUAGES:
            target_flores = self.TARGET_FLORES[lang_code]
            try:
                self._tokenizer.src_lang = source_flores
                encoded = self._tokenizer.encode(text)
                source_tokens = [
                    self._tokenizer.convert_ids_to_tokens(encoded)
                ]
                target_prefix = [[target_flores]]

                output = self._translator.translate_batch(
                    source_tokens,
                    target_prefix=target_prefix,
                )

                translated_tokens = output[0].hypotheses[0]
                translated_text = self._tokenizer.decode(
                    self._tokenizer.convert_tokens_to_ids(translated_tokens)
                )
                results[lang_code] = translated_text
            except Exception:
                logger.exception(
                    "Translation failed for target language '%s' (%s)",
                    lang_code,
                    target_flores,
                )
                results[lang_code] = ""

        return results
