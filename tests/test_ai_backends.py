"""
AI backend logic that can run without QGIS or network access.

Generators are built with __new__ so their __init__ (QSettings, auth database)
never runs; only the pure helpers are exercised.
"""

import pytest

from archeoglyph import defaults
from archeoglyph.generators.gemini_generator import GeminiGenerator
from archeoglyph.generators.huggingface_generator import HuggingFaceGenerator


# ---------------------------------------------------------------- model routing

def _gemini():
    """A Gemini generator without __init__ (which would touch QSettings)."""
    generator = GeminiGenerator.__new__(GeminiGenerator)
    generator._model_list_cache = None
    return generator


def _resolve(preferred, available, image_mode):
    return _gemini()._resolve_models_to_try(
        available_names=available, preferred_model=preferred, image_mode=image_mode
    )


def test_model_loop_is_capped():
    available = [f"gemini-3.{i}-pro-preview" for i in range(8)]
    assert len(_resolve("", available, False)) <= GeminiGenerator.MAX_MODELS_PER_ROUTE


def test_text_route_never_picks_an_image_model():
    available = ["gemini-2.5-flash-image", "gemini-2.5-flash-lite"]
    text_models = _resolve("", available, image_mode=False)
    assert all("image" not in name for name in text_models)

    image_models = _resolve("", available, image_mode=True)
    assert image_models and all("image" in name for name in image_models)


def test_excluded_keywords_are_filtered():
    available = ["gemini-2.5-flash-tts", "gemini-2.5-flash-lite"]
    assert all("tts" not in name for name in _resolve("", available, False))


# ---------------------------------------------------------------- SVG validation

class _StubContour:
    def generate_result(self, **kwargs):
        raise AssertionError("fallback should not run in these tests")


def _validator():
    generator = _gemini()
    generator.contour_gen = _StubContour()
    generator._render_svg_to_image = lambda svg, w, h: None
    return generator._validate_svg


def test_valid_svg_passes_without_a_silhouette():
    svg = (
        "Here is your symbol:\n```svg\n"
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<path d="M10,10 L90,10 L90,90 Z" fill="#8b4513"/></svg>\n```'
    )
    clean, reason = _validator()(svg, "Typology", None)
    assert clean and not reason and "```" not in clean


def test_scripted_svg_is_rejected_or_stripped():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<script>fetch("https://example.invalid")</script>'
        '<path d="M0,0 L5,5" fill="#111"/></svg>'
    )
    clean, _reason = _validator()(svg, "Typology", None)
    assert clean and "script" not in clean.lower()


def test_broken_and_overloaded_svg_is_rejected():
    clean, reason = _validator()("not an svg at all", "Typology", None)
    assert clean is None and reason

    many = "".join(f'<path d="M{i},{i} L{i + 1},{i + 1}" fill="#11{i % 10}{i % 10}00"/>' for i in range(40))
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">{many}</svg>'
    clean, reason = _validator()(svg, "Typology", None)
    assert clean is None and ("shapes" in reason or "colours" in reason)


# ---------------------------------------------------------------- HF routing

def test_model_lists_are_split_by_pipeline():
    assert defaults.HF_DEFAULT_MODEL_ID in defaults.HF_IMG2IMG_MODELS
    assert not set(defaults.HF_IMG2IMG_MODELS) & set(defaults.HF_TXT2IMG_MODELS)
    assert set(defaults.HF_FALLBACK_MODEL_IDS) == set(defaults.HF_IMG2IMG_MODELS) | set(defaults.HF_TXT2IMG_MODELS)
    # Guidance/negative prompts only go to models that accept them.
    assert not set(defaults.HF_GUIDANCE_MODELS) & set(defaults.HF_IMG2IMG_MODELS)


def _hf():
    generator = HuggingFaceGenerator.__new__(HuggingFaceGenerator)
    generator.model_id = defaults.HF_DEFAULT_MODEL_ID
    return generator


def test_normalize_model_id_accepts_urls_and_prefixes():
    normalize = _hf()._normalize_model_id
    assert normalize("https://huggingface.co/Qwen/Qwen-Image") == "Qwen/Qwen-Image"
    assert normalize("models/Qwen/Qwen-Image") == "Qwen/Qwen-Image"
    assert normalize("") == defaults.HF_DEFAULT_MODEL_ID


def test_call_model_only_sends_guidance_to_models_that_support_it():
    calls = {}

    class _Client:
        def text_to_image(self, prompt, model=None, **params):
            calls["text"] = (model, params)
            return b"png"

    call = _hf()._call_model

    call(_Client(), defaults.HF_GUIDANCE_MODELS[0], "p", steps=30, guidance=5.0)
    assert "guidance_scale" in calls["text"][1] and "negative_prompt" in calls["text"][1]

    call(_Client(), defaults.HF_IMG2IMG_MODELS[0], "p", steps=30, guidance=5.0)
    assert "guidance_scale" not in calls["text"][1] and "negative_prompt" not in calls["text"][1]


def test_try_models_stops_at_the_cap_and_reports_errors():
    attempted = []

    class _Client:
        def text_to_image(self, prompt, model=None, **params):
            attempted.append(model)
            raise RuntimeError("404 not found")

    generator = _hf()
    generator.LOAD_RETRIES = 0
    logs = []
    result = generator._try_models(
        client=_Client(),
        models_to_try=[f"org/model-{i}" for i in range(10)],
        prompt="p",
        error_logs=logs,
    )
    assert result is None
    assert len(attempted) == defaults.HF_MAX_MODELS_PER_ROUTE
    assert logs and all("404" in entry for entry in logs)


def test_try_models_honours_cancellation():
    class _Client:
        def text_to_image(self, prompt, model=None, **params):
            raise AssertionError("should not be called after cancellation")

    generator = _hf()
    generator.LOAD_RETRIES = 0
    logs = []
    result = generator._try_models(
        client=_Client(),
        models_to_try=["org/model"],
        prompt="p",
        error_logs=logs,
        cancel_check=lambda: True,
    )
    assert result is None and not logs


@pytest.mark.parametrize("service", ["gemini", "huggingface"])
def test_auth_module_falls_back_to_settings_without_qgis_auth(service):
    from archeoglyph import auth
    from tests.conftest import _FakeQSettings

    settings = _FakeQSettings()
    assert auth.set_api_key(service, "secret-token", settings) is False
    assert auth.get_api_key(service, settings) == "secret-token"
    auth.clear_api_key(service, settings)
    assert auth.get_api_key(service, settings) == ""


# ------------------------------------------------------- Korean subject terms

def _gemini_prompt(note):
    """The Gemini prompt for a user note, with everything else neutral."""
    generator = _gemini()
    generator.settings = None
    return generator._build_prompt(
        style_key="Simple Symbol",
        user_prompt_text=note,
        color=None,
        symmetry=False,
        silhouette_bytes=None,
        factuality=50,
        symbolic_looseness=50,
        exaggeration=0,
    )


def _hf_prompt(note):
    generator = _hf()
    generator.settings = None
    return generator._build_prompt(note, style="Simple Symbol")


def test_a_korean_note_tells_gemini_what_the_artifact_is():
    """
    The note reaches the model verbatim, and the English type name is added
    beside it - the model cannot act on "빗살무늬토기" on its own.
    """
    prompt = _gemini_prompt("빗살무늬토기 조각입니다")

    assert "빗살무늬토기 조각입니다" in prompt, "the user's own words must survive"
    assert "Comb-pattern Pottery" in prompt
    assert "photograph does not show" in prompt, "the caution must travel with it"


def test_a_korean_note_tells_hugging_face_what_the_artifact_is():
    prompt = _hf_prompt("세형동검 실측도")

    assert "세형동검 실측도" in prompt
    assert "the artifact is a Bronze Dagger (Slender)" in prompt


@pytest.mark.parametrize("builder", [_gemini_prompt, _hf_prompt])
def test_nothing_is_added_when_no_type_is_named(builder):
    """An English or vague note must not gain an invented subject."""
    plain = builder("make the outline cleaner")
    assert "SUBJECT" not in plain
    assert "the artifact is a" not in plain


@pytest.mark.parametrize("builder", [_gemini_prompt, _hf_prompt])
def test_an_empty_note_is_unchanged(builder):
    assert "SUBJECT" not in builder("")
    assert "the artifact is a" not in builder(None)


def test_the_specific_type_is_named_rather_than_the_generic_one():
    """
    "돌화살촉" must not reach the model as "Arrowhead": the shapes differ, and
    a generic subject is worse than none.
    """
    prompt = _gemini_prompt("돌화살촉 사진")
    assert "Stone Arrowhead" in prompt
