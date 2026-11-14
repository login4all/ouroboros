"""Route-aware best-effort pricing tests."""

from __future__ import annotations

import hashlib
import json
import queue
from unittest.mock import MagicMock, patch

import pytest

from ouroboros.llm import fetch_cloudru_pricing, fetch_openrouter_pricing
from ouroboros.pricing import (
    PricingSchedule,
    emit_llm_usage_event,
    estimate_cost_optional,
    get_pricing,
    infer_api_key_type,
    infer_model_category,
)


@pytest.fixture(autouse=True)
def _reset_pricing_cache():
    import ouroboros.pricing as pricing
    from ouroboros import pricing_catalog

    def clear():
        pricing._cached_pricing.clear()
        pricing._pricing_fetched_at.clear()
        pricing._pricing_retry_after.clear()
        pricing._pricing_fetch_in_progress.clear()
        pricing_catalog._cache.clear()
        pricing_catalog._retry_after.clear()
        pricing_catalog._fetching.clear()

    clear()
    yield
    clear()


def test_unknown_direct_model_cost_is_none_and_does_not_query_openrouter():
    with patch("ouroboros.llm.fetch_openrouter_pricing") as fetch:
        assert estimate_cost_optional(
            "openai::future-model", 1_000, 500, provider="openai",
        ) is None
    fetch.assert_not_called()


def test_provider_is_inferred_without_openrouter_fallback():
    with patch("ouroboros.llm.fetch_openrouter_pricing") as fetch:
        assert estimate_cost_optional(
            "openai::future-model", 1_000, 500, provider=None,
        ) is None
    fetch.assert_not_called()


def test_live_catalog_prices_exact_new_openrouter_model():
    with patch(
        "ouroboros.llm.fetch_openrouter_pricing",
        return_value={"openai/gpt-new": (2.0, 0.2, None, 8.0)},
    ) as fetch:
        cost = estimate_cost_optional(
            "openai/gpt-new", 1_000, 500, provider="openrouter",
        )
    assert cost == 0.006
    fetch.assert_called_once_with(timeout_sec=5.0)


def test_similar_model_name_does_not_inherit_prefix_price():
    with patch(
        "ouroboros.llm.fetch_openrouter_pricing",
        return_value={"openai/gpt-new": (2.0, 0.2, None, 8.0)},
    ):
        assert estimate_cost_optional(
            "openai/gpt-new:beta", 1_000, 500, provider="openrouter",
        ) is None


def test_failed_catalog_fetch_has_short_process_local_cooldown():
    with patch("ouroboros.llm.fetch_openrouter_pricing", return_value={}) as fetch:
        assert get_pricing(provider="openrouter") == {}
        assert get_pricing(provider="openrouter") == {}
    fetch.assert_called_once_with(timeout_sec=5.0)


def test_missing_cache_prices_are_not_invented():
    with patch(
        "ouroboros.llm.fetch_openrouter_pricing",
        return_value={"provider/model": (1.0, None, None, 3.0)},
    ):
        assert estimate_cost_optional(
            "provider/model", 1_000, 100, provider="openrouter",
        ) == 0.0013
        assert estimate_cost_optional(
            "provider/model", 1_000, 100, cache_usage={"cached_tokens": 10},
            provider="openrouter", allow_live_fetch=False,
        ) is None
        assert estimate_cost_optional(
            "provider/model", 1_000, 100, cache_usage={"cache_write_tokens": 10},
            provider="openrouter", allow_live_fetch=False,
        ) is None


def test_cache_heavy_anthropic_usage_keeps_a_nonzero_fresh_input_component():
    """v6.77.0 accounting boundary: `regular_input = prompt_tokens - cached - cache_write`
    only yields the FRESH input when `prompt_tokens` is the OpenAI-semantics TOTAL input.
    Direct Anthropic used to report `input_tokens` alone (cache reads/writes excluded), so
    the subtraction clamped fresh input to 0 and the row understated the prompt."""
    with patch(
        "ouroboros.llm.fetch_openrouter_pricing",
        return_value={"anthropic/claude-x": (3.0, 0.3, 3.75, 15.0)},
    ):
        # Provider row: input_tokens=500, cache_read=9_000, cache_creation=500.
        pre_fix = estimate_cost_optional(
            "anthropic/claude-x", 500, 100,
            cache_usage={"cached_tokens": 9_000, "cache_write_tokens": 500},
            provider="openrouter",
        )
        post_fix = estimate_cost_optional(
            "anthropic/claude-x", 10_000, 100,
            cache_usage={"cached_tokens": 9_000, "cache_write_tokens": 500},
            provider="openrouter",
        )

    cached_and_write = (9_000 * 0.3 + 500 * 3.75) / 1_000_000
    completion = 100 * 15.0 / 1_000_000
    assert pre_fix == round(cached_and_write + completion, 6)  # regular_input clamped to 0
    assert post_fix == round(500 * 3.0 / 1_000_000 + cached_and_write + completion, 6)
    assert post_fix > pre_fix


def test_exact_prompt_tier_is_applied_without_prefix_matching():
    row = PricingSchedule(
        (1.0, 0.1, None, 3.0),
        ((100_000, (2.0, 0.2, None, 5.0)),),
    )
    with patch("ouroboros.llm.fetch_openrouter_pricing", return_value={"x/model": row}):
        assert estimate_cost_optional(
            "x/model", 100_000, 1_000, provider="openrouter",
        ) == 0.205


def test_openrouter_catalog_accepts_arbitrary_model_family():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [{
            "id": "mistralai/brand-new",
            "pricing": {"prompt": "0.000002", "completion": "0.000006"},
        }]
    }
    with patch("requests.get", return_value=response) as request:
        rows = fetch_openrouter_pricing(timeout_sec=5.0)
    assert rows["mistralai/brand-new"] == (2.0, None, None, 6.0)
    request.assert_called_once_with("https://openrouter.ai/api/v1/models", timeout=5.0)


def test_cloudru_requires_explicit_fx_rate(monkeypatch):
    monkeypatch.setenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", "secret")
    monkeypatch.delenv("OUROBOROS_RUB_USD_RATE", raising=False)
    with patch("requests.get") as request:
        assert fetch_cloudru_pricing(timeout_sec=5.0) == {}
    request.assert_not_called()


def test_cloudru_catalog_uses_exact_model_and_explicit_fx(monkeypatch):
    monkeypatch.setenv("CLOUDRU_FOUNDATION_MODELS_API_KEY", "secret")
    monkeypatch.setenv("OUROBOROS_RUB_USD_RATE", "100")
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": [{
        "id": "vendor/new-model",
        "metadata": {
            "is_billable": True,
            "prompt_tokens_cost": 100,
            "generated_tokens_cost": 500,
            "cache_read_tokens_cost": None,
            "cache_write_tokens_cost": None,
        },
    }]}
    with patch("requests.get", return_value=response):
        rows = fetch_cloudru_pricing(timeout_sec=5.0)
    assert rows["cloudru/vendor/new-model"] == (1.0, None, None, 5.0)


@pytest.mark.parametrize("provider", ["openai", "gigachat", "anthropic"])
def test_routes_without_automatic_catalog_return_empty(provider):
    assert get_pricing(provider=provider) == {}


def test_nullable_usage_event_does_not_label_unknown_as_estimated():
    events = queue.Queue()
    emit_llm_usage_event(
        events,
        "task",
        "openai::future-model",
        {"prompt_tokens": 3, "completion_tokens": 2},
        None,
        provider="openai",
    )
    event = events.get_nowait()
    assert event["cost"] is None
    assert event["cost_estimated"] is False


def test_provider_reported_zero_cost_remains_known_zero():
    events = queue.Queue()
    emit_llm_usage_event(
        events, "task", "local/model", {"cost": 0}, 0.0, provider="local",
    )
    assert events.get_nowait()["cost"] == 0.0


def test_inference_helpers_keep_route_identity(monkeypatch):
    assert infer_api_key_type("openai::gpt-x") == "openai"
    assert infer_api_key_type("mistralai/model") == "openrouter"
    # Direct MiniMax uses the :: spelling; the slash form is a REAL OpenRouter
    # vendor namespace and must keep routing (and safety-key inference) through
    # the OpenRouter key, unlike cloudru/gigachat which exist nowhere on OpenRouter.
    assert infer_api_key_type("minimax::MiniMax-M3") == "minimax"
    assert infer_api_key_type("minimax/minimax-m2") == "openrouter"
    monkeypatch.setenv("OUROBOROS_MODEL", "mistralai/model")
    assert infer_model_category("mistralai/model") == "main"


def _endpoint(tag, prompt="0.000002", completion="0.000008", **pricing):
    return {"tag": tag, "model_id": "vendor/tier-model", "status": 0,
            "supported_parameters": ["temperature"],
            "pricing": {"prompt": prompt, "completion": completion, **pricing}}


@pytest.fixture
def endpoint_catalog():
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": {"id": "vendor/tier-model", "endpoints": [
        _endpoint("vendor", "0.000002", "0.000008"),
        _endpoint("vendor/flex", "0.000001", "0.000003"),
        _endpoint("vendor/fast", "0.000004", "0.000015"),
        _endpoint("other/region", "0.000005", "0.000006"),
    ]}}
    with patch("requests.get", return_value=response) as request:
        yield response.json.return_value["data"]["endpoints"], request


@pytest.mark.parametrize("mode,expected", [("default", 0.008), ("priority", 0.0115), ("flex", 0.0025)])
def test_processing_prices_the_complete_eligible_endpoint_pool(endpoint_catalog, mode, expected):
    rows, request = endpoint_catalog
    # Endpoint status and supported_parameters are not tariff eligibility:
    # unhealthy price rows remain covered; service_tier need not be in the list.
    rows[3]["status"] = -2
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode=mode) == pytest.approx(expected)
    request.assert_called_once_with("https://openrouter.ai/api/v1/models/vendor/tier-model/endpoints", timeout=5.0)


def test_a_dotted_router_id_reads_its_own_endpoints_not_its_shared_identity(endpoint_catalog):
    """``anthropic/claude-opus-5.5`` shares its identity with direct ``claude-opus-5-5``, yet its
    tariffs are OpenRouter's for the slug the route names."""
    rows, request = endpoint_catalog
    slug = "anthropic/claude-opus-5.5"
    request.return_value.json.return_value = {"data": {"id": slug, "endpoints": [{**r, "model_id": slug} for r in rows]}}
    assert estimate_cost_optional(slug, 1000, 500, processing_mode="fast") == pytest.approx(0.0115)
    request.assert_called_once_with(f"https://openrouter.ai/api/v1/models/{slug}/endpoints", timeout=5.0)


def test_priority_bound_includes_a_more_expensive_standard_fallback(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[3]["pricing"]["completion"] = "0.0001"
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") == pytest.approx(0.055)
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="flex") == pytest.approx(0.0025)


def test_flex_uses_standard_only_when_no_flex_endpoint_exists(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[:] = [row for row in rows if not row["tag"].endswith("/flex")]
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="flex") == pytest.approx(0.008)


def test_a_missing_potential_tariff_makes_the_bound_unknown(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[3]["pricing"].pop("completion")
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") is None
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="default") is None
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="flex") == pytest.approx(0.0025)


@pytest.mark.parametrize("mode", ["unknown", "auto", "unsupported"])
def test_unknown_processing_never_uses_an_ordinary_model_price(endpoint_catalog, mode):
    _, request = endpoint_catalog
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode=mode) is None
    request.assert_not_called()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "cloudru"])
def test_direct_processing_does_not_borrow_openrouter_tariffs(endpoint_catalog, provider):
    _, request = endpoint_catalog
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, provider=provider, processing_mode="priority") is None
    request.assert_not_called()


def test_endpoint_tiers_keep_literal_cache_prices_and_original_precision(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[:] = [_endpoint("vendor", input_cache_read="0.0000001", input_cache_write="0.000003",
        input_cache_write_1h="0.000007", discount=0.9,
        overrides=[{"min_prompt_tokens": 2000, "prompt": "0.000003", "input_cache_write_1h": "0.000009"}])]
    cache = {"cached_tokens": 100, "cache_write_tokens": 500, "prompt_cache_ttl": "1h",
             "cache_write_tokens_by_ttl": {"5m": 200, "1h": 300}}
    assert estimate_cost_optional("vendor/tier-model", 1000, 100, cache_usage=cache,
                                  processing_mode="default") == pytest.approx(0.00431)
    assert estimate_cost_optional("vendor/tier-model", 2000, 100, cache_usage=cache,
                                  processing_mode="default") == pytest.approx(0.00831)
    schedule = get_pricing(provider="openrouter", model="vendor/tier-model")["vendor"]
    assert schedule.cache_write_1h == 7 and schedule.tiers[0][1].cache_write_1h == 9
    assert schedule.source == {"provider": "openrouter", "model": "vendor/tier-model", "endpoint_tag": "vendor",
        "service_tier": "default", "url": "https://openrouter.ai/api/v1/models/vendor/tier-model/endpoints", "status": 0}


def test_missing_exact_hour_cache_rate_does_not_use_the_legacy_multiplier(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[:] = [_endpoint("vendor", input_cache_write="0.000003")]
    assert estimate_cost_optional("vendor/tier-model", 1000, 100, processing_mode="default",
                                  cache_usage={"cache_write_tokens": 500, "prompt_cache_ttl": "1h"}) is None
    assert estimate_cost_optional("vendor/tier-model", 1000, 100, processing_mode="default",
                                  cache_usage={"cached_tokens": 1}) is None


def test_tiny_endpoint_prices_are_not_rounded_into_free_work(endpoint_catalog):
    rows, _ = endpoint_catalog
    rows[:] = [_endpoint("vendor", "0.000000000123456789", "0.000000000234567891")]
    cost = estimate_cost_optional("vendor/tier-model", 1, 1, processing_mode="default")
    assert cost == pytest.approx(0.000000000358024680, rel=1e-12, abs=0)


@pytest.mark.parametrize("value", [True, "nan", "unknown"])
def test_invalid_endpoint_rates_remain_unknown(endpoint_catalog, value):
    rows, _ = endpoint_catalog
    rows[3]["pricing"]["prompt"] = value
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="default") is None


def test_endpoint_fetch_reuses_existing_cache_owner_without_fetching_all_models(endpoint_catalog):
    _, request = endpoint_catalog
    for mode in ("priority", "flex", "default"):
        assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode=mode) is not None
    request.assert_called_once()
    import ouroboros.pricing as pricing
    assert set(pricing._cached_pricing) == {("openrouter", "vendor/tier-model")}
    assert estimate_cost_optional("vendor/other-model", 1000, 500, processing_mode="priority",
                                  allow_live_fetch=False) is None


def test_expired_endpoint_prices_are_unknown_until_the_shared_cache_refreshes(endpoint_catalog, monkeypatch):
    import ouroboros.pricing as pricing
    rows, request = endpoint_catalog
    clock = [100.0]
    monkeypatch.setattr(pricing.time, "time", lambda: clock[0])
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") is not None
    clock[0] += 21601
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority", allow_live_fetch=False) is None
    rows.clear()
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") is None
    assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="priority") is None
    assert request.call_count == 2


def test_concurrent_endpoint_readers_share_one_inflight_fetch(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from ouroboros import llm_pricing

    entered, release = Event(), Event()
    calls = []
    schedule = PricingSchedule((2.0, None, None, 8.0), source={"service_tier": "default"})
    def fetch(model, **kwargs):
        calls.append(model)
        entered.set()
        assert release.wait(2)
        return {"vendor": schedule}
    monkeypatch.setattr(llm_pricing, "fetch_openrouter_endpoint_pricing", fetch)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(estimate_cost_optional, "vendor/tier-model", 1000, 500, processing_mode="default")
        try:
            assert entered.wait(2)
            assert estimate_cost_optional("vendor/tier-model", 1000, 500, processing_mode="default") is None
        finally:
            release.set()
        assert first.result() == pytest.approx(0.006)
    assert calls == ["vendor/tier-model"]


def test_legacy_empty_mode_preserves_its_model_price_and_cache_ratio():
    with patch("ouroboros.llm.fetch_openrouter_pricing", return_value={"vendor/model": (2.0, None, 3.0, 8.0)}):
        assert estimate_cost_optional("vendor/model", 1000, 100,
            cache_usage={"cache_write_tokens": 500, "prompt_cache_ttl": "1h"}) == 0.0042


def _catalog_stub(monkeypatch, entries):
    """Seed the public tariff catalogue without network I/O (the live-stack pattern)."""
    from ouroboros import pricing_catalog
    from ouroboros.utils import utc_now_iso

    raw = json.dumps(entries)
    pricing_catalog._cache_path(None).unlink(missing_ok=True)
    monkeypatch.setattr(pricing_catalog, "_fetch_catalog", lambda: pricing_catalog._decode(dict(
        url=pricing_catalog.CATALOG_URL, content=raw, fetched_at=utc_now_iso(),
        sha256=hashlib.sha256(raw.encode()).hexdigest())))


def test_direct_text_route_prices_its_exact_catalog_entry(monkeypatch):
    _catalog_stub(monkeypatch, {"gigachat/GigaChat-3-Ultra": {
        "litellm_provider": "gigachat", "input_cost_per_token": 1.17127e-06,
        "output_cost_per_token": 3.51306e-06, "cache_read_input_token_cost": 1.17127e-06}})
    cost = estimate_cost_optional("gigachat::GigaChat-3-Ultra", 1_000_000, 500_000, provider="gigachat")
    assert cost == pytest.approx(1.17127 + 1.75653)
    assert cost.quote["catalog_id"] == "gigachat/GigaChat-3-Ultra"
    assert cost.quote["fetched_at"]


def test_direct_text_catalog_reads_cache_tokens_at_the_published_rate(monkeypatch):
    _catalog_stub(monkeypatch, {"deepseek/deepseek-v4-flash": {
        "litellm_provider": "deepseek", "input_cost_per_token": 3e-07,
        "output_cost_per_token": 1.2e-06, "cache_read_input_token_cost": 6e-09}})
    cost = estimate_cost_optional("deepseek::deepseek-v4-flash", 1_000_000, 0,
                                  cache_usage={"cached_tokens": 1_000_000}, provider="deepseek")
    assert cost == pytest.approx(0.006)


def test_direct_text_catalog_without_a_rate_keeps_the_call_unknown(monkeypatch):
    _catalog_stub(monkeypatch, {"deepseek/deepseek-v4-flash": {
        "litellm_provider": "deepseek", "input_cost_per_token": 3e-07, "output_cost_per_token": 1.2e-06}})
    # Cached tokens without a published cache-read rate are unknown, never free work.
    assert estimate_cost_optional("deepseek::deepseek-v4-flash", 1_000, 10,
                                  cache_usage={"cached_tokens": 500}, provider="deepseek") is None


def test_catalog_never_borrows_another_provider_or_model(monkeypatch):
    from ouroboros.pricing_catalog import catalog_id

    _catalog_stub(monkeypatch, {"gigachat/GigaChat-3-Ultra": {
        "litellm_provider": "gigachat", "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}})
    assert estimate_cost_optional("gigachat::GigaChat-9", 1_000, 10, provider="gigachat") is None
    assert estimate_cost_optional("deepseek::deepseek-v4-flash", 1_000, 10, provider="deepseek") is None
    assert catalog_id("GigaChat-3-Ultra", "gigachat") == "gigachat/GigaChat-3-Ultra"
    assert catalog_id("deepseek/deepseek-v4-flash", "deepseek") == "deepseek/deepseek-v4-flash"
    assert catalog_id("deepseek/GigaChat-3-Ultra", "gigachat") is None
    assert catalog_id("gigachat::GigaChat-3-Ultra", "deepseek") is None
    assert catalog_id("GigaChat-3-Ultra", "openrouter") is None


def test_a_catalog_entry_with_another_provider_label_is_refused(monkeypatch):
    _catalog_stub(monkeypatch, {"gigachat/GigaChat-3-Ultra": {
        "litellm_provider": "openai", "input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06}})
    assert estimate_cost_optional("gigachat::GigaChat-3-Ultra", 1_000, 10, provider="gigachat") is None


def test_price_source_names_the_cached_row_only(monkeypatch):
    from ouroboros.pricing import price_source

    _catalog_stub(monkeypatch, {"gigachat/GigaChat-3-Lightning": {
        "litellm_provider": "gigachat", "input_cost_per_token": 1e-07, "output_cost_per_token": 2e-07}})
    # Warm the cache the way a settled call does (one live fetch), then the display reads it.
    assert estimate_cost_optional("gigachat::GigaChat-3-Lightning", 100, 10, provider="gigachat")
    assert price_source("gigachat", model="GigaChat-3-Lightning") == "catalog"
    assert price_source("gigachat", model="GigaChat-404") == "unknown"
    assert price_source("gigachat") == "unknown"  # no model named: the row cannot be proven


def test_a_zero_row_without_a_published_source_is_unknown_not_free(monkeypatch):
    _catalog_stub(monkeypatch, {
        "gigachat/GigaChat-2-Max": {"litellm_provider": "gigachat", "input_cost_per_token": 0.0,
                                     "output_cost_per_token": 0.0},
        "gigachat/GigaChat-2-Free": {"litellm_provider": "gigachat", "input_cost_per_token": 0.0,
                                      "output_cost_per_token": 0.0,
                                      "source": "https://example.test/published-pricing"}})
    assert estimate_cost_optional("gigachat::GigaChat-2-Max", 1_000, 10, provider="gigachat") is None
    free = estimate_cost_optional("gigachat::GigaChat-2-Free", 1_000, 10, provider="gigachat")
    assert free == 0.0 and free.quote["catalog_id"] == "gigachat/GigaChat-2-Free"
