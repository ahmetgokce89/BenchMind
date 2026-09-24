import os
import inspect
import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from streamlit.testing.v1 import AppTest

from benchmark_data import (
    CRITERIA,
    benchmark_dataframe,
    calculate_confidence,
    calculate_overall_score,
    export_csv,
    findings_dataframe,
    load_demo_data,
)
from openai_service import (
    CHAT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    answer_from_research,
    clean_company_inputs,
    company_name_from_url,
    estimate_live_research_cost,
    get_api_config,
    research_companies,
    stream_answer_from_research,
    _run_claude_code,
    _parse_json_text,
)
from rag_service import (
    SourceDocument,
    cached_research_index,
    chunk_text,
    index_documents,
    query_research_index,
)
from research_cache import (
    load_cached_research,
    load_cached_research_by_key,
    list_cached_research,
    research_cache_key,
    save_cached_research,
)


class BenchmarkDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_demo_data()

    def test_demo_contains_target_and_three_competitors(self):
        self.assertEqual(self.data["target_company"], "Fathom")
        self.assertEqual(len(self.data["companies"]), 4)

    def test_default_openai_model_is_gpt_5_6_sol(self):
        self.assertEqual(DEFAULT_MODEL, "gpt-5.6-sol")

    def test_anthropic_live_research_cost_estimate_includes_searches_and_tokens(self):
        estimate = estimate_live_research_cost("Anthropic", "claude-sonnet-5", 4)
        self.assertTrue(estimate["known"])
        self.assertAlmostEqual(estimate["search_cost_usd"], 0.04)
        self.assertAlmostEqual(estimate["low_usd"], 0.10)
        self.assertAlmostEqual(estimate["high_usd"], 0.20)

    def test_unknown_model_discloses_search_cost_without_token_estimate(self):
        estimate = estimate_live_research_cost("Anthropic", "future-model", 4)
        self.assertFalse(estimate["known"])
        self.assertAlmostEqual(estimate["search_cost_usd"], 0.04)

    def test_gemini_live_research_cost_uses_free_monthly_search_allowance(self):
        estimate = estimate_live_research_cost("Gemini", "gemini-3.8-flash", 4)
        self.assertTrue(estimate["known"])
        self.assertAlmostEqual(estimate["search_cost_usd"], 0.0)
        self.assertAlmostEqual(estimate["low_usd"], 0.03)
        self.assertAlmostEqual(estimate["high_usd"], 0.0825)
        self.assertIn("5.000", estimate["search_note"])

    def test_company_name_can_be_inferred_from_url(self):
        self.assertEqual(company_name_from_url("amazon.com"), "Amazon")
        self.assertEqual(company_name_from_url("https://www.meetgeek.ai/pricing"), "Meetgeek")

    def test_company_url_overrides_a_conflicting_name(self):
        companies = clean_company_inputs(
            [
                {"name": "Fathom", "website": "https://amazon.com.tr"},
                {"name": "Otter.ai", "website": "https://otter.ai"},
            ]
        )
        self.assertEqual(companies[0]["name"], "Amazon")

    def test_domain_in_name_field_overrides_stale_demo_url(self):
        companies = clean_company_inputs(
            [
                {"name": "amazon.com.tr", "website": "https://www.fathom.ai/"},
                {"name": "Otter.ai", "website": "https://otter.ai"},
            ]
        )
        self.assertEqual(
            companies[0],
            {"name": "Amazon", "website": "https://amazon.com.tr"},
        )

    def test_editing_default_company_name_clears_its_stale_url(self):
        app = AppTest.from_file("app.py").run(timeout=30)
        self.assertFalse(app.exception)

        app.text_input(key="name_0").set_value("Amazon").run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(app.text_input(key="name_0").value, "Amazon")
        self.assertEqual(app.text_input(key="url_0").value, "")

    def test_loading_demo_resets_company_input_widgets(self):
        app = AppTest.from_file("app.py").run(timeout=30)
        self.assertFalse(app.exception)

        app.text_input(key="name_0").set_value("Amazon")
        app.text_input(key="url_0").set_value("https://amazon.com.tr").run(timeout=30)
        app.button[0].click().run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(app.text_input(key="name_0").value, "Fathom")
        self.assertEqual(app.text_input(key="url_0").value, "https://www.fathom.ai/")

    def test_product_profiles_have_labels_and_live_research_button_is_not_translated(self):
        app = AppTest.from_file("app.py").run(timeout=30)

        self.assertFalse(app.exception)
        self.assertTrue(any(button.label == "Live Research" for button in app.button))
        self.assertFalse(any("Canlı Araştır" in button.label for button in app.button))
        self.assertEqual(
            sum("**Güçlü yön**" in success.value for success in app.success),
            len(self.data["companies"]),
        )
        self.assertEqual(
            sum("**Sınırlamalar**" in warning.value for warning in app.warning),
            len(self.data["companies"]),
        )

    def test_selecting_saved_research_updates_target_inputs_and_benchmark_table(self):
        saved_research = deepcopy(self.data)
        saved_research["target_company"] = "Acme"
        saved_research["companies"][0]["name"] = "Acme"
        saved_research["companies"][0]["website"] = "https://acme.example"
        companies = [
            {
                "name": company["name"],
                "website": company.get("website", ""),
            }
            for company in saved_research["companies"]
        ]

        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            save_cached_research(companies, saved_research, cache_dir=cache_dir)
            cache_key = research_cache_key(companies)
            acme_path = cache_dir / f"{cache_key}.json"
            acme_payload = json.loads(acme_path.read_text(encoding="utf-8"))
            acme_payload["saved_at"] = "2026-09-04T12:00:00+00:00"
            acme_path.write_text(json.dumps(acme_payload), encoding="utf-8")
            for day in range(1, 4):
                older_research = deepcopy(self.data)
                older_research["target_company"] = f"Older {day}"
                older_research["companies"][0]["name"] = f"Older {day}"
                older_companies = [
                    {
                        "name": company["name"],
                        "website": company.get("website", ""),
                    }
                    for company in older_research["companies"]
                ]
                save_cached_research(
                    older_companies, older_research, cache_dir=cache_dir
                )
                older_path = cache_dir / f"{research_cache_key(older_companies)}.json"
                older_payload = json.loads(older_path.read_text(encoding="utf-8"))
                older_payload["saved_at"] = f"2026-09-{day:02d}T12:00:00+00:00"
                older_path.write_text(json.dumps(older_payload), encoding="utf-8")
            with (
                patch("research_cache.DEFAULT_CACHE_DIR", cache_dir),
                patch("rag_service.build_research_index") as build_index,
            ):
                build_index.return_value = SimpleNamespace(
                    to_dict=lambda: {
                        "collection_name": "test",
                        "documents": 0,
                        "chunks": 0,
                        "embeddings": 0,
                        "failed_urls": 0,
                        "status": "ready",
                    }
                )
                app = AppTest.from_file("app.py").run(timeout=30)
                next(
                    button
                    for button in app.button
                    if "Kayıtlı Sonucu Getir" in button.label
                ).click().run(timeout=30)
                dialog_options = app.radio(
                    key="saved_research_dialog_key"
                ).options
                self.assertEqual(len(dialog_options), 3)
                self.assertTrue(
                    any("Acme" in option for option in dialog_options)
                )
                self.assertFalse(any("Older 1" in option for option in dialog_options))
                app.radio(key="saved_research_dialog_key").set_value(cache_key).run(
                    timeout=30
                )
                next(
                    button
                    for button in app.button
                    if "Seçili Kaydı Yükle" in button.label
                ).click().run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["research"]["target_company"], "Acme")
        self.assertEqual(app.text_input(key="name_0").value, "Acme")
        active_messages = [
            info.value for info in app.info if "Aktif benchmark" in info.value
        ]
        self.assertTrue(any("**Acme**" in message for message in active_messages))
        benchmark_tables = [
            markdown.value
            for markdown in app.markdown
            if '<table class="benchmark-table">' in markdown.value
        ]
        self.assertEqual(len(benchmark_tables), 1)
        self.assertIn("Acme", benchmark_tables[0])
        self.assertNotIn("Fathom", benchmark_tables[0])

    def test_live_research_requires_cost_confirmation(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            app = AppTest.from_file("app.py").run(timeout=30)
            live_button = next(
                button for button in app.button if "Research" in button.label
            )
            live_button.click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any(button.label == "Onayla ve araştır" for button in app.button))
        self.assertTrue(any(button.label == "Vazgeç" for button in app.button))
        self.assertTrue(
            any(
                "Tahmini Live Research maliyeti" in info.value
                for info in app.info
            )
        )

    def test_rag_chunks_and_persists_real_vectors(self):
        import chromadb

        document = SourceDocument(
            document_id="D0001",
            source_id="S01",
            title="Test source",
            url="https://example.com/source",
            publisher="Example",
            source_type="Official documentation",
            text=("Pricing and product documentation. " * 120).strip(),
        )
        chunks = chunk_text(document.text, chunk_size=400, overlap=80)
        self.assertGreater(len(chunks), 1)

        client = chromadb.Client()
        stats = index_documents(
            self.data,
            [document],
            client=client,
            encoder=lambda texts: [[float(len(text)), 1.0] for text in texts],
        )
        self.assertEqual(stats.status, "ready")
        self.assertEqual(stats.documents, 1)
        self.assertGreater(stats.chunks, 1)
        self.assertEqual(stats.embeddings, stats.chunks)
        self.assertEqual(client.get_collection(stats.collection_name).count(), stats.chunks)
        cached = cached_research_index(self.data, client=client)
        self.assertIsNotNone(cached)
        self.assertTrue(cached.cache_hit)
        self.assertEqual(cached.embeddings, stats.embeddings)

    def test_scores_are_complete_and_bounded(self):
        for company in self.data["companies"]:
            self.assertEqual(set(company["scores"]), set(CRITERIA))
            for score in company["scores"].values():
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 10)
            self.assertEqual(company["overall_score"], calculate_overall_score(company["scores"]))

    def test_confidence_is_derived_from_the_evidence_package(self):
        confidence = calculate_confidence(self.data)
        self.assertGreaterEqual(confidence["score"], 0)
        self.assertLessEqual(confidence["score"], 100)
        self.assertIn(confidence["level"], {"Düşük", "Orta", "Yüksek"})
        self.assertIn("Kaynak güncelliği henüz puana dahil değil", confidence["explanation"])
        self.assertEqual(self.data["confidence"], confidence)

    def test_confidence_drops_when_evidence_is_missing_or_conflicted(self):
        weak_data = {
            "companies": [{"name": "Acme"}],
            "sources": [{"id": "S1"}],
            "findings": [
                {
                    "company": "Acme",
                    "evidence_type": "AI inference",
                    "source_ids": [],
                }
            ],
            "conflicts": [{"topic": "Pricing"}, {"topic": "Limits"}],
        }
        confidence = calculate_confidence(weak_data)
        self.assertEqual(confidence["level"], "Düşük")
        self.assertLess(confidence["score"], 50)

    def test_confidence_is_zero_without_findings(self):
        confidence = calculate_confidence({"findings": [], "sources": []})
        self.assertEqual(confidence["score"], 0)
        self.assertEqual(confidence["level"], "Düşük")

    def test_findings_only_reference_known_sources(self):
        source_ids = {source["id"] for source in self.data["sources"]}
        for finding in self.data["findings"]:
            self.assertTrue(set(finding["source_ids"]).issubset(source_ids))

    def test_frames_and_csv_are_exportable(self):
        benchmark = benchmark_dataframe(self.data)
        self.assertEqual(len(benchmark), 4)
        self.assertIn("Şirket", benchmark.columns)
        self.assertIn("Genel puan", benchmark.columns)
        self.assertEqual(len(findings_dataframe(self.data)), len(self.data["findings"]))
        csv_data = export_csv(self.data)
        self.assertTrue(csv_data.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"Fathom", csv_data)

    def test_findings_are_grouped_in_benchmark_company_order(self):
        frame = findings_dataframe(self.data)
        companies = frame["Şirket"].tolist()
        self.assertEqual(
            companies,
            [
                "Fathom", "Fathom", "Fathom", "Fathom",
                "Fireflies.ai", "Fireflies.ai", "Fireflies.ai",
                "Otter.ai", "Otter.ai",
                "MeetGeek", "MeetGeek",
            ],
        )
        fathom_rows = frame[frame["Şirket"] == "Fathom"]
        self.assertEqual(fathom_rows.iloc[-1]["Kategori"], "Strateji")

    def test_live_research_uses_web_search_and_structured_output(self):
        fake_client = Mock()
        fake_client.responses.create.return_value = SimpleNamespace(
            output_text=__import__("json").dumps(self.data)
        )
        companies = [
            {"name": "Fathom", "website": "https://www.fathom.ai/"},
            {"name": "Otter.ai", "website": "https://otter.ai/"},
        ]
        with patch("openai_service.OpenAI", return_value=fake_client):
            result = research_companies(companies, api_key="test-key")
        self.assertEqual(result["target_company"], "Fathom")
        call = fake_client.responses.create.call_args.kwargs
        self.assertEqual(call["model"], "gpt-5.6-sol")
        self.assertEqual(call["max_output_tokens"], 10000)
        self.assertEqual(call["reasoning"], {"effort": "none"})
        self.assertIn("doğal ve profesyonel Türkçe", call["instructions"])
        self.assertEqual(call["tools"], [{"type": "web_search"}])
        self.assertEqual(call["max_tool_calls"], 4)
        self.assertEqual(call["text"]["format"]["type"], "json_schema")
        self.assertFalse(call["store"])
        self.assertIn("sektörünü ve iş modelini önce kaynaklardan belirle", call["input"])
        self.assertIn("Bu alanları boş bırakma", call["input"])

    def test_live_research_rejects_a_benchmark_for_the_wrong_target(self):
        fake_client = Mock()
        fake_client.responses.create.return_value = SimpleNamespace(
            output_text=__import__("json").dumps(self.data)
        )
        companies = [
            {"name": "Amazon", "website": "https://amazon.com.tr"},
            {"name": "Fireflies.ai", "website": "https://fireflies.ai"},
        ]
        with (
            patch("openai_service.OpenAI", return_value=fake_client),
            self.assertRaisesRegex(RuntimeError, "Amazon.*bulunamadı"),
        ):
            research_companies(companies, api_key="test-key")

    def test_anthropic_live_research_uses_bounded_web_search(self):
        fake_client = Mock()
        fake_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=__import__("json").dumps(self.data))]
        )
        companies = [
            {"name": "Fathom", "website": "https://www.fathom.ai/"},
            {"name": "Otter.ai", "website": "https://otter.ai/"},
        ]
        with patch("openai_service.Anthropic", return_value=fake_client):
            result = research_companies(
                companies,
                api_key="test-key",
                model="claude-sonnet-5",
                provider="Anthropic",
            )
        self.assertEqual(result["target_company"], "Fathom")
        call = fake_client.messages.create.call_args.kwargs
        self.assertEqual(call["model"], "claude-sonnet-5")
        self.assertEqual(call["max_tokens"], 10000)
        self.assertEqual(call["tools"][0]["type"], "web_search_20260318")
        self.assertEqual(call["tools"][0]["max_uses"], 4)
        self.assertEqual(call["tools"][0]["response_inclusion"], "excluded")
        self.assertEqual(call["output_config"], {"effort": "low"})
        self.assertIn("Yalnızca geçerli bir JSON object döndür", call["messages"][0]["content"])

    def test_anthropic_live_research_continues_pause_turn(self):
        fake_client = Mock()
        paused_content = [SimpleNamespace(type="server_tool_use")]
        fake_client.messages.create.side_effect = [
            SimpleNamespace(content=paused_content, stop_reason="pause_turn"),
            SimpleNamespace(
                content=[SimpleNamespace(type="text", text=__import__("json").dumps(self.data))],
                stop_reason="end_turn",
            ),
        ]
        companies = [
            {"name": "Fathom", "website": "https://www.fathom.ai/"},
            {"name": "Otter.ai", "website": "https://otter.ai/"},
        ]
        with patch("openai_service.Anthropic", return_value=fake_client):
            result = research_companies(
                companies,
                api_key="test-key",
                model="claude-sonnet-5",
                provider="Anthropic",
            )
        self.assertEqual(result["target_company"], "Fathom")
        self.assertEqual(fake_client.messages.create.call_count, 2)
        continuation = fake_client.messages.create.call_args_list[1].kwargs
        self.assertEqual(continuation["messages"][-1], {"role": "assistant", "content": paused_content})
        self.assertEqual(continuation["tools"][0]["max_uses"], 4)

    def test_gemini_live_research_uses_search_and_structured_output(self):
        fake_client = Mock()
        grounding = SimpleNamespace(
            grounding_chunks=[
                SimpleNamespace(web=SimpleNamespace(title="Fathom", uri="https://fathom.ai"))
            ],
            web_search_queries=["Fathom pricing", "Otter pricing"],
        )
        fake_client.models.generate_content.side_effect = [
            SimpleNamespace(
                text="Grounded research notes",
                candidates=[SimpleNamespace(grounding_metadata=grounding)],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100,
                    candidates_token_count=200,
                    thoughts_token_count=50,
                    total_token_count=350,
                ),
            ),
            SimpleNamespace(
                text=__import__("json").dumps(self.data),
                candidates=[],
                usage_metadata=SimpleNamespace(
                    prompt_token_count=500,
                    candidates_token_count=1000,
                    thoughts_token_count=100,
                    total_token_count=1600,
                ),
            ),
        ]
        companies = [
            {"name": "Fathom", "website": "https://www.fathom.ai/"},
            {"name": "Otter.ai", "website": "https://otter.ai/"},
        ]
        with patch("openai_service.genai.Client", return_value=fake_client) as client:
            result = research_companies(
                companies,
                api_key="gemini-test-key",
                model="gemini-3.5-flash-lite",
                provider="Gemini",
            )
        self.assertEqual(result["target_company"], "Fathom")
        self.assertEqual(result["_usage"]["search_queries"], 2)
        self.assertEqual(result["_usage"]["total_tokens"], 1950)
        self.assertAlmostEqual(result["_usage"]["token_cost_usd"], 0.003555)
        self.assertAlmostEqual(
            result["_usage"]["cost_after_free_search_quota_usd"], 0.031555
        )
        client.assert_called_once_with(api_key="gemini-test-key")
        self.assertEqual(fake_client.models.generate_content.call_count, 2)
        search_call, structure_call = fake_client.models.generate_content.call_args_list
        self.assertEqual(search_call.kwargs["model"], "gemini-3.5-flash-lite")
        self.assertEqual(search_call.kwargs["config"].max_output_tokens, 6000)
        self.assertIsNotNone(search_call.kwargs["config"].tools[0].google_search)
        self.assertEqual(structure_call.kwargs["config"].max_output_tokens, 10000)
        self.assertEqual(structure_call.kwargs["config"].response_mime_type, "application/json")
        self.assertIsNone(structure_call.kwargs["config"].response_json_schema)
        self.assertIn(
            "doğal ve profesyonel Türkçe",
            structure_call.kwargs["config"].system_instruction,
        )
        self.assertIsNone(structure_call.kwargs["config"].tools)
        self.assertIn("https://fathom.ai", structure_call.kwargs["contents"])
        self.assertIn("companies", structure_call.kwargs["contents"])

    def test_json_parser_accepts_a_wrapped_object(self):
        self.assertEqual(_parse_json_text('Sonuç:\n```json\n{"ok": true}\n```'), {"ok": True})

    def test_anthropic_key_is_selected_without_exposing_it(self):
        clean_environment = {
            "ANTHROPIC_API_KEY": "anthropic-test-key",
            "OPENAI_API_KEY": "openai-test-key",
            "LLM_PROVIDER": "anthropic",
        }
        with patch.dict(os.environ, clean_environment, clear=True):
            provider, api_key = get_api_config()
        self.assertEqual(provider, "Anthropic")
        self.assertEqual(api_key, "anthropic-test-key")

    def test_gemini_key_and_preference_are_selected(self):
        environment = {
            "GEMINI_API_KEY": "gemini-test-key",
            "ANTHROPIC_API_KEY": "anthropic-test-key",
            "LLM_PROVIDER": "gemini",
        }
        with patch.dict(os.environ, environment, clear=True):
            provider, api_key = get_api_config()
        self.assertEqual((provider, api_key), ("Gemini", "gemini-test-key"))

    def test_anthropic_is_the_default_when_claude_code_is_also_installed(self):
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-test-key"}, clear=True),
            patch("openai_service.shutil.which", return_value=r"C:\bin\claude.exe"),
        ):
            provider, api_key = get_api_config()
        self.assertEqual((provider, api_key), ("Anthropic", "anthropic-test-key"))

    def test_claude_code_is_selected_and_runs_with_budgets(self):
        environment = {
            "ANTHROPIC_API_KEY": "anthropic-test-key",
            "LLM_PROVIDER": "claude_code",
            "CLAUDE_CODE_RESEARCH_BUDGET_USD": "0.75",
        }
        completed = SimpleNamespace(
            returncode=0,
            stdout=__import__("json").dumps(
                {"is_error": False, "structured_output": {"ok": True}}
            ),
            stderr="",
        )
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("openai_service.shutil.which", return_value=r"C:\bin\claude.exe"),
            patch("openai_service.subprocess.run", return_value=completed) as run,
        ):
            provider, api_key = get_api_config()
            output = _run_claude_code(
                "System",
                "Prompt",
                "claude-sonnet-5",
                schema=schema,
                web_search=True,
            )
        self.assertEqual((provider, api_key), ("Claude Code", "anthropic-test-key"))
        self.assertEqual(__import__("json").loads(output), {"ok": True})
        command = run.call_args.args[0]
        self.assertIn("WebSearch", command)
        self.assertIn("--json-schema", command)
        self.assertEqual(command[command.index("--max-budget-usd") + 1], "0.75")
        self.assertIn("--no-session-persistence", command)

    def test_grounded_chat_does_not_enable_web_search(self):
        fake_client = Mock()
        fake_client.responses.create.return_value = SimpleNamespace(output_text="Fathom leads on value. [S01]")
        with patch("openai_service.OpenAI", return_value=fake_client):
            answer = answer_from_research("Who leads on value?", self.data, api_key="test-key")
        self.assertIn("[S01]", answer)
        call = fake_client.responses.create.call_args.kwargs
        self.assertEqual(call["model"], "gpt-5.6-sol")
        self.assertEqual(call["reasoning"], {"effort": "none"})
        self.assertEqual(call["max_output_tokens"], 500)
        self.assertIn("Yanıtı kısa, karar odaklı ve doğal Türkçe yaz", call["instructions"])
        self.assertNotIn("tools", call)
        self.assertFalse(call["store"])

    def test_anthropic_grounded_chat_does_not_enable_web_search(self):
        fake_client = Mock()
        fake_client.messages.create.return_value = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Fathom leads on value. [S01]")]
        )
        with patch("openai_service.Anthropic", return_value=fake_client):
            answer = answer_from_research(
                "Who leads on value?",
                self.data,
                api_key="test-key",
                model="claude-sonnet-5",
                provider="Anthropic",
            )
        self.assertIn("[S01]", answer)
        call = fake_client.messages.create.call_args.kwargs
        self.assertNotIn("tools", call)
        self.assertEqual(call["output_config"], {"effort": "low"})
        self.assertEqual(call["max_tokens"], 500)

    def test_anthropic_grounded_chat_streams_with_500_token_limit(self):
        fake_client = Mock()
        stream_context = MagicMock()
        stream_context.__enter__.return_value.text_stream = iter(["Fathom ", "önde. [S01]"])
        fake_client.messages.stream.return_value = stream_context
        with patch("openai_service.Anthropic", return_value=fake_client):
            chunks = list(
                stream_answer_from_research(
                    "Kim önde?",
                    self.data,
                    api_key="test-key",
                    model="claude-sonnet-5",
                    provider="Anthropic",
                )
            )
        self.assertEqual("".join(chunks), "Fathom önde. [S01]")
        call = fake_client.messages.stream.call_args.kwargs
        self.assertEqual(call["max_tokens"], CHAT_MAX_OUTPUT_TOKENS)
        self.assertEqual(call["output_config"], {"effort": "low"})

    def test_rag_query_defaults_to_five_chunks(self):
        parameter = inspect.signature(query_research_index).parameters["n_results"]
        self.assertEqual(parameter.default, 5)
        similarity = inspect.signature(query_research_index).parameters["min_similarity"]
        self.assertEqual(similarity.default, 0.40)

    def test_rag_query_filters_chunks_below_similarity_threshold(self):
        collection = Mock()
        collection.count.return_value = 3
        collection.query.return_value = {
            "ids": [["strong", "boundary", "weak"]],
            "documents": [["Strong evidence", "Boundary evidence", "Weak evidence"]],
            "metadatas": [[
                {"source_id": "S1"},
                {"source_id": "S2"},
                {"source_id": "S3"},
            ]],
            "distances": [[0.20, 0.45, 0.80]],
        }
        client = Mock()
        client.get_collection.return_value = collection
        with (
            patch("rag_service.get_chroma_client", return_value=client),
            patch("rag_service._encode", return_value=[[1.0, 0.0]]),
        ):
            rows = query_research_index("collection", "pricing", min_similarity=0.55)
        self.assertEqual([row["chunk_id"] for row in rows], ["strong", "boundary"])
        self.assertEqual([row["similarity"] for row in rows], [0.8, 0.55])

    def test_rag_query_rejects_invalid_similarity_threshold(self):
        with self.assertRaises(ValueError):
            query_research_index("collection", "pricing", min_similarity=1.1)

    def test_live_research_cache_round_trip(self):
        companies = [
            {"name": "Amazon", "website": "https://www.amazon.com.tr/"},
            {"name": "Trendyol", "website": "https://trendyol.com/"},
        ]
        with TemporaryDirectory() as directory:
            saved_at = save_cached_research(
                companies,
                self.data,
                cache_dir=Path(directory),
            )
            cached = load_cached_research(companies, cache_dir=Path(directory))
        self.assertIsNotNone(cached)
        cached_research, cached_at = cached
        self.assertEqual(cached_at, saved_at)
        self.assertEqual(cached_research["target_company"], self.data["target_company"])

    def test_live_research_cache_can_be_restored_by_safe_url_key(self):
        companies = [
            {"name": "Amazon", "website": "https://www.amazon.com.tr/"},
            {"name": "Trendyol", "website": "https://trendyol.com/"},
        ]
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            save_cached_research(companies, self.data, cache_dir=cache_dir)
            cached = load_cached_research_by_key(
                research_cache_key(companies), cache_dir=cache_dir
            )
        self.assertIsNotNone(cached)
        self.assertEqual(cached[0]["target_company"], self.data["target_company"])

    def test_cache_key_loader_rejects_path_traversal(self):
        with TemporaryDirectory() as directory:
            cached = load_cached_research_by_key(
                "../../data/demo_benchmark", cache_dir=Path(directory)
            )
        self.assertIsNone(cached)

    def test_saved_research_list_contains_valid_entries_and_ignores_bad_files(self):
        companies = [
            {"name": "Amazon", "website": "https://amazon.com.tr"},
            {"name": "Trendyol", "website": "https://trendyol.com"},
        ]
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            save_cached_research(companies, self.data, cache_dir=cache_dir)
            (cache_dir / ("f" * 64 + ".json")).write_text("not-json", encoding="utf-8")
            (cache_dir / "not-a-cache.json").write_text("{}", encoding="utf-8")
            entries = list_cached_research(cache_dir=cache_dir)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["key"], research_cache_key(companies))
        self.assertEqual(entries[0]["target_company"], self.data["target_company"])
        self.assertEqual(entries[0]["companies"][0], "Fathom")

    def test_saved_research_list_can_be_limited_to_three_newest_entries(self):
        with TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            for index in range(4):
                companies = [
                    {"name": f"Target {index}", "website": f"https://target-{index}.example"},
                    {"name": "Rival", "website": "https://rival.example"},
                ]
                save_cached_research(companies, self.data, cache_dir=cache_dir)
                path = cache_dir / f"{research_cache_key(companies)}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["saved_at"] = f"2026-09-{index + 1:02d}T12:00:00+00:00"
                path.write_text(json.dumps(payload), encoding="utf-8")

            entries = list_cached_research(cache_dir=cache_dir, limit=3)

        self.assertEqual(len(entries), 3)
        self.assertEqual(
            [entry["saved_at"] for entry in entries],
            [
                "2026-09-04T12:00:00+00:00",
                "2026-09-03T12:00:00+00:00",
                "2026-09-02T12:00:00+00:00",
            ],
        )

    def test_saved_research_list_rejects_a_negative_limit(self):
        with self.assertRaises(ValueError):
            list_cached_research(limit=-1)

    def test_live_research_cache_normalizes_equivalent_urls(self):
        first = [{"name": " Amazon ", "website": "https://www.amazon.com.tr/"}]
        second = [{"name": "amazon", "website": "amazon.com.tr"}]
        self.assertEqual(research_cache_key(first), research_cache_key(second))


if __name__ == "__main__":
    unittest.main()
