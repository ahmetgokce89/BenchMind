import os
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import api.main as api_main


class BenchMindAPITests(unittest.TestCase):
    def setUp(self):
        api_main._jobs.clear()
        self.client = TestClient(api_main.app)

    def test_health_is_public(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_token_protects_config(self):
        with patch.dict(
            os.environ,
            {"BENCHMIND_API_TOKEN": "secret", "BENCHMIND_REQUIRE_API_TOKEN": "true"},
            clear=False,
        ):
            self.assertEqual(self.client.get("/v1/config").status_code, 401)
            with patch.object(api_main, "get_api_config", return_value=("OpenAI", "key")):
                response = self.client.get(
                    "/v1/config", headers={"Authorization": "Bearer secret"}
                )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["configured"])

    def test_create_benchmark_returns_background_job(self):
        payload = {
            "companies": [
                {"name": "Alpha", "website": "https://alpha.example"},
                {"name": "Beta", "website": "https://beta.example"},
            ]
        }
        with (
            patch.dict(os.environ, {"BENCHMIND_API_TOKEN": ""}, clear=False),
            patch.object(api_main, "get_api_config", return_value=("OpenAI", "key")),
            patch.object(api_main._executor, "submit") as submit,
        ):
            response = self.client.post("/v1/benchmarks", json=payload)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "queued")
        self.assertTrue(response.json()["job_id"])
        submit.assert_called_once()

    def test_job_runner_stores_result_and_rag_stats(self):
        job_id = "job-1"
        api_main._jobs[job_id] = {"job_id": job_id, "status": "queued"}
        rag_stats = Mock()
        rag_stats.to_dict.return_value = {"status": "ready", "embeddings": 3}
        result = {"target_company": "Alpha", "companies": []}
        with (
            patch.object(api_main, "research_companies", return_value=result),
            patch.object(api_main, "save_cached_research"),
            patch.object(api_main, "build_research_index", return_value=rag_stats),
        ):
            api_main._run_benchmark_job(
                job_id,
                [{"name": "Alpha", "website": ""}],
                "OpenAI",
                "key",
                "model",
            )
        self.assertEqual(api_main._jobs[job_id]["status"], "completed")
        self.assertEqual(api_main._jobs[job_id]["result"], result)
        self.assertEqual(api_main._jobs[job_id]["rag_stats"]["embeddings"], 3)

    def test_job_exposes_research_result_before_rag_index_finishes(self):
        job_id = "job-visible-during-indexing"
        api_main._jobs[job_id] = {"job_id": job_id, "status": "queued"}
        result = {
            "target_company": "Amazon",
            "companies": [{"name": "Amazon", "website": "https://amazon.com.tr"}],
        }
        rag_stats = Mock()
        rag_stats.to_dict.return_value = {"status": "ready", "embeddings": 7}

        def inspect_indexing_state(received_result):
            self.assertIs(received_result, result)
            visible_job = api_main._jobs[job_id]
            self.assertEqual(visible_job["status"], "indexing")
            self.assertEqual(visible_job["result"], result)
            self.assertIn("research_completed_at", visible_job)
            return rag_stats

        with (
            patch.object(api_main, "research_companies", return_value=result),
            patch.object(api_main, "save_cached_research"),
            patch.object(
                api_main,
                "build_research_index",
                side_effect=inspect_indexing_state,
            ),
        ):
            api_main._run_benchmark_job(
                job_id,
                [{"name": "Amazon", "website": "https://amazon.com.tr"}],
                "Gemini",
                "key",
                "model",
            )

        self.assertEqual(api_main._jobs[job_id]["status"], "completed")

    def test_chat_stops_when_no_chunk_clears_similarity_threshold(self):
        payload = {
            "question": "What is the enterprise price?",
            "research": {"target_company": "Amazon"},
            "collection_name": "amazon-index",
        }
        with (
            patch.dict(os.environ, {"BENCHMIND_API_TOKEN": ""}, clear=False),
            patch.object(api_main, "get_api_config", return_value=("OpenAI", "key")),
            patch.object(api_main, "query_research_index", return_value=[]),
            patch.object(api_main, "answer_from_research") as answer,
        ):
            response = self.client.post("/v1/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("yeterli retrieval kanıtı yok", response.json()["answer"])
        self.assertEqual(response.json()["retrieved_sources"], [])
        answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
