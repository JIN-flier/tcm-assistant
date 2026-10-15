"""Run with python -m unittest discover -s tests -v; no API/DB calls."""

import importlib.util
import threading
import unittest

from src.agent.tui_sources import empty_sources_message, source_cards


def sample_result():
    return {
        "final_answer": "参考 [S2] 和 [R3]。",
        "keywords": ["黑芝麻"],
        "llm_call_count": 3,
        "rag": {
            "status": "completed",
            "ranked": [{"chunk_id": "omitted"}, {"chunk_id": "hit"}],
            "context_sources": [
                {"chunk_id": "hit", "title": "本草", "text_kind": "original", "text": "检索原文", "source_file": "raw/book.txt", "line_start": 2, "line_end": 4},
                {"chunk_id": "neighbor", "title": "本草", "node_title": "相邻章节", "text_kind": "modern_translation", "text": "相邻译文", "channels": ["context"]},
            ],
        },
        "original_source_read": {"excerpts": [
            {"source_label": "R3", "title": "本草", "status": "read", "source_file": "raw/book.txt", "start": 0, "end": 12, "encoding": "utf-8", "text": "原始文件内容"},
        ]},
    }


class SourceTests(unittest.TestCase):
    def test_citations_use_context_order_including_neighbors(self):
        cards = source_cards(sample_result())
        self.assertEqual([card.label for card in cards], ["S1", "S2", "R3"])
        self.assertEqual([card.text for card in cards], ["检索原文", "相邻译文", "原始文件内容"])
        self.assertEqual([card.cited for card in cards], [False, True, True])
        self.assertIn("行号：2–4", cards[0].metadata)
        self.assertIn("现代译文", cards[1].metadata)
        self.assertIn("相邻上下文", cards[1].metadata)

    def test_failed_original_read_keeps_its_label_and_reason(self):
        result = sample_result()
        result["original_source_read"]["excerpts"][0].update(status="not_read", error="file missing", text=None)
        card = source_cards(result)[-1]
        self.assertEqual(card.label, "R3")
        self.assertIn("读取失败", card.metadata)
        self.assertEqual(card.text, "file missing")

    def test_ranked_candidates_are_never_assigned_s_labels(self):
        result = sample_result()
        result["rag"]["context_sources"] = []
        result["original_source_read"] = None
        self.assertEqual(source_cards(result), [])
        self.assertIn("没有片段进入回答上下文", empty_sources_message(result))

    def test_skipped_and_no_hits_have_distinct_messages(self):
        self.assertEqual(source_cards({"rag": None}), [])
        self.assertIn("未执行", empty_sources_message({"rag": {"status": "skipped"}}))
        self.assertIn("未找到", empty_sources_message({"rag": {"status": "completed"}}))

    def test_long_excerpt_is_marked_and_literal_text_is_preserved(self):
        result = sample_result()
        result["rag"]["context_sources"][0]["text"] = "[bold]原文[/bold]…"
        card = source_cards(result)[0]
        self.assertTrue(card.text.startswith("[bold]原文[/bold]…"))
        self.assertIn("截断", card.text)


HAS_TEXTUAL = importlib.util.find_spec("textual") is not None
if HAS_TEXTUAL:
    from textual.widgets import Button, Collapsible, Input
    from src.agent.tui import HealthAgentApp


@unittest.skipUnless(HAS_TEXTUAL, "Textual 未安装；安装 requirements.txt 后可运行界面测试")
class InterfaceTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_is_nonblocking_and_prevents_duplicate_requests(self):
        release = threading.Event()
        calls = []

        def ask(question):
            calls.append(question)
            release.wait(timeout=5)
            return sample_result()

        app = HealthAgentApp(ask)
        async with app.run_test(size=(120, 40)) as pilot:
            field = app.query_one("#question", Input)
            field.value = "黑芝麻"
            try:
                await pilot.press("enter")
                self.assertTrue(app.busy)
                self.assertTrue(field.disabled)
                await app.submit_question()
                self.assertTrue(app.query_one("#submit", Button).disabled)
            finally:
                release.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            self.assertEqual(calls, ["黑芝麻"])
            self.assertFalse(app.busy)
            self.assertFalse(field.disabled)
            cards = list(app.query(Collapsible))
            self.assertEqual(len(cards), 3)
            self.assertEqual([card.collapsed for card in cards], [True, False, False])
            self.assertFalse(app.screen.has_class("compact"))
            await pilot.resize_terminal(80, 30)
            self.assertTrue(app.screen.has_class("compact"))

    async def test_error_reenables_input_for_retry_and_clears_old_sources(self):
        calls = []

        def ask(question):
            calls.append(question)
            if len(calls) == 2:
                raise RuntimeError("测试服务不可用")
            return sample_result()

        app = HealthAgentApp(ask)
        async with app.run_test() as pilot:
            field = app.query_one("#question", Input)
            for expected in (3, 0, 3):
                field.value = "测试问题"
                await pilot.press("enter")
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertFalse(field.disabled)
                self.assertFalse(app.busy)
                self.assertEqual(field.value, "测试问题")
                self.assertEqual(len(app.query(Collapsible)), expected)
            self.assertEqual(len(calls), 3)

    async def test_blank_input_never_calls_backend(self):
        def unexpected(question):
            self.fail("空问题不应调用后端")

        app = HealthAgentApp(unexpected)
        async with app.run_test() as pilot:
            app.query_one("#question", Input).value = "  "
            await pilot.press("enter")
            self.assertFalse(app.busy)
            self.assertFalse(app.query_one("#submit", Button).disabled)


if __name__ == "__main__":
    unittest.main()
