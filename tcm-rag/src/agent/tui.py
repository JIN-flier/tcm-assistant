"""Small Textual frontend; the backend callable returns public_result()."""

from __future__ import annotations

from collections.abc import Callable
from time import monotonic
from typing import Any

from rich.markup import escape
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Collapsible, Footer, Header, Input, LoadingIndicator, Markdown, Static

from src.agent.tui_sources import empty_sources_message, source_cards


class HealthAgentApp(App):
    TITLE = "中医知识助手"
    SUB_TITLE = "回答与文献来源"
    BINDINGS = [("ctrl+q", "quit", "退出"), ("ctrl+l", "focus_question", "输入问题")]
    CSS = """
    Screen { background: $surface; }
    #composer { height: auto; margin: 1 2 0 2; }
    #question { width: 1fr; }
    #submit { margin-left: 1; min-width: 10; }
    #status { height: auto; max-height: 4; margin: 0 2; color: $text-muted; }
    #loading { height: 1; margin: 0 2; display: none; }
    #loading.busy { display: block; }
    #results { height: 1fr; margin: 1 2; }
    #answer-pane { width: 3fr; height: 1fr; border: round $primary; padding: 0 1; }
    #sources { width: 2fr; height: 1fr; margin-left: 1; border: round $secondary; padding: 0 1; }
    #answer-scroll { height: 1fr; }
    .pane-title { height: auto; text-style: bold; margin-bottom: 1; }
    #current-question { height: auto; color: $text-muted; margin-bottom: 1; }
    #answer { margin: 0; padding: 0; }
    #sources-note { height: auto; margin-bottom: 1; color: $text-muted; }
    #source-list { height: auto; }
    Collapsible { padding: 0; margin-bottom: 1; }
    .source-meta { height: auto; color: $text-muted; margin-bottom: 1; }
    .source-text { height: auto; }
    Screen.compact #results { layout: vertical; }
    Screen.compact #answer-pane { width: 1fr; height: 3fr; }
    Screen.compact #sources { width: 1fr; height: 2fr; margin: 1 0 0 0; }
    """

    class Completed(Message):
        def __init__(self, result: dict[str, Any] | None, error: str | None = None) -> None:
            super().__init__()
            self.result = result
            self.error = error

    def __init__(self, ask: Callable[[str], dict[str, Any]]) -> None:
        super().__init__()
        self.ask = ask
        self.busy = False
        self.started_at = 0.0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="composer"):
            yield Input(placeholder="输入问题，按 Enter 提交，例如：黑芝麻有哪些日常吃法？", id="question")
            yield Button("提问", variant="primary", id="submit")
        yield Static("就绪 · 每次独立提问，沿用本地用户档案。", id="status", markup=False)
        yield LoadingIndicator(id="loading")
        with Horizontal(id="results"):
            with Vertical(id="answer-pane"):
                yield Static("最终回答", classes="pane-title")
                with VerticalScroll(id="answer-scroll"):
                    yield Static("", id="current-question", markup=False)
                    yield Markdown("输入问题后，回答会显示在这里。", id="answer")
            with VerticalScroll(id="sources"):
                yield Static("RAG 来源", classes="pane-title")
                yield Static("引用编号与回答中的 [S1]、[R1] 对应；展开条目查看片段。", id="sources-note", markup=False)
                with Vertical(id="source-list"):
                    yield Static("等待提问。", markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.screen.set_class(self.size.width < 100, "compact")
        self.query_one("#question", Input).focus()
        self.set_interval(1, self.update_elapsed)

    def on_resize(self) -> None:
        self.screen.set_class(self.size.width < 100, "compact")

    def action_focus_question(self) -> None:
        self.query_one("#question", Input).focus()

    def update_elapsed(self) -> None:
        if self.busy:
            seconds = int(monotonic() - self.started_at)
            self.query_one("#status", Static).update(f"正在分析、检索并生成回答… 已等待 {seconds} 秒")

    @on(Input.Submitted, "#question")
    @on(Button.Pressed, "#submit")
    async def submit_question(self) -> None:
        field = self.query_one("#question", Input)
        question = field.value.strip()
        if self.busy:
            return
        if not question:
            self.query_one("#status", Static).update("请先输入问题。")
            field.focus()
            return
        self.busy = True
        self.started_at = monotonic()
        field.disabled = True
        self.query_one("#submit", Button).disabled = True
        self.query_one("#loading").add_class("busy")
        self.query_one("#current-question", Static).update(f"问题：{question}")
        await self.query_one("#answer", Markdown).update("正在等待 Agent 返回结果…")
        sources = self.query_one("#source-list", Vertical)
        await sources.remove_children()
        await sources.mount(Static("正在检索来源…", markup=False))
        self.query_one("#answer-scroll", VerticalScroll).scroll_home(animate=False)
        self.query_one("#sources", VerticalScroll).scroll_home(animate=False)
        self.update_elapsed()
        self.run_question(question)

    @work(thread=True, exit_on_error=False)
    def run_question(self, question: str) -> None:
        # post_message is thread safe. All widget changes stay on the UI thread.
        try:
            result = self.ask(question)
        except Exception as error:
            self.post_message(self.Completed(None, f"{type(error).__name__}: {error}"))
        else:
            self.post_message(self.Completed(result))

    @on(Completed)
    async def handle_completed(self, message: Completed) -> None:
        try:
            await self.show_result(message)
        finally:
            self.busy = False
            self.query_one("#loading").remove_class("busy")
            field = self.query_one("#question", Input)
            field.disabled = False
            self.query_one("#submit", Button).disabled = False
            field.focus()

    async def show_result(self, message: Completed) -> None:
        sources = self.query_one("#source-list", Vertical)
        await sources.remove_children()
        if message.error is not None:
            self.query_one("#status", Static).update(f"请求失败 · {message.error}")
            await self.query_one("#answer", Markdown).update("本次请求未完成。请检查配置或服务后再次提交；输入框保留了原问题。")
            await sources.mount(Static("本次请求失败，未获取来源。", markup=False))
            return
        result = message.result or {}
        await self.query_one("#answer", Markdown).update(result.get("final_answer") or "Agent 未返回回答。")
        cards = source_cards(result)
        if cards:
            for card in cards:
                title = f"[{card.label}] {card.title}" + (" · 已引用" if card.cited else "")
                await sources.mount(Collapsible(
                    Static(card.metadata, classes="source-meta", markup=False),
                    Static(card.text, classes="source-text", markup=False),
                    title=escape(title), collapsed=not card.cited,
                ))
        else:
            await sources.mount(Static(empty_sources_message(result), markup=False))
        keywords = "、".join(result.get("keywords") or []) or "无"
        elapsed = monotonic() - self.started_at
        self.query_one("#status", Static).update(
            f"完成 · {elapsed:.1f} 秒 · LLM 调用 {result.get('llm_call_count', 0)} 次"
            f" · 来源 {len(cards)} 条 · 关键词：{keywords}"
        )
