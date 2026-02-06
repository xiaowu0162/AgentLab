"""Manual action agent for interactive trajectory collection.

This agent exposes a GenericAgent-like observation view in the terminal and asks
the operator to provide the next action string. The experiment loop and file
artifacts remain unchanged, so trajectories are saved in standard AgentLab format.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import bgym
from bgym import Benchmark
from browsergym.experiments.agent import Agent, AgentInfo
from PIL import Image

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS


@dataclass
class ManualActionAgentArgs(AgentArgs):
    """Args for ManualActionAgent."""

    flags: GenericPromptFlags = field(default_factory=lambda: deepcopy(BASE_FLAGS))
    state_view: Literal["axtree", "pruned_html", "both"] = "axtree"
    show_goal_each_step: bool = False
    dump_screenshot: bool = True
    screenshot_dir: str = "/tmp/agentlab_manual_previews"
    max_state_chars: int = 12_000

    def __post_init__(self):
        self.agent_name = "ManualActionAgent"

    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        """Mirror GenericAgent benchmark wiring for action space compatibility."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(benchmark.high_level_action_set_args)

        # backward compatibility fields kept in some configs
        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict

        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def make_agent(self):
        return ManualActionAgent(
            flags=self.flags,
            state_view=self.state_view,
            show_goal_each_step=self.show_goal_each_step,
            dump_screenshot=self.dump_screenshot,
            screenshot_dir=self.screenshot_dir,
            max_state_chars=self.max_state_chars,
        )


class ManualActionAgent(Agent):
    """Terminal-driven agent that asks a human/coding operator for each action."""

    def __init__(
        self,
        flags: GenericPromptFlags,
        state_view: Literal["axtree", "pruned_html", "both"] = "axtree",
        show_goal_each_step: bool = False,
        dump_screenshot: bool = True,
        screenshot_dir: str = "/tmp/agentlab_manual_previews",
        max_state_chars: int = 12_000,
    ):
        self.flags = flags
        self.state_view = state_view
        self.show_goal_each_step = show_goal_each_step
        self.dump_screenshot = dump_screenshot
        self.max_state_chars = max_state_chars

        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(self.flags.obs)

        self._step = 0
        self._screenshot_session_dir: Path | None = None
        if self.dump_screenshot:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._screenshot_session_dir = (
                Path(screenshot_dir).expanduser().resolve() / f"manual_session_{now}"
            )
            self._screenshot_session_dir.mkdir(parents=True, exist_ok=True)

    def obs_preprocessor(self, obs):
        return self._obs_preprocessor(obs)

    def _extract_goal_text(self, obs: dict) -> str:
        goal_object = obs.get("goal_object")
        if isinstance(goal_object, (list, tuple)) and goal_object:
            first = goal_object[0]
            if isinstance(first, dict):
                goal_text = first.get("text")
                if isinstance(goal_text, str) and goal_text.strip():
                    return goal_text.strip()

        goal = obs.get("goal")
        if isinstance(goal, str) and goal.strip():
            return goal.strip()
        return ""

    def _truncate(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        if self.max_state_chars <= 0:
            return text
        if len(text) <= self.max_state_chars:
            return text
        hidden = len(text) - self.max_state_chars
        return (
            text[: self.max_state_chars]
            + f"\n\n[Truncated: {hidden} chars hidden. Increase --max-state-chars to see more.]"
        )

    def _format_open_tabs(self, obs: dict) -> str:
        urls = obs.get("open_pages_urls")
        titles = obs.get("open_pages_titles")
        active_idx = obs.get("active_page_index")
        if not isinstance(urls, list) or not isinstance(titles, list):
            return ""
        lines = ["Open Tabs:"]
        for i, (title, url) in enumerate(zip(titles, urls)):
            tag = " (active)" if i == active_idx else ""
            lines.append(f"  - Tab {i}{tag}: {title} | {url}")
        return "\n".join(lines)

    def _dump_screenshot(self, obs: dict) -> Path | None:
        if not self.dump_screenshot or self._screenshot_session_dir is None:
            return None
        screenshot = obs.get("screenshot")
        if screenshot is None:
            return None

        out_path = self._screenshot_session_dir / f"step_{self._step}.png"
        try:
            Image.fromarray(screenshot).save(out_path)
        except Exception:
            return None
        return out_path

    def _print_state(self, obs: dict):
        print("\n" + "=" * 80)
        print(f"Manual Step {self._step}")

        if self._step == 0 or self.show_goal_each_step:
            goal = self._extract_goal_text(obs)
            if goal:
                print(f"\nGoal:\n{goal}")

        url = obs.get("url", "")
        print(f"\nURL: {url}")

        focused_bid = obs.get("focused_element_bid")
        print(f"Focused element bid: {focused_bid}")

        last_action_error = obs.get("last_action_error")
        if isinstance(last_action_error, str) and last_action_error.strip():
            print(f"\nLast action error:\n{last_action_error.strip()}")

        tabs_text = self._format_open_tabs(obs)
        if tabs_text:
            print(f"\n{tabs_text}")

        shot_path = self._dump_screenshot(obs)
        if shot_path is not None:
            print(f"\nScreenshot path: {shot_path}")

        if self.state_view in ("axtree", "both"):
            axtree = self._truncate(obs.get("axtree_txt", ""))
            print("\nAXTree:")
            print(axtree if axtree else "[empty]")

        if self.state_view in ("pruned_html", "both"):
            html = self._truncate(obs.get("pruned_html", ""))
            print("\nPruned HTML:")
            print(html if html else "[empty]")

        print("=" * 80)

    def _print_help(self):
        description = self.action_set.describe(with_long_description=False, with_examples=True)
        print("\nAction space:\n")
        print(description)
        print()

    def _prompt_for_action(self) -> str:
        while True:
            raw = input("Action (:help for action space)> ")
            action = raw.strip()

            if action.lower() in {":help", "help", "?"}:
                self._print_help()
                continue

            if not action:
                print("Empty action is not allowed. Please retry.")
                continue

            # Validate locally before returning to the experiment loop.
            try:
                self.action_set.to_python_code(action)
            except Exception as exc:
                print(f"Invalid action format: {exc}")
                print("Please retry.")
                continue

            return action

    def get_action(self, obs):
        self._print_state(obs)
        action = self._prompt_for_action()

        agent_info = AgentInfo(
            think=f"Manual action selected at step {self._step}",
            chat_messages=[],
            stats={"manual_step": self._step},
            extra_info={"manual_input": True, "state_view": self.state_view},
        )
        self._step += 1
        return action, agent_info


MANUAL_ACTION_AGENT = ManualActionAgentArgs()
