"""Recorded-action replay agent for trajectory regeneration.

This agent replays previously saved BrowserGym high-level action strings
instead of querying a human or an LLM. It is intended for regenerating
screenshots or other environment-side artifacts from an existing trajectory.
"""

from __future__ import annotations

import gzip
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from browsergym.experiments.agent import AgentInfo

from agentlab.agents.manual_action_agent import ManualActionAgent, ManualActionAgentArgs


STEP_RE = re.compile(r"^step_(\d+)\.pkl\.gz$")


def _step_index(path: Path) -> int:
    match = STEP_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unrecognized step filename: {path.name}")
    return int(match.group(1))


def iter_step_paths(exp_dir: str | Path) -> list[Path]:
    exp_dir = Path(exp_dir).expanduser().resolve()
    step_paths = [path for path in exp_dir.glob("step_*.pkl.gz") if STEP_RE.match(path.name)]
    return sorted(step_paths, key=_step_index)


def _normalize_action(action) -> list[str]:
    if action is None:
        return []
    if isinstance(action, (list, tuple)):
        items = [str(item).strip() for item in action if item is not None]
        return [item for item in items if item]

    action_str = str(action).strip()
    if not action_str:
        return []
    return [action_str]


def extract_replay_actions_from_steps(steps: Iterable[object]) -> list[str]:
    replay_actions: list[str] = []
    for step in steps:
        replay_actions.extend(_normalize_action(getattr(step, "action", None)))
    return replay_actions


def load_replay_actions_from_exp_dir(exp_dir: str | Path) -> list[str]:
    actions: list[str] = []
    for step_path in iter_step_paths(exp_dir):
        with gzip.open(step_path, "rb") as f:
            step_info = pickle.load(f)
        actions.extend(_normalize_action(getattr(step_info, "action", None)))
    return actions


@dataclass
class ReplayActionAgentArgs(ManualActionAgentArgs):
    """Args for ReplayActionAgent."""

    actions: list[str] = field(default_factory=list)
    source_exp_dir: str | None = None
    print_state: bool = False

    def __post_init__(self):
        self.agent_name = "ReplayActionAgent"

    @classmethod
    def from_exp_dir(
        cls,
        exp_dir: str | Path,
        *,
        print_state: bool = False,
        dump_screenshot: bool = False,
        max_state_chars: int = 12_000,
    ) -> "ReplayActionAgentArgs":
        exp_dir = Path(exp_dir).expanduser().resolve()
        actions = load_replay_actions_from_exp_dir(exp_dir)
        if not actions:
            raise ValueError(f"No recorded actions found in {exp_dir}")
        return cls(
            actions=actions,
            source_exp_dir=str(exp_dir),
            print_state=print_state,
            dump_screenshot=dump_screenshot,
            max_state_chars=max_state_chars,
        )

    def make_agent(self):
        if not self.actions:
            raise ValueError("ReplayActionAgent requires a non-empty action list.")
        return ReplayActionAgent(
            flags=self.flags,
            state_view=self.state_view,
            show_goal_each_step=self.show_goal_each_step,
            dump_screenshot=self.dump_screenshot,
            screenshot_dir=self.screenshot_dir,
            max_state_chars=self.max_state_chars,
            actions=self.actions,
            source_exp_dir=self.source_exp_dir,
            print_state=self.print_state,
        )


class ReplayActionAgent(ManualActionAgent):
    """Agent that replays previously recorded action strings."""

    def __init__(
        self,
        *,
        actions: Sequence[str],
        source_exp_dir: str | None = None,
        print_state: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._replay_actions = [str(action).strip() for action in actions if str(action).strip()]
        if not self._replay_actions:
            raise ValueError("ReplayActionAgent was initialized with no actions.")
        self._source_exp_dir = source_exp_dir
        self._print_state_each_step = print_state

    def get_action(self, obs):
        if self._print_state_each_step:
            self._print_state(obs)

        if self._step >= len(self._replay_actions):
            agent_info = AgentInfo(
                think=f"Replay actions exhausted at step {self._step}",
                chat_messages=[],
                stats={"replay_step": self._step, "replay_exhausted": 1},
                extra_info={
                    "replay_input": True,
                    "source_exp_dir": self._source_exp_dir,
                    "remaining_actions": 0,
                },
            )
            return None, agent_info

        action = self._replay_actions[self._step]
        try:
            self.action_set.to_python_code(action)
        except Exception as exc:
            raise ValueError(
                f"Invalid recorded action at replay step {self._step}: {action!r}"
            ) from exc

        remaining = len(self._replay_actions) - self._step - 1
        agent_info = AgentInfo(
            think=f"Replaying recorded action at step {self._step}",
            chat_messages=[],
            stats={"replay_step": self._step, "replay_remaining_actions": remaining},
            extra_info={
                "replay_input": True,
                "source_exp_dir": self._source_exp_dir,
                "remaining_actions": remaining,
            },
        )
        self._step += 1
        return action, agent_info
