from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return url


def _seed_legacy_webarena_env_vars() -> None:
    compat_map = {
        "SHOPPING": os.environ.get("WA_SHOPPING"),
        "SHOPPING_ADMIN": os.environ.get("WA_SHOPPING_ADMIN"),
        "REDDIT": _origin(os.environ["WA_REDDIT"]) if os.environ.get("WA_REDDIT") else None,
        "GITLAB": _origin(os.environ["WA_GITLAB"]) if os.environ.get("WA_GITLAB") else None,
        "WIKIPEDIA": os.environ.get("WA_WIKIPEDIA"),
        "MAP": os.environ.get("WA_MAP"),
        "HOMEPAGE": os.environ.get("WA_HOMEPAGE"),
    }
    for legacy_key, compat_value in compat_map.items():
        if not os.environ.get(legacy_key) and compat_value:
            os.environ[legacy_key] = compat_value


def install_webarena_html_evaluator_patch() -> bool:
    """Patch WebArena HTML evaluation to be less brittle on reddit mutations.

    This keeps the fix in the editable AgentLab repo instead of relying on
    direct site-packages edits. The patch is intentionally narrow:
    - keep longer reddit-specific waits when evaluator opens a new page
    - reuse the current page when it already matches the target URL
    - retry once after waiting if the selected element is still empty
    """

    _seed_legacy_webarena_env_vars()

    try:
        import webarena.evaluation_harness.evaluators as eval_mod
    except Exception:
        return False

    if getattr(eval_mod, "_agentlab_html_eval_patch_installed", False):
        return True

    def _patched_call(self, trajectory, config_file: Path | str, page, client=None) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        targets = configs["eval"]["program_html"]
        is_reddit_task = "reddit" in configs.get("sites", [])
        eval_page_delay = float(os.environ.get("WEBARENA_EVAL_PAGE_DELAY", "3.0"))
        reddit_eval_page_delay = float(
            os.environ.get("WEBARENA_REDDIT_EVAL_PAGE_DELAY", "6.0")
        )
        target_page_delay = reddit_eval_page_delay if is_reddit_task else eval_page_delay

        def _clean_url(url: str) -> str:
            return str(url).rstrip("/")

        def _wait_for_target_page(target_page) -> None:
            try:
                target_page.wait_for_load_state(
                    "networkidle", timeout=int(target_page_delay * 1000)
                )
            except Exception:
                pass
            time.sleep(target_page_delay)

        score = 1.0
        for target in targets:
            target_url: str = target["url"]
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", page.url)
                target_url = eval(func)

            locator: str = target["locator"]

            prev_page = None
            if target_url != "last":
                if _clean_url(target_url) != _clean_url(page.url):
                    prev_page = page
                    page = page.context.new_page()
                    page.goto(target_url, wait_until="load")
                    _wait_for_target_page(page)

            if not locator.strip():
                selected_element = page.content()
            elif locator.startswith("document.") or locator.startswith("[...document."):
                if "prep_actions" in target:
                    try:
                        for prep_action in target["prep_actions"]:
                            page.evaluate(f"() => {prep_action}")
                    except Exception:
                        pass
                try:
                    selected_element = str(page.evaluate(f"() => {locator}"))
                    if not selected_element:
                        selected_element = ""
                except Exception:
                    selected_element = ""

                if not selected_element and target_url != "last":
                    _wait_for_target_page(page)
                    try:
                        selected_element = str(page.evaluate(f"() => {locator}"))
                        if not selected_element:
                            selected_element = ""
                    except Exception:
                        selected_element = ""
            elif locator.startswith("func:"):
                func = locator.split("func:")[1]
                func = func.replace("__page__", "page")
                selected_element = eval(func)
            else:
                raise ValueError(f"Unknown locator: {locator}")

            selected_element = html.unescape(selected_element)

            if "exact_match" in target["required_contents"]:
                required_contents = target["required_contents"]["exact_match"]
                cur_score = eval_mod.StringEvaluator.exact_match(
                    ref=required_contents, pred=selected_element
                )
                score *= float(cur_score)
            elif "must_include" in target["required_contents"]:
                required_contents = target["required_contents"]["must_include"]
                assert isinstance(required_contents, list)
                for content in required_contents:
                    content_or = content.split(" |OR| ")
                    cur_score = any(
                        [
                            eval_mod.StringEvaluator.must_include(
                                ref=content,
                                pred=selected_element,
                                tokenize=False,
                            )
                            for content in content_or
                        ]
                    )
                    score *= float(cur_score)
            else:
                raise ValueError(
                    f"Unknown required_contents: {target['required_contents'].keys()}"
                )

            if prev_page:
                page.close()
                page = prev_page
                prev_page = None

        return score

    eval_mod.HTMLContentEvaluator.__call__ = _patched_call
    eval_mod._agentlab_html_eval_patch_installed = True
    return True
