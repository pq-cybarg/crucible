"""The fullscreen TUI mounts + composes and degrades gracefully when the backend is unreachable.
(Interaction against a live backend is covered by the agent-session endpoint tests; this guards the
widget tree + offline resilience without needing a server.)"""
import asyncio


def test_tui_mounts_and_survives_offline_backend():
    from crucible.tui import CrucibleTUI

    async def go() -> None:
        app = CrucibleTUI(control="http://127.0.0.1:59997")   # nothing listening → offline path
        async with app.run_test() as pilot:
            await pilot.pause(0.2)
            assert app.query_one("#tabs") is not None
            assert app.query_one("#context") is not None
            assert app.query_one("#browser") is not None
            assert app.query_one("#composer") is not None
            # no sessions surfaced (backend offline) but the app is alive and interactive
            assert app._sessions == []

    asyncio.run(go())
