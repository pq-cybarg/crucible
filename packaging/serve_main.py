"""Entry for a self-contained Crucible server (backend + bundled GUI)."""
import os

import uvicorn


def main() -> None:
    uvicorn.run("crucible.app:app",
                host=os.environ.get("CRUCIBLE_HOST", "127.0.0.1"),
                port=int(os.environ.get("CRUCIBLE_PORT", "8400")))


if __name__ == "__main__":
    main()
