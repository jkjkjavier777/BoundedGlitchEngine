#!/usr/bin/env python3

"""
BoundedGlitchEngine → The-BoundedGlitchGPT client

This file does NOT import PyTorch.

Architecture:

    BoundedGlitchEngine
            |
            | HTTP
            v
    The-BoundedGlitchGPT
            |
            v
         PyTorch
            |
            v
      model.pt
"""

import json
import os
import urllib.error
import urllib.request


DEFAULT_SERVER_URL = os.environ.get(
    "BOUNDED_GLITCHGPT_URL",
    "http://127.0.0.1:8000"
)

DEFAULT_TIMEOUT = int(
    os.environ.get(
        "BOUNDED_GLITCHGPT_TIMEOUT",
        "120"
    )
)


class BoundedGlitchGPTClient:
    """HTTP client for The-BoundedGlitchGPT inference server."""

    def __init__(
        self,
        server_url=DEFAULT_SERVER_URL,
        timeout=DEFAULT_TIMEOUT,
    ):
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout

    def health(self):
        """Check whether The-BoundedGlitchGPT server is reachable."""

        url = f"{self.server_url}/health"

        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json"
            },
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=10,
            ) as response:

                body = response.read().decode("utf-8")

            return json.loads(body)

        except urllib.error.URLError as exc:

            return {
                "status": "error",
                "error": str(exc),
            }

        except Exception as exc:

            return {
                "status": "error",
                "error": str(exc),
            }

    def generate(
        self,
        prompt,
        max_tokens=150,
        temperature=0.8,
        top_k=None,
    ):
        """Send a prompt to The-BoundedGlitchGPT."""

        if not isinstance(prompt, str):
            raise ValueError(
                "prompt must be a string"
            )

        prompt = prompt.strip()

        if not prompt:
            raise ValueError(
                "prompt cannot be empty"
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than 0"
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0"
            )

        payload = {
            "prompt": prompt if prompt.startswith("User:") else f"User: {prompt}\nBot:",
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }

        if top_k is not None:
            payload["top_k"] = int(top_k)

        body = json.dumps(payload).encode("utf-8")

        request = urllib.request.Request(
            f"{self.server_url}/generate",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                response_body = response.read().decode(
                    "utf-8"
                )

        except urllib.error.HTTPError as exc:

            error_body = exc.read().decode(
                "utf-8",
                errors="replace"
            )

            raise RuntimeError(
                f"GPT server returned HTTP {exc.code}: "
                f"{error_body}"
            ) from exc

        except urllib.error.URLError as exc:

            raise RuntimeError(
                "Could not connect to "
                "The-BoundedGlitchGPT.\n"
                f"Server: {self.server_url}\n"
                "Check that the GPT server is running "
                "and that the server address is correct."
            ) from exc

        except TimeoutError as exc:

            raise RuntimeError(
                "The-BoundedGlitchGPT request timed out."
            ) from exc

        try:

            data = json.loads(response_body)

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "GPT server returned invalid JSON:\n"
                f"{response_body}"
            ) from exc

        if not isinstance(data, dict):

            raise RuntimeError(
                "GPT server returned an invalid response."
            )

        if "error" in data:

            raise RuntimeError(
                str(data["error"])
            )

        text = data.get("text")

        if not isinstance(text, str):

            raise RuntimeError(
                "GPT server response did not contain "
                "a 'text' field."
            )

        return text.strip()


def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Test the BoundedGlitchEngine "
            "connection to The-BoundedGlitchGPT."
        )
    )

    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER_URL,
        help="The-BoundedGlitchGPT server URL",
    )

    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt to send to the GPT",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=50,
        help="Maximum generated tokens",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Generation temperature",
    )

    args = parser.parse_args()

    client = BoundedGlitchGPTClient(
        server_url=args.server
    )

    print()
    print("=" * 60)
    print("BOUNDEDGLITCHENGINE → BOUNDEDGLITCHGPT")
    print("=" * 60)
    print()
    print(f"Server: {client.server_url}")
    print()

    status = client.health()

    print("Health:")
    print(json.dumps(status, indent=2))
    print()

    if status.get("status") != "ok":

        print(
            "[ERROR] The-BoundedGlitchGPT "
            "server is not reachable."
        )

        raise SystemExit(1)

    print("[✓] GPT connection established.")
    print()

    if args.prompt:

        try:

            response = client.generate(
                args.prompt,
                max_tokens=args.tokens,
                temperature=args.temperature,
            )

            print("GPT:")
            print(response)

        except Exception as exc:

            print(f"[ERROR] {exc}")
            raise SystemExit(1)

        return

    print("Enter a prompt.")
    print("Type /quit to exit.")
    print()

    while True:

        try:

            prompt = input("You: ").strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print()
            break

        if not prompt:
            continue

        if prompt.lower() in {
            "/quit",
            "/exit",
        }:

            break

        try:

            response = client.generate(
                prompt
            )

            print()
            print("GPT:", response)
            print()

        except Exception as exc:

            print()
            print(f"[ERROR] {exc}")
            print()


if __name__ == "__main__":
    main()

