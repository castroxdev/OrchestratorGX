import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from app.core.llm_client import LLMClient
from app.core.supervisor import SupervisorAgent


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MODEL = "qwen2.5-coder:3b"


class WebChatHandler(BaseHTTPRequestHandler):
    # This handler is the web entry point of the project.
    # It does not contain the assistant logic itself: it only receives HTTP
    # requests, forwards user messages to the supervisor, and returns the
    # supervisor output to the browser.
    supervisor: SupervisorAgent
    static_routes = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/app.css": ("app.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    }

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        # The UI files are served directly from here so the browser can load
        # the chat page, CSS, and JavaScript.
        static_route = self.static_routes.get(path)
        if not static_route:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        filename, content_type = static_route
        self._serve_static_file(filename, content_type)

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        if path == "/api/chat/stream":
            self._handle_chat_stream()
            return

        if path != "/api/chat":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        # This is where a normal chat message first enters through the web API.
        # The browser sends JSON like {"message": "..."} to this route.
        try:
            payload = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body."})
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Message is required."})
            return

        try:
            # The user message is forwarded to the same SupervisorAgent used by
            # the rest of the project. The web layer does not decide agents or
            # tools on its own.
            response = self.supervisor.handle(message)
        except Exception as error:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": "Failed to process the message.",
                    "details": str(error),
                },
            )
            return

        # This is where the web layer receives the final SupervisorResponse and
        # converts it to JSON for the frontend.
        self._send_json(HTTPStatus.OK, response.model_dump())

    def _handle_chat_stream(self) -> None:
        # This route receives the user message the same way as /api/chat, but it
        # returns the final answer as a stream of small events instead of one
        # complete JSON response.
        try:
            payload = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid JSON body."})
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Message is required."})
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            def write_progress(progress_message: str) -> None:
                # Progress events are small messages for the UI, such as
                # "Selected agent: planner" or "Running tool: ...".
                self._write_stream_event(
                    {
                        "type": "progress",
                        "message": progress_message,
                    }
                )

            # The streaming path still uses the supervisor as the source of
            # truth. The difference is that the supervisor now yields progress
            # events and final response chunks over time.
            for event in self.supervisor.handle_stream(message, progress_callback=write_progress):
                self._write_stream_event(event)
        except Exception as error:
            self._write_stream_event(
                {
                    "type": "error",
                    "error": "Failed to process the message.",
                    "details": str(error),
                }
            )

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json_body(self) -> dict:
        # Reads the raw HTTP request body and parses it into a Python dict.
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        if not raw_body:
            return {}
        return json.loads(raw_body)

    def _serve_static_file(self, filename: str, content_type: str) -> None:
        file_path = STATIC_DIR / filename

        if not file_path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Static file not found")
            return

        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        # Sends a normal JSON response back to the browser.
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _write_stream_event(self, payload: dict) -> None:
        # Sends one NDJSON event line to the frontend stream reader.
        content = f"{json.dumps(payload)}\n".encode("utf-8")
        self.wfile.write(content)
        self.wfile.flush()


def build_server(host: str, port: int, model: str) -> ThreadingHTTPServer:
    # The web server creates a single SupervisorAgent instance and shares it
    # with request handlers, so both the CLI and web UI keep the same core flow.
    WebChatHandler.supervisor = SupervisorAgent(LLMClient(model=model))
    return ThreadingHTTPServer((host, port), WebChatHandler)


def main() -> None:
    # This is the web equivalent of app/main.py: it creates the server and
    # starts listening for browser requests instead of terminal input.
    parser = argparse.ArgumentParser(description="Run the OrchestratorGX web UI.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind the server to.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model used by the backend.")
    args = parser.parse_args()

    with build_server(args.host, args.port, args.model) as server:
        print(f"OrchestratorGX web UI running at http://{args.host}:{args.port}")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping web UI.")


if __name__ == "__main__":
    main()
