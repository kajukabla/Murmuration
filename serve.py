#!/usr/bin/env python3
"""
Local dev server with benchmark coordination endpoints.
- GET /api/bench_request — page polls this to check for benchmark work
- POST /api/bench_result — page posts benchmark results here
- GET/POST handled for benchmark coordination, everything else is static files
"""

import os
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(DIR, "bench_result.json")
REQUEST_FILE = os.path.join(DIR, "bench_request.json")


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/bench_request':
            # Page polls this — returns {boids: N} if there's a pending request
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            if os.path.exists(REQUEST_FILE):
                with open(REQUEST_FILE) as f:
                    self.wfile.write(f.read().encode())
            else:
                self.wfile.write(b'{}')
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == '/api/bench_result':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            with open(RESULT_FILE, 'w') as f:
                f.write(body.decode())
            # Clear the request since it's been fulfilled
            try: os.remove(REQUEST_FILE)
            except: pass
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == '__main__':
    os.chdir(DIR)
    print(f"Serving on http://localhost:{PORT}")
    HTTPServer(('', PORT), Handler).serve_forever()
