#!/usr/bin/env python3
"""
Local dev server that also handles POST /api/bench_result.
Writes benchmark results to bench_result.json for evaluate.py to read.
Usage: python3 serve.py [port]
"""

import os
import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
RESULT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_result.json")


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/bench_result':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            with open(RESULT_FILE, 'w') as f:
                f.write(body.decode())
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # quiet


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Serving on http://localhost:{PORT}")
    HTTPServer(('', PORT), Handler).serve_forever()
