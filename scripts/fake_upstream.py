"""A scripted OpenAI-compatible upstream for smoke tests: python fake_upstream.py PORT NAME|down"""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if sys.argv[2] == 'down':
            self.send_response(503); self.end_headers(); self.wfile.write(b'{"error":{"message":"down"}}'); return
        if body.get('stream'):
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            for w in ['Hello', ' from', ' ' + sys.argv[2]]:
                self.wfile.write(b'data: ' + json.dumps({"choices":[{"index":0,"delta":{"content":w}}]}).encode() + b'\n\n')
            self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}\n\ndata: [DONE]\n\n'); return
        out = json.dumps({"id":"x","object":"chat.completion","model":body["model"],"choices":[{"index":0,"message":{"role":"assistant","content":"answer from "+sys.argv[2]},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}).encode()
        self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length', str(len(out))); self.end_headers(); self.wfile.write(out)
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
