#!/usr/bin/env python3
"""
Live dashboard for the autoresearch optimization loop.
Auto-refreshes every 5 seconds, reads run_metrics.csv and results.tsv.

Usage: python3 dashboard.py
Then open http://localhost:8050
"""

import os
import csv
import signal
import subprocess
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Global process handle for the agent loop
agent_proc = None

PORT = 8050
CSV_FILE = "run_metrics.csv"
TSV_FILE = "results.tsv"

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Autoresearch Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a1a; color: #ccd; font-family: system-ui, sans-serif; padding: 24px; }
  h1 { font-size: 22px; color: #8af; margin-bottom: 4px; }
  .subtitle { color: #667; font-size: 13px; margin-bottom: 24px; }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 28px; }
  .card {
    background: rgba(20,20,45,0.9); border: 1px solid rgba(100,100,180,0.2);
    border-radius: 8px; padding: 16px;
  }
  .card .label { color: #668; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 32px; font-weight: 700; color: #adf; margin-top: 4px; }
  .card .value.good { color: #6f8; }
  .card .value.warn { color: #fa6; }
  .chart-container {
    background: rgba(20,20,45,0.9); border: 1px solid rgba(100,100,180,0.2);
    border-radius: 8px; padding: 20px; margin-bottom: 20px;
  }
  .chart-title { color: #889; font-size: 13px; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }
  canvas { width: 100%; height: 280px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: #668; font-size: 11px; text-transform: uppercase;
       letter-spacing: 1px; padding: 8px 12px; border-bottom: 1px solid rgba(100,100,180,0.2); }
  td { padding: 8px 12px; border-bottom: 1px solid rgba(100,100,180,0.08); color: #aab; }
  tr.kept td { color: #6f8; }
  tr.reverted td { color: #f88; }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  .status-dot.kept { background: #6f8; }
  .status-dot.reverted { background: #f66; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  #last-update { color: #445; font-size: 11px; position: fixed; bottom: 12px; right: 16px; }
</style>
</head>
<body>
<h1>Autoresearch Optimization Dashboard</h1>
<div class="subtitle">Murmuration Quality Optimization — Score 0-1</div>

<div style="margin-bottom: 20px; display: flex; gap: 10px; align-items: center;">
  <button id="startBtn" onclick="startAgent()" style="padding: 8px 20px; font-size: 14px; cursor: pointer;
    background: #2a4; color: #fff; border: none; border-radius: 6px; font-weight: 600;">Start Agent</button>
  <button id="stopBtn" onclick="stopAgent()" style="padding: 8px 20px; font-size: 14px; cursor: pointer;
    background: #c44; color: #fff; border: none; border-radius: 6px; font-weight: 600;" disabled>Stop Agent</button>
  <span id="agentStatus" style="color: #667; font-size: 13px; margin-left: 8px;">Agent idle</span>
</div>

<div class="grid">
  <div class="card"><div class="label">Best Score</div><div class="value good" id="best">—</div></div>
  <div class="card"><div class="label">Baseline</div><div class="value" id="baseline">0.695</div></div>
  <div class="card"><div class="label">Experiments</div><div class="value" id="experiments">—</div></div>
  <div class="card"><div class="label">Kept</div><div class="value" id="kept">—</div></div>
</div>

<div class="two-col">
  <div class="chart-container">
    <div class="chart-title">Quality Score Over Time (kept experiments)</div>
    <canvas id="boidChart"></canvas>
  </div>
  <div class="chart-container">
    <div class="chart-title">Frame Time vs Boid Count (all probes)</div>
    <canvas id="timeChart"></canvas>
  </div>
</div>

<div class="two-col">
  <div class="chart-container">
    <div class="chart-title">Experiment Log</div>
    <div style="max-height: 280px; overflow-y: auto;">
    <table>
      <thead><tr><th>#</th><th>Max Boids</th><th>Description</th><th>Result</th></tr></thead>
      <tbody id="logBody"></tbody>
    </table>
    </div>
  </div>
  <div class="chart-container">
    <div class="chart-title">Git Activity (live)</div>
    <div id="gitLog" style="max-height: 280px; overflow-y: auto; font: 12px/1.6 monospace; color: #99a;"></div>
  </div>
</div>
<div class="chart-container">
  <div class="chart-title">Agent Log (tail)</div>
  <pre id="agentLog" style="max-height: 200px; overflow-y: auto; font: 11px/1.5 monospace; color: #889; white-space: pre-wrap; margin: 0;"></pre>
</div>

<div id="last-update"></div>

<script>
async function fetchData() {
  const resp = await fetch('/api/data');
  return await resp.json();
}

function drawChart(canvas, points, opts) {
  const ctx = canvas.getContext('2d');
  const dpr = devicePixelRatio;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  const pad = { top: 10, right: 20, bottom: 30, left: 60 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  if (points.length === 0) return;

  const xs = points.map(p => p.x);
  const ys = points.map(p => p.y);
  const xMin = opts.xMin ?? Math.min(...xs);
  const xMax = opts.xMax ?? Math.max(...xs);
  const yMin = opts.yMin ?? Math.min(...ys) * 0.9;
  const yMax = opts.yMax ?? Math.max(...ys) * 1.1;

  const toX = v => pad.left + (v - xMin) / (xMax - xMin || 1) * plotW;
  const toY = v => pad.top + plotH - (v - yMin) / (yMax - yMin || 1) * plotH;

  // Grid
  ctx.strokeStyle = 'rgba(100,100,180,0.1)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + plotH * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
  }

  // Axis labels
  ctx.fillStyle = '#556';
  ctx.font = '10px system-ui';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = yMin + (yMax - yMin) * (4 - i) / 4;
    ctx.fillText(opts.yFmt ? opts.yFmt(val) : val.toFixed(1), pad.left - 6, pad.top + plotH * i / 4 + 3);
  }
  ctx.textAlign = 'center';
  for (let i = 0; i <= 4; i++) {
    const val = xMin + (xMax - xMin) * i / 4;
    ctx.fillText(opts.xFmt ? opts.xFmt(val) : val.toFixed(0), toX(val), H - 8);
  }

  // Threshold line
  if (opts.threshold) {
    ctx.strokeStyle = '#f664';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, toY(opts.threshold));
    ctx.lineTo(W - pad.right, toY(opts.threshold));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f66';
    ctx.textAlign = 'left';
    ctx.fillText(opts.thresholdLabel || '', W - pad.right + 4, toY(opts.threshold) + 3);
  }

  // Line
  if (opts.line) {
    ctx.strokeStyle = opts.color || '#8af';
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((p, i) => {
      const x = toX(p.x), y = toY(p.y);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Points
  points.forEach(p => {
    ctx.fillStyle = p.color || opts.color || '#8af';
    ctx.beginPath();
    ctx.arc(toX(p.x), toY(p.y), opts.pointSize || 3, 0, Math.PI * 2);
    ctx.fill();
  });
}

async function refresh() {
  try {
    const data = await fetchData();

    // Summary cards
    document.getElementById('best').textContent = typeof data.best === 'number' && data.best < 100 ? data.best.toFixed(4) : data.best.toLocaleString();
    document.getElementById('experiments').textContent = data.experiments.length.toString();
    document.getElementById('kept').textContent = data.experiments.filter(e => e.result === 'kept').length.toString();

    // Experiment log table
    const tbody = document.getElementById('logBody');
    tbody.innerHTML = '';
    for (const exp of [...data.experiments].reverse()) {
      const tr = document.createElement('tr');
      tr.className = exp.result;
      tr.innerHTML = `<td>${exp.id}</td><td>${exp.max_boids.toLocaleString()}</td>` +
        `<td>${exp.description}</td>` +
        `<td><span class="status-dot ${exp.result}"></span>${exp.result}</td>`;
      tbody.appendChild(tr);
    }

    // Score chart (kept experiments over time)
    const kept = data.experiments.filter(e => e.result === 'kept');
    const isScore = kept.length > 0 && kept[kept.length-1].max_boids < 100;
    drawChart(document.getElementById('boidChart'),
      kept.map((e, i) => ({ x: i, y: e.max_boids })),
      { line: true, color: '#6f8',
        yFmt: isScore ? (v => v.toFixed(3)) : (v => (v/1000).toFixed(0) + 'k'),
        xFmt: v => '#' + Math.round(v),
        yMin: isScore ? 0 : 0, yMax: isScore ? 1.0 : undefined }
    );

    // Time chart (all probes as scatter)
    drawChart(document.getElementById('timeChart'),
      data.probes.map(p => ({
        x: p.boids / 1000,
        y: p.p99_ms,
        color: p.passed ? '#6f84' : '#f664',
      })),
      { line: false, color: '#8af', pointSize: 3,
        xFmt: v => v.toFixed(0) + 'k', yFmt: v => v.toFixed(0) + 'ms',
        threshold: 16.6, thresholdLabel: '60 FPS', yMin: 0 }
    );

    // Git log
    const gitDiv = document.getElementById('gitLog');
    if (gitDiv) {
      gitDiv.innerHTML = (data.git_log || []).map(g => {
        const isExp = g.msg.startsWith('experiment:');
        const color = isExp ? '#8af' : '#556';
        return `<div style="color:${color}"><span style="color:#445">${g.hash}</span> ${g.msg}</div>`;
      }).join('');
    }

    // Agent log
    const agentLog = document.getElementById('agentLog');
    if (agentLog) agentLog.textContent = data.agent_log || '(buffered — output appears when agent completes)';

    // Update agent status
    updateAgentUI(data.agent_running);

    document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
  } catch (e) {
    console.error('Refresh error:', e);
  }
}

async function startAgent() {
  const resp = await fetch('/api/start', { method: 'POST' });
  const data = await resp.json();
  updateAgentUI(data.running);
}
async function stopAgent() {
  const resp = await fetch('/api/stop', { method: 'POST' });
  const data = await resp.json();
  updateAgentUI(data.running);
}
function updateAgentUI(running) {
  document.getElementById('startBtn').disabled = running;
  document.getElementById('stopBtn').disabled = !running;
  document.getElementById('agentStatus').textContent = running ? 'Agent running...' : 'Agent idle';
  document.getElementById('agentStatus').style.color = running ? '#6f8' : '#667';
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>"""


def is_agent_running():
    global agent_proc
    if agent_proc is None:
        return False
    ret = agent_proc.poll()
    if ret is not None:
        agent_proc = None
        return False
    return True


def start_agent():
    global agent_proc
    if is_agent_running():
        return True
    work_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(work_dir, 'agent.log')
    # Clear log for fresh run
    open(log_path, 'w').close()
    log_file = open(log_path, 'a')
    # Use --verbose --output-format stream-json for live streaming output
    cmd = 'claude --verbose --output-format stream-json --dangerously-skip-permissions "Read program.md and begin the optimization loop. Do not stop or ask questions."'
    agent_proc = subprocess.Popen(
        cmd, shell=True,
        stdout=log_file, stderr=subprocess.STDOUT,
        cwd=work_dir,
        env={**os.environ, 'PATH': os.environ.get('PATH', '') + ':/usr/local/bin'},
        bufsize=0,  # unbuffered
    )
    return True


def stop_agent():
    global agent_proc
    if agent_proc is not None:
        agent_proc.send_signal(signal.SIGTERM)
        try:
            agent_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            agent_proc.kill()
        agent_proc = None
    return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress logs

    def do_GET(self):
        if self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            data = get_data()
            data['agent_running'] = is_agent_running()
            self.wfile.write(json.dumps(data).encode())
        else:
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML.encode())

    def do_POST(self):
        if self.path == '/api/start':
            running = start_agent()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'running': running}).encode())
        elif self.path == '/api/stop':
            running = stop_agent()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'running': running}).encode())
        else:
            self.send_response(404)
            self.end_headers()


def get_data():
    experiments = []
    probes = []

    # Read results.tsv
    if os.path.exists(TSV_FILE):
        with open(TSV_FILE) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                try:
                    val = row.get('max_boids', '0')
                    experiments.append({
                        'id': int(row.get('experiment', 0)),
                        'max_boids': float(val) if '.' in val else int(val),
                        'description': row.get('description', ''),
                        'result': row.get('result', ''),
                    })
                except (ValueError, KeyError):
                    pass

    # Read run_metrics.csv
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    probes.append({
                        'boids': int(row.get('Particle_Count', 0)),
                        'avg_ms': float(row.get('Avg_Frame_Time', 0)),
                        'p99_ms': float(row.get('P99_Frame_Time', 0)),
                        'passed': row.get('Result', '') == 'Pass',
                    })
                except (ValueError, KeyError):
                    pass

    best = max((e['max_boids'] for e in experiments), default=0)

    # Git log
    git_log = []
    try:
        result = subprocess.run(
            ['git', 'log', '--oneline', '-20', '--no-decorate'],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    git_log.append({'hash': parts[0], 'msg': parts[1] if len(parts) > 1 else ''})
    except Exception:
        pass

    # Agent log tail — parse stream-json for readable messages
    agent_log_lines = []
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent.log')
    try:
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    msg_type = obj.get('type', '')
                    if msg_type == 'assistant' and 'message' in obj:
                        # Extract text content from assistant messages
                        for block in obj['message'].get('content', []):
                            if isinstance(block, dict) and block.get('type') == 'text':
                                agent_log_lines.append(f"[THINKING] {block['text'][:200]}")
                            elif isinstance(block, dict) and block.get('type') == 'tool_use':
                                name = block.get('name', '?')
                                inp = str(block.get('input', ''))[:150]
                                agent_log_lines.append(f"[TOOL] {name}: {inp}")
                    elif msg_type == 'result' and 'result' in obj:
                        text = obj['result'].get('text', '')[:200] if isinstance(obj['result'], dict) else str(obj['result'])[:200]
                        if text:
                            agent_log_lines.append(f"[RESULT] {text}")
                except json.JSONDecodeError:
                    # Not JSON, just show raw line
                    if len(line) < 300:
                        agent_log_lines.append(line)
    except Exception:
        pass
    agent_log_tail = '\n'.join(agent_log_lines[-50:])

    return {
        'best': best,
        'experiments': experiments,
        'probes': probes,
        'git_log': git_log,
        'agent_log': agent_log_tail,
    }


if __name__ == '__main__':
    print(f"Dashboard running at http://localhost:{PORT}")
    print("Auto-refreshes every 5 seconds. Ctrl+C to stop.")
    HTTPServer(('', PORT), Handler).serve_forever()
