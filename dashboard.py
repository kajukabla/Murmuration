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

# 6 results files — one per optimization phase
RESULT_FILES = {
    'classic_perf': 'results_classic_perf.tsv',
    'classic_compute': 'results_classic_compute.tsv',
    'classic_quality': 'results_classic_quality.tsv',
    'topo_perf': 'results_topo_perf.tsv',
    'topo_compute': 'results_topo_compute.tsv',
    'topo_quality': 'results_topo_quality.tsv',
}

HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Murmuration Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a1a;color:#ccd;font-family:system-ui,sans-serif;padding:24px}
h1{font-size:22px;color:#8af;margin-bottom:16px}
.row{display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.card{background:rgba(20,20,45,.9);border:1px solid rgba(100,100,180,.2);border-radius:8px;padding:14px}
.card .l{color:#668;font-size:10px;text-transform:uppercase;letter-spacing:1px}
.card .v{font-size:28px;font-weight:700;color:#6f8;margin-top:4px}
.box{background:rgba(20,20,45,.9);border:1px solid rgba(100,100,180,.2);border-radius:8px;padding:14px;margin-bottom:12px}
.box h3{color:#889;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
canvas{width:100%;height:220px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:12px}
table{width:100%;border-collapse:collapse;font-size:11px}
th{text-align:left;color:#668;font-size:10px;text-transform:uppercase;padding:5px 8px;border-bottom:1px solid rgba(100,100,180,.2)}
td{padding:4px 8px;border-bottom:1px solid rgba(100,100,180,.06);color:#aab}
tr.kept td{color:#6f8} tr.reverted td{color:#f88}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:4px}
.dot.kept{background:#6f8}.dot.reverted{background:#f66}
.btn{padding:7px 16px;font-size:12px;cursor:pointer;border:none;border-radius:6px;font-weight:600}
.tab{padding:6px 14px;font-size:11px;cursor:pointer;border:1px solid rgba(100,100,180,.2);border-bottom:none;border-radius:5px 5px 0 0}
.tab.on{background:rgba(40,40,80,.9);color:#aac}.tab.off{background:rgba(15,15,30,.5);color:#556}
@keyframes spin{to{transform:rotate(360deg)}}
.spin{display:inline-block;width:12px;height:12px;border:2px solid #334;border-top-color:#6f8;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:5px}
#ts{color:#445;font-size:10px;position:fixed;bottom:10px;right:14px}
.sched{background:rgba(20,30,20,.5);border:1px solid rgba(100,180,100,.15);border-radius:6px;padding:8px 14px;font-size:12px;color:#8a8}
</style></head><body>
<h1>Murmuration Dashboard</h1>
<div class="row">
<button class="btn" id="go" onclick="api('start')" style="background:#2a4;color:#fff">Start Agent</button>
<button class="btn" id="no" onclick="api('stop')" style="background:#c44;color:#fff" disabled>Stop Agent</button>
<span id="st" style="color:#667;font-size:13px;margin-left:8px">Idle</span></div>
<div id="sched" class="sched" style="margin-bottom:12px;display:none"></div>
<div class="row" style="gap:2px">
<button class="tab on" id="t-classic_perf" onclick="T('classic_perf')">Classic Perf</button>
<button class="tab off" id="t-classic_compute" onclick="T('classic_compute')">Classic Compute</button>
<button class="tab off" id="t-classic_quality" onclick="T('classic_quality')">Classic Quality</button>
<button class="tab off" id="t-topo_perf" onclick="T('topo_perf')">Topo Perf</button>
<button class="tab off" id="t-topo_compute" onclick="T('topo_compute')">Topo Compute</button>
<button class="tab off" id="t-topo_quality" onclick="T('topo_quality')">Topo Quality</button></div>
<div class="grid4">
<div class="card"><div class="l" id="lb">Max Boids</div><div class="v" id="vb">-</div></div>
<div class="card"><div class="l">Baseline</div><div class="v" id="vbl">-</div></div>
<div class="card"><div class="l">Experiments</div><div class="v" id="ve">-</div></div>
<div class="card"><div class="l">Kept</div><div class="v" id="vk">-</div></div></div>
<div class="cols">
<div class="box"><h3 id="c1t">Improvements Over Time</h3><canvas id="c1"></canvas></div>
<div class="box"><h3>All Experiments</h3><canvas id="c2"></canvas></div></div>
<div class="cols">
<div class="box"><h3>Experiment Log</h3><div style="max-height:220px;overflow-y:auto">
<table><thead><tr><th>#</th><th id="cv">Value</th><th>Description</th><th>Result</th></tr></thead>
<tbody id="tb"></tbody></table></div></div>
<div class="box"><h3>Git Log</h3><div id="gl" style="max-height:220px;overflow-y:auto;font:11px/1.5 monospace;color:#99a"></div></div></div>
<div class="box"><h3>Agent Log</h3><pre id="al" style="max-height:140px;overflow-y:auto;font:10px/1.4 monospace;color:#889;white-space:pre-wrap;margin:0"></pre></div>
<div id="ts"></div>
<script>
const CFG={
classic_perf:{lbl:'Max Boids (Browser)',bl:'118,000',sc:0},
classic_compute:{lbl:'Max Boids (Deno)',bl:'170,000',sc:0},
classic_quality:{lbl:'Quality Score',bl:'0.50',sc:1},
topo_perf:{lbl:'Max Boids (Browser)',bl:'20,000',sc:0},
topo_compute:{lbl:'Max Boids (Deno)',bl:'20,000',sc:0},
topo_quality:{lbl:'Quality Score',bl:'0.695',sc:1}};
let tab='classic_perf';
function T(t){tab=t;document.querySelectorAll('.tab').forEach(b=>{b.className='tab off'});
document.getElementById('t-'+t).className='tab on';
const c=CFG[t];document.getElementById('lb').textContent=c.lbl;
document.getElementById('vbl').textContent=c.bl;
document.getElementById('cv').textContent=c.sc?'Score':'Max Boids';
document.getElementById('c1t').textContent=c.sc?'Score Over Time':'Max Boids Over Time';
R();}
function ch(cv,pts,o){if(!pts.length)return;const c=cv.getContext('2d'),d=devicePixelRatio,
r=cv.getBoundingClientRect();cv.width=r.width*d;cv.height=r.height*d;c.scale(d,d);
const W=r.width,H=r.height,p={t:8,r:16,b:28,l:55},pW=W-p.l-p.r,pH=H-p.t-p.b;
const xs=pts.map(v=>v.x),ys=pts.map(v=>v.y);
const x0=o.xMin??Math.min(...xs),x1=o.xMax??Math.max(...xs);
const y0=o.yMin??Math.min(...ys)*.9,y1=o.yMax??Math.max(...ys)*1.1;
const tx=v=>p.l+(v-x0)/(x1-x0||1)*pW,ty=v=>p.t+pH-(v-y0)/(y1-y0||1)*pH;
c.strokeStyle='rgba(100,100,180,.1)';c.lineWidth=1;
for(let i=0;i<=4;i++){const y=p.t+pH*i/4;c.beginPath();c.moveTo(p.l,y);c.lineTo(W-p.r,y);c.stroke();}
c.fillStyle='#556';c.font='9px system-ui';c.textAlign='right';
for(let i=0;i<=4;i++){const v=y0+(y1-y0)*(4-i)/4;c.fillText(o.yF?o.yF(v):v.toFixed(1),p.l-4,p.t+pH*i/4+3);}
c.textAlign='center';for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4;c.fillText('#'+Math.round(v),tx(v),H-6);}
if(o.line){c.strokeStyle=o.color||'#8af';c.lineWidth=2;c.beginPath();
pts.forEach((v,i)=>{i?c.lineTo(tx(v.x),ty(v.y)):c.moveTo(tx(v.x),ty(v.y))});c.stroke();}
pts.forEach(v=>{c.fillStyle=v.c||o.color||'#8af';c.beginPath();c.arc(tx(v.x),ty(v.y),o.ps||3,0,Math.PI*2);c.fill();});}
async function R(){try{const d=await(await fetch('/api/data')).json();
const ae=d.all_exps||{};const ab=d.all_bests||{};
const ex=ae[tab]||[];const best=ab[tab]||0;const kept=ex.filter(e=>e.result==='kept');
const sc=CFG[tab].sc;const f=v=>sc?v.toFixed(4):v.toLocaleString();
const yF=sc?(v=>v.toFixed(2)):(v=>v<1e4?v.toFixed(0):(v/1e3).toFixed(0)+'k');
document.getElementById('vb').textContent=f(best);
document.getElementById('ve').textContent=ex.length;
document.getElementById('vk').textContent=kept.length;
const tb=document.getElementById('tb');tb.innerHTML='';
[...ex].reverse().forEach(e=>{const tr=document.createElement('tr');tr.className=e.result;
tr.innerHTML='<td>'+e.id+'</td><td>'+f(e.max_boids)+'</td><td>'+e.description+'</td><td><span class="dot '+e.result+'"></span>'+e.result+'</td>';
tb.appendChild(tr);});
ch(document.getElementById('c1'),kept.map((e,i)=>({x:i,y:e.max_boids})),{line:1,color:'#6f8',yF,yMin:0});
ch(document.getElementById('c2'),ex.map((e,i)=>({x:i,y:e.max_boids,c:e.result==='kept'?'#6f8':'#f664'})),{color:'#8af',ps:4,yF,yMin:0});
const g=document.getElementById('gl');
if(g)g.innerHTML=(d.git_log||[]).map(x=>'<div style="color:'+(x.msg.startsWith('experiment:')?'#8af':'#556')+'"><span style="color:#445">'+x.hash+'</span> '+x.msg+'</div>').join('');
const a=document.getElementById('al');if(a){a.textContent=d.agent_log||'';a.scrollTop=a.scrollHeight;}
document.getElementById('go').disabled=d.agent_running;
document.getElementById('no').disabled=!d.agent_running;
const s=document.getElementById('st');
if(d.agent_running){s.innerHTML='<span class="spin"></span>Running';s.style.color='#6f8';}
else{s.textContent='Idle';s.style.color='#667';}
const sd=d.scheduler||{};const se=document.getElementById('sched');
if(sd.running){se.style.display='block';const rem=Math.max(0,sd.duration_min-Math.floor((Date.now()/1000-new Date(sd.phase_start).getTime()/1000)/60));
se.innerHTML='<b>Scheduler:</b> '+sd.current_phase+' ('+rem+'min left) | Phase '+(sd.total_completed+1);}
else{se.style.display='none';}
document.getElementById('ts').textContent=new Date().toLocaleTimeString();
}catch(e){console.error(e);}}
async function api(a){await fetch('/api/'+a,{method:'POST'});R();}
T('classic_perf');setInterval(R,5000);
</script></body></html>"""


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
        bufsize=0,
        preexec_fn=os.setsid,  # new process group so we can kill ALL children
    )
    return True


def stop_agent():
    global agent_proc
    if agent_proc is not None:
        try:
            # Kill the ENTIRE process group (agent + evaluate.py + bench + chromium)
            pgid = os.getpgid(agent_proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            agent_proc.wait(timeout=5)
        except (ProcessLookupError, ChildProcessError):
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(agent_proc.pid), signal.SIGKILL)
            except (ProcessLookupError, ChildProcessError):
                pass
        agent_proc = None
    # Also kill any stray processes and clear benchmark requests
    subprocess.run(['pkill', '-9', '-f', 'chromium-1208'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'bench_browser'], capture_output=True)
    subprocess.run(['pkill', '-9', '-f', 'eval_wrapper'], capture_output=True)
    # Clear bench request so the browser stops reloading
    req_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_request.json')
    try: os.remove(req_file)
    except: pass
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


def read_tsv(filepath):
    results = []
    if os.path.exists(filepath):
        with open(filepath) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                try:
                    val = row.get('max_boids', '0') or '0'
                    exp_id = row.get('experiment', '0') or '0'
                    results.append({
                        'id': int(exp_id),
                        'max_boids': float(val) if '.' in val else int(val),
                        'description': row.get('description', ''),
                        'result': row.get('result', ''),
                    })
                except (ValueError, KeyError, TypeError):
                    pass
    return results


def get_data():
    # Read all 6 result files
    all_exps = {}
    all_bests = {}
    for key, filepath in RESULT_FILES.items():
        exps = read_tsv(filepath)
        all_exps[key] = exps
        kept = [e['max_boids'] for e in exps if e['result'] == 'kept']
        all_bests[key] = max(kept) if kept else 0

    # Backwards compat
    quality_exps = all_exps.get('topo_quality', [])
    compute_exps = all_exps.get('classic_compute', [])
    perf_exps = all_exps.get('classic_perf', [])
    probes = []

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    probes.append({
                        'boids': int(row.get('Particle_Count') or 0),
                        'avg_ms': float(row.get('Avg_Frame_Time') or 0),
                        'p99_ms': float(row.get('P99_Frame_Time') or 0),
                        'passed': row.get('Result', '') == 'Pass',
                    })
                except (ValueError, KeyError, TypeError):
                    pass

    quality_best = all_bests.get('topo_quality', 0)
    perf_best = all_bests.get('classic_perf', 0)
    compute_best = all_bests.get('classic_compute', 0)

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

    # Read scheduler status if available
    sched_status = {}
    sched_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scheduler_status.json')
    if os.path.exists(sched_file):
        try:
            with open(sched_file) as f:
                sched_status = json.loads(f.read())
        except: pass

    return {
        'quality_best': quality_best,
        'perf_best': perf_best,
        'compute_best': compute_best,
        'quality_experiments': quality_exps,
        'perf_experiments': perf_exps,
        'compute_experiments': compute_exps,
        'all_exps': all_exps,
        'all_bests': all_bests,
        'scheduler': sched_status,
        'probes': probes,
        'git_log': git_log,
        'agent_log': agent_log_tail,
    }


if __name__ == '__main__':
    print(f"Dashboard running at http://localhost:{PORT}")
    print("Auto-refreshes every 5 seconds. Ctrl+C to stop.")
    HTTPServer(('', PORT), Handler).serve_forever()
