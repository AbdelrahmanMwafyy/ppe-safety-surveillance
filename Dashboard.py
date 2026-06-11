import os
import sys

# ── ENVIRONMENT ENFORCER ─────────────────────────────────────────────
# Force this script to ALWAYS use the Python 3.13 environment
CORRECT_PYTHON = r"C:\Mwafy\working apps\python\python.exe"

if os.path.normcase(sys.executable) != os.path.normcase(CORRECT_PYTHON):
    # Safe quote-wrapping for paths with spaces to prevent Windows argument splitting
    args = [f'"{arg}"' if ' ' in arg and not (arg.startswith('"') and arg.endswith('"')) else arg for arg in sys.argv]
    # Relaunch using system command execution to respect space formatting
    import subprocess
    subprocess.run([CORRECT_PYTHON] + sys.argv)
    sys.exit()
# ─────────────────────────────────────────────────────────────────────

from datetime import datetime, timedelta
from flask import Flask, request, redirect, url_for, send_from_directory
from flask import render_template_string, session
# ... (rest of your code below stays exactly the same)


# ── Make sure database.py is importable ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database import SafetyDB, SCREENSHOT_DIR

app   = Flask(__name__)
app.secret_key = 'safetyai_2026_mee'   # change this if you want

# ── Dashboard password ──────────────────────────────────────────
DASHBOARD_PASSWORD = 'Grad26'   # ← change this to your password

db    = SafetyDB()

VIOLATION_COLORS = {
    'NO_HELMET':   '#f97316',
    'NO_VEST':     '#eab308',
    'UNAUTH_ZONE': '#ef4444',
    'FIRE':        '#ff2d2d',
}

# ─────────────────────────────────────────────────────────────────────
#  BASE LAYOUT
# ─────────────────────────────────────────────────────────────────────
def base(title, content, active='home'):
    nav_items = [
        ('home',       '/',           '⬡', 'Overview'),
        ('persons',    '/persons',    '◈', 'Workers'),
        ('violations', '/violations', '⚠', 'Violations'),
        ('charts',     '/charts',     '◎', 'Trends'),
        ('review',     '/review',     '◉', 'Review'),
    ]
    nav_html = ''
    for key, href, icon, label in nav_items:
        active_cls = 'nav-active' if key == active else ''
        nav_html += f'''
        <a href="{href}" class="nav-item {active_cls}">
            <span class="nav-icon">{icon}</span>
            <span class="nav-label">{label}</span>
        </a>'''

    return render_template_string(f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — AI Safety Surveillance</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {{
    --bg:       #0b0d12;
    --surface:  #12151d;
    --card:     #181c27;
    --border:   #252b3b;
    --orange:   #f97316;
    --yellow:   #eab308;
    --green:    #22c55e;
    --red:      #ef4444;
    --cyan:     #22d3ee;
    --blue:     #3b82f6;
    --text:     #dde3f0;
    --muted:    #5a6480;
    --sidebar:  220px;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    display: flex;
    min-height: 100vh;
}}
/* ── Sidebar ── */
.sidebar {{
    width: var(--sidebar);
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    position: fixed;
    height: 100vh;
    z-index: 10;
}}
.sidebar-logo {{
    padding: 24px 20px 20px;
    border-bottom: 1px solid var(--border);
}}
.sidebar-logo .logo-text {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--orange);
    letter-spacing: 2px;
    text-transform: uppercase;
}}
.sidebar-logo .logo-sub {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 1px;
    margin-top: 2px;
}}
.sidebar-status {{
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--muted);
}}
.status-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 6px var(--green);
    animation: pulse 2s infinite;
}}
@keyframes pulse {{ 0%,100%{{ opacity:1 }} 50%{{ opacity:0.4 }} }}
.nav-item {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 13px 20px;
    color: var(--muted);
    text-decoration: none;
    font-family: 'Rajdhani', sans-serif;
    font-size: 15px;
    font-weight: 500;
    letter-spacing: 1px;
    text-transform: uppercase;
    border-left: 3px solid transparent;
    transition: all 0.15s;
}}
.nav-item:hover {{ color: var(--text); background: rgba(255,255,255,0.03); }}
.nav-active {{ color: var(--orange); border-left-color: var(--orange); background: rgba(249,115,22,0.06); }}
.nav-icon {{ font-size: 16px; width: 20px; text-align: center; }}
.sidebar-footer {{
    margin-top: auto;
    padding: 16px 20px;
    border-top: 1px solid var(--border);
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    color: var(--muted);
}}
/* ── Main ── */
.main {{
    margin-left: var(--sidebar);
    flex: 1;
    padding: 32px;
    max-width: calc(100vw - var(--sidebar));
}}
.page-header {{
    margin-bottom: 28px;
}}
.page-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text);
}}
.page-sub {{
    color: var(--muted);
    font-size: 13px;
    margin-top: 4px;
    font-family: 'DM Mono', monospace;
}}
/* ── Cards ── */
.cards {{ display: grid; gap: 16px; margin-bottom: 28px; }}
.cards-4 {{ grid-template-columns: repeat(4, 1fr); }}
.cards-3 {{ grid-template-columns: repeat(3, 1fr); }}
.cards-2 {{ grid-template-columns: repeat(2, 1fr); }}
.card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
}}
.card-label {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}}
.card-value {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
}}
.card-sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
.card-orange .card-value {{ color: var(--orange); }}
.card-red    .card-value {{ color: var(--red); }}
.card-green  .card-value {{ color: var(--green); }}
.card-cyan   .card-value {{ color: var(--cyan); }}
.card-yellow .card-value {{ color: var(--yellow); }}
/* ── Section ── */
.section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 24px;
    overflow: hidden;
}}
.section-head {{
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
}}
.section-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text);
}}
.section-body {{ padding: 0; }}
/* ── Table ── */
table {{ width: 100%; border-collapse: collapse; }}
th {{
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    padding: 12px 16px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-weight: 400;
}}
td {{
    padding: 12px 16px;
    border-bottom: 1px solid rgba(37,43,59,0.5);
    font-size: 13px;
    vertical-align: middle;
}}
tr:last-child td {{ border-bottom: none; }}
tr:hover td {{ background: rgba(255,255,255,0.02); }}
.td-mono {{ font-family: 'DM Mono', monospace; font-size: 12px; }}
/* ── Badges ── */
.badge {{
    display: inline-block;
    padding: 3px 9px;
    border-radius: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    font-weight: 500;
}}
.badge-helmet  {{ background: rgba(249,115,22,0.15); color: #f97316; }}
.badge-vest    {{ background: rgba(234,179,8,0.15);  color: #eab308; }}
.badge-zone    {{ background: rgba(239,68,68,0.15);  color: #ef4444; }}
.badge-fire    {{ background: rgba(255,45,45,0.15);  color: #ff4444; }}
.badge-high    {{ background: rgba(34,197,94,0.15);  color: #22c55e; }}
.badge-low     {{ background: rgba(90,100,128,0.15); color: #5a6480; }}
.badge-eng     {{ background: rgba(59,130,246,0.15); color: #3b82f6; }}
.badge-worker  {{ background: rgba(34,211,238,0.15); color: #22d3ee; }}
.badge-visitor {{ background: rgba(168,85,247,0.15); color: #a855f7; }}
/* ── Chart wrapper ── */
.chart-wrap {{ padding: 20px; position: relative; height: 260px; }}
.chart-wrap-tall {{ padding: 20px; position: relative; height: 320px; }}
/* ── Person link ── */
a.person-link {{
    color: var(--cyan);
    text-decoration: none;
    font-weight: 500;
}}
a.person-link:hover {{ text-decoration: underline; }}
/* ── Screenshot thumb ── */
.thumb-link {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: rgba(249,115,22,0.1);
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 4px;
    text-decoration: none;
    font-size: 13px;
    transition: all 0.15s;
}}
.thumb-link:hover {{ background: rgba(249,115,22,0.25); }}
/* ── Pill ── */
.pill {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'DM Mono', monospace;
}}
/* ── Filter bar ── */
.filter-bar {{
    display: flex;
    gap: 10px;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    align-items: center;
}}
.filter-bar select, .filter-bar input {{
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 6px 10px;
    border-radius: 5px;
    font-size: 12px;
    font-family: 'DM Sans', sans-serif;
    outline: none;
}}
.filter-bar select:focus, .filter-bar input:focus {{
    border-color: var(--orange);
}}
.btn {{
    padding: 6px 14px;
    border-radius: 5px;
    font-size: 12px;
    cursor: pointer;
    border: none;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    transition: all 0.15s;
}}
.btn-primary {{ background: var(--orange); color: #000; }}
.btn-primary:hover {{ background: #ea6c0a; }}
.btn-ghost {{ background: transparent; color: var(--muted); border: 1px solid var(--border); }}
.btn-ghost:hover {{ color: var(--text); border-color: var(--text); }}
.btn-sm {{ padding: 4px 10px; font-size: 11px; }}
/* ── Empty state ── */
.empty {{
    text-align: center;
    padding: 48px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 12px;
}}
/* ── Grid 2 col ── */
.grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
/* ── Scrollable table ── */
.table-scroll {{ max-height: 420px; overflow-y: auto; }}
/* Scrollbar */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: var(--surface); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>
<aside class="sidebar">
    <div class="sidebar-logo">
        <div class="logo-text">AI Safety Surveillance</div>
        <div class="logo-sub">Surveillance Dashboard</div>
    </div>
    <div class="sidebar-status">
        <div class="status-dot"></div>
        <span>System Active</span>
    </div>
    <nav style="padding-top:8px">
        {nav_html}
    </nav>
    <div class="sidebar-footer">
        MEE . 2026 <br>
        AI Safety System<br><br>
        <br><a href="/reset" style="color:#ef4444;font-size:10px;text-decoration:none">↺ RESET DB</a>
         <a href="/logout" style="color:#5a6480;font-size:10px;text-decoration:none">⏻ LOGOUT</a>

        </a>
    </div>
</aside>
<main class="main">
    {content}
</main>
</body>
</html>''')


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────
def vtype_badge(vtype):
    cls = {'NO_HELMET':'helmet','NO_VEST':'vest','UNAUTH_ZONE':'zone','FIRE':'fire'}.get(vtype,'')
    labels = {'NO_HELMET':'No Helmet','NO_VEST':'No Vest','UNAUTH_ZONE':'Unauth Zone','FIRE':'🔥 Fire'}
    return f'<span class="badge badge-{cls}">{labels.get(vtype, vtype)}</span>'

def role_badge(role):
    cls = {'Engineer':'eng','Worker':'worker','Visitor':'visitor'}.get(role,'')
    return f'<span class="badge badge-{cls}">{role or "—"}</span>'

def conf_badge(conf):
    return f'<span class="badge badge-{"high" if conf=="HIGH" else "low"}">{conf}</span>'

def fmt_dt(dt_str):
    if not dt_str: return '—'
    try: return datetime.fromisoformat(dt_str).strftime('%d %b %H:%M:%S')
    except: return dt_str

def fmt_dur(dur):
    if dur is None: return '—'
    return f'{dur:.1f}s'

def screenshot_link(path):
    if not path: return '—'
    fname = os.path.basename(path)
    return f'<a class="thumb-link" href="/screenshots/{fname}" target="_blank">📷</a>'


# ─────────────────────────────────────────────────────────────────────
#  AUTH — LOGIN / LOGOUT
# ─────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == DASHBOARD_PASSWORD:
            session['authenticated'] = True
            return redirect('/')
        error = 'Wrong password. Try again.'
    err_html = (
        '<div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);'
        'color:#ef4444;padding:9px 12px;border-radius:5px;font-size:11px;'
        'margin-bottom:14px;text-align:center">' + (error or '') + '</div>'
    ) if error else ''
    return (
        '<!DOCTYPE html><html><head>'
        '<meta charset="UTF-8"><title> AI Safety Surveillance Login</title>'
        '<style>'
        '*{margin:0;padding:0;box-sizing:border-box}'
        'body{background:#0b0d12;display:flex;align-items:center;'
        'justify-content:center;min-height:100vh;font-family:monospace}'
        '.box{background:#12151d;border:1px solid #252b3b;border-radius:10px;'
        'padding:40px 36px;width:320px}'
        '.logo{font-size:22px;font-weight:700;color:#f97316;letter-spacing:3px;'
        'text-align:center;margin-bottom:4px}'
        '.sub{font-size:10px;color:#5a6480;text-align:center;'
        'margin-bottom:28px;letter-spacing:1px}'
        'label{font-size:10px;letter-spacing:2px;color:#5a6480;'
        'text-transform:uppercase;display:block;margin-bottom:6px}'
        'input{width:100%;background:#0b0d12;border:1px solid #252b3b;'
        'color:#dde3f0;padding:10px 12px;border-radius:5px;'
        'font-size:13px;outline:none;margin-bottom:18px}'
        'input:focus{border-color:#f97316}'
        'button{width:100%;background:#f97316;color:#000;border:none;'
        'padding:11px;border-radius:5px;font-size:15px;font-weight:700;'
        'letter-spacing:2px;cursor:pointer;text-transform:uppercase}'
        'button:hover{background:#ea6c0a}'
        '</style></head><body>'
        '<div class="box">'
        '<div class="logo">AI Safety Surveillance</div>'
        '<div class="sub">SURVEILLANCE DASHBOARD</div>'
        + err_html +
        '<form method="post">'
        '<label>Password</label>'
        '<input type="password" name="password" autofocus>'
        '<button type="submit">Enter</button>'
        '</form></div></body></html>'
    )
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# ─────────────────────────────────────────────────────────────────────
#  AUTH DECORATOR
# ─────────────────────────────────────────────────────────────────────
from functools import wraps

def login_required(f):
    """Redirects to /login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('authenticated'):
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────────────────────────────
#  ROUTE: / — OVERVIEW
# ─────────────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def overview():
    today_viols  = db.get_violations_today()
    attendance   = db.get_attendance_today()
    all_persons  = db.get_all_persons()

    total_today  = len(today_viols)
    present      = len(attendance)
    active_now   = sum(1 for v in today_viols if not v['ended_at'])

    # Most common type today
    if today_viols:
        from collections import Counter
        types = Counter(v['violation_type'] for v in today_viols)
        top_type = types.most_common(1)[0][0]
        top_label = {'NO_HELMET':'No Helmet','NO_VEST':'No Vest',
                     'UNAUTH_ZONE':'Unauth Zone','FIRE':'Fire'}.get(top_type, top_type)
    else:
        top_label = 'None'

    # Stats cards
    cards = f'''
    <div class="cards cards-4">
        <div class="card card-orange">
            <div class="card-label">Violations Today</div>
            <div class="card-value">{total_today}</div>
            <div class="card-sub">logged events</div>
        </div>
        <div class="card card-red">
            <div class="card-label">Active Now</div>
            <div class="card-value">{active_now}</div>
            <div class="card-sub">no end time yet</div>
        </div>
        <div class="card card-green">
            <div class="card-label">Present Today</div>
            <div class="card-value">{present}</div>
            <div class="card-sub">of {len(all_persons)} registered</div>
        </div>
        <div class="card card-yellow">
            <div class="card-label">Top Violation</div>
            <div class="card-value" style="font-size:22px;padding-top:6px">{top_label}</div>
            <div class="card-sub">most common today</div>
        </div>
    </div>'''

    # Today's violation feed
    rows = ''
    for v in today_viols[:50]:
        pname = v['person_name'] or '<span style="color:var(--muted)">Unknown</span>'
        if v['person_id'] is not None:
            pname = f'<a class="person-link" href="/person/{v["person_id"]}">{v["person_name"]}</a>'
        rows += f'''<tr>
            <td class="td-mono">{fmt_dt(v["started_at"])}</td>
            <td>{pname}</td>
            <td>{vtype_badge(v["violation_type"])}</td>
            <td style="color:var(--muted)">{v["zone_name"] or "—"}</td>
            <td class="td-mono">{fmt_dur(v["duration_sec"])}</td>
            <td>{conf_badge(v["confidence"])}</td>
            <td>{screenshot_link(v["screenshot_path"])}</td>
        </tr>'''
    if not rows:
        rows = '<tr><td colspan="7" class="empty">No violations recorded today</td></tr>'

    violations_table = f'''
    <div class="section" style="margin-bottom:24px">
        <div class="section-head">
            <span class="section-title">Today\'s Violations</span>
            <span style="font-family:\'DM Mono\',monospace;font-size:11px;color:var(--muted)">
                AUTO-REFRESH 5s
            </span>
        </div>
        <div class="table-scroll">
        <table>
            <thead><tr>
                <th>Time</th><th>Person</th><th>Type</th>
                <th>Zone</th><th>Duration</th><th>Confidence</th><th>Screenshot</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
    </div>'''

    # Attendance
    att_rows = ''
    for a in attendance:
        att_rows += f'''<tr>
            <td><a class="person-link" href="/person/{a["person_id"]}">{a["person_name"]}</a></td>
            <td class="td-mono">{a["first_seen"][11:19]}</td>
        </tr>'''
    if not att_rows:
        att_rows = '<tr><td colspan="2" class="empty">No one detected today yet</td></tr>'

    attendance_section = f'''
    <div class="section">
        <div class="section-head"><span class="section-title">Today\'s Attendance</span></div>
        <table>
            <thead><tr><th>Person</th><th>First Seen</th></tr></thead>
            <tbody>{att_rows}</tbody>
        </table>
    </div>'''

    content = f'''
    <div class="page-header">
        <div class="page-title">Overview</div>
        <div class="page-sub">{datetime.now().strftime("%A, %d %B %Y")}</div>
    </div>
    {cards}
    {violations_table}
    {attendance_section}
    <script>setTimeout(()=>location.reload(), 5000);</script>'''

    return base('Overview', content, 'home')


# ─────────────────────────────────────────────────────────────────────
#  ROUTE: /persons — ALL WORKERS
# ─────────────────────────────────────────────────────────────────────
@app.route('/persons')
@login_required
def persons():
    all_persons = db.get_all_persons()
    today = datetime.now().strftime('%Y-%m-%d')

    rows = ''
    for p in all_persons:
        stats  = db.get_person_stats(p['aruco_id'])
        today_v = len([v for v in db.get_violations_today()
                       if v['person_id'] == p['aruco_id']])
        att_today = db.get_attendance_by_date(today)
        present = any(a['person_id'] == p['aruco_id'] for a in att_today)
        dot = f'<span style="color:var(--green)">●</span>' if present else \
              f'<span style="color:var(--muted)">○</span>'

        rows += f'''<tr>
            <td class="td-mono" style="color:var(--muted)">{p["aruco_id"]}</td>
            <td>{dot} <a class="person-link" href="/person/{p["aruco_id"]}">{p["name"]}</a></td>
            <td>{role_badge(p["role"])}</td>
            <td style="color:{"var(--red)" if today_v > 0 else "var(--muted)"};font-family:\'DM Mono\',monospace">
                {today_v}
            </td>
            <td class="td-mono">{stats["total"]}</td>
            <td class="td-mono">{stats["days_present"]}</td>
            <td class="td-mono">{stats["avg_duration"]}s</td>
            <td><a href="/person/{p["aruco_id"]}" class="btn btn-ghost btn-sm">View →</a></td>
        </tr>'''

    content = f'''
    <div class="page-header">
        <div class="page-title">Workers</div>
        <div class="page-sub">{len(all_persons)} registered persons</div>
    </div>
    <div class="section">
        <div class="section-head"><span class="section-title">All Registered Workers</span></div>
        <table>
            <thead><tr>
                <th>ID</th><th>Name</th><th>Role</th>
                <th>Today</th><th>Total</th><th>Days Present</th><th>Avg Duration</th><th></th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
    </div>'''

    return base('Workers', content, 'persons')


# ─────────────────────────────────────────────────────────────────────
#  ROUTE: /person/<id> — INDIVIDUAL PROFILE
# ─────────────────────────────────────────────────────────────────────
@app.route('/person/<int:pid>')
@login_required
def person_profile(pid):
    p = db.get_person(pid)
    if not p:
        return redirect('/')

    stats    = db.get_person_stats(pid)
    viols    = db.get_violations_by_person(pid, limit=100)
    att_log  = db.get_attendance_by_person(pid, limit=30)
    weekly   = db.get_weekly_summary()

    # Per-day violation counts for this person (last 30 days)
    from collections import defaultdict
    daily_counts = defaultdict(int)
    for v in viols:
        day = v['started_at'][:10]
        daily_counts[day] += 1

    days_30 = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(29,-1,-1)]
    daily_labels = [d[5:] for d in days_30]  # MM-DD
    daily_data   = [daily_counts.get(d, 0) for d in days_30]

    # Type breakdown for pie
    type_labels = list(stats['by_type'].keys())
    type_data   = list(stats['by_type'].values())
    type_colors = [VIOLATION_COLORS.get(t, '#666') for t in type_labels]

    # Stats cards
    cards = f'''
    <div class="cards cards-4">
        <div class="card card-orange">
            <div class="card-label">Total Violations</div>
            <div class="card-value">{stats["total"]}</div>
        </div>
        <div class="card card-cyan">
            <div class="card-label">Days Present</div>
            <div class="card-value">{stats["days_present"]}</div>
        </div>
        <div class="card card-yellow">
            <div class="card-label">Avg Duration</div>
            <div class="card-value">{stats["avg_duration"]}s</div>
        </div>
        <div class="card card-green">
            <div class="card-label">Role</div>
            <div class="card-value" style="font-size:22px;padding-top:6px">{p["role"]}</div>
        </div>
    </div>'''

    # Charts row
    charts = f'''
    <div class="grid-2" style="margin-bottom:24px">
        <div class="section">
            <div class="section-head"><span class="section-title">Daily Violations (30 days)</span></div>
            <div class="chart-wrap">
                <canvas id="dailyChart"></canvas>
            </div>
        </div>
        <div class="section">
            <div class="section-head"><span class="section-title">Violation Types</span></div>
            <div class="chart-wrap">
                <canvas id="typeChart"></canvas>
            </div>
        </div>
    </div>'''

    # Violation history table
    vrows = ''
    for v in viols:
        vrows += f'''<tr>
            <td class="td-mono">{fmt_dt(v["started_at"])}</td>
            <td>{vtype_badge(v["violation_type"])}</td>
            <td style="color:var(--muted)">{v["zone_name"] or "—"}</td>
            <td class="td-mono">{fmt_dur(v["duration_sec"])}</td>
            <td style="color:var(--muted);font-size:12px">{v["camera_position"] or "—"}</td>
            <td>{conf_badge(v["confidence"])}</td>
            <td>{screenshot_link(v["screenshot_path"])}</td>
        </tr>'''
    if not vrows:
        vrows = '<tr><td colspan="7" class="empty">No violations recorded</td></tr>'

    att_rows = ''
    for a in att_log:
        att_rows += f'''<tr>
            <td class="td-mono">{a["work_date"]}</td>
            <td class="td-mono">{a["first_seen"][11:19]}</td>
        </tr>'''

    content = f'''
    <div class="page-header" style="display:flex;align-items:center;justify-content:space-between">
        <div>
            <div class="page-title">{p["name"]}</div>
            <div class="page-sub">ArUco ID {pid} · {p["role"]}</div>
        </div>
        <a href="/persons" class="btn btn-ghost">← Back</a>
    </div>
    {cards}
    {charts}
    <div class="grid-2">
        <div class="section">
            <div class="section-head"><span class="section-title">Violation History</span></div>
            <div class="table-scroll">
            <table>
                <thead><tr><th>Time</th><th>Type</th><th>Zone</th>
                    <th>Duration</th><th>Camera</th><th>Confidence</th><th>Screenshot</th></tr></thead>
                <tbody>{vrows}</tbody>
            </table>
            </div>
        </div>
        <div class="section">
            <div class="section-head"><span class="section-title">Attendance Log</span></div>
            <div class="table-scroll">
            <table>
                <thead><tr><th>Date</th><th>Arrived</th></tr></thead>
                <tbody>{att_rows}</tbody>
            </table>
            </div>
        </div>
    </div>

    <script>
    const orange = '#f97316', yellow = '#eab308', red = '#ef4444',
          cyan = '#22d3ee', muted = '#5a6480', bg = '#12151d', border = '#252b3b';
    Chart.defaults.color = '#5a6480';
    Chart.defaults.font.family = "'DM Mono', monospace";

    new Chart(document.getElementById('dailyChart'), {{
        type: 'bar',
        data: {{
            labels: {daily_labels},
            datasets: [{{
                label: 'Violations',
                data: {daily_data},
                backgroundColor: 'rgba(249,115,22,0.5)',
                borderColor: orange,
                borderWidth: 1,
                borderRadius: 3,
            }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ grid: {{ color: border }}, ticks: {{ maxTicksLimit: 8 }} }},
                y: {{ grid: {{ color: border }}, beginAtZero: true, ticks: {{ stepSize: 1 }} }}
            }}
        }}
    }});

    {"new Chart(document.getElementById('typeChart'), {type:'doughnut',data:{labels:" + str(type_labels) + ",datasets:[{data:" + str(type_data) + ",backgroundColor:" + str(type_colors) + ",borderWidth:0,hoverOffset:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom',labels:{padding:16,boxWidth:12}}}}});" if type_data else ""}
    </script>'''

    return base(p['name'], content, 'persons')


# ─────────────────────────────────────────────────────────────────────
#  ROUTE: /violations — FULL LOG
# ─────────────────────────────────────────────────────────────────────
@app.route('/violations')
@login_required
def violations():
    vtype  = request.args.get('type', '')
    conf   = request.args.get('conf', '')
    person = request.args.get('person', '')

    all_v = db.get_all_violations(limit=300)

    # Filter
    if vtype:   all_v = [v for v in all_v if v['violation_type'] == vtype]
    if conf:    all_v = [v for v in all_v if v['confidence'] == conf]
    if person:  all_v = [v for v in all_v if (v['person_name'] or '').lower().find(person.lower()) >= 0]

    rows = ''
    for v in all_v:
        pname = v['person_name'] or '<span style="color:var(--muted)">Unknown</span>'
        if v['person_id'] is not None:
            pname = f'<a class="person-link" href="/person/{v["person_id"]}">{v["person_name"]}</a>'
        rows += f'''<tr>
            <td class="td-mono" style="color:var(--muted)">{v["id"]}</td>
            <td class="td-mono">{fmt_dt(v["started_at"])}</td>
            <td>{pname}</td>
            <td>{vtype_badge(v["violation_type"])}</td>
            <td style="color:var(--muted)">{v["zone_name"] or "—"}</td>
            <td class="td-mono">{fmt_dur(v["duration_sec"])}</td>
            <td style="color:var(--muted);font-size:12px">{v["camera_position"] or "—"}</td>
            <td>{conf_badge(v["confidence"])}</td>
            <td>{screenshot_link(v["screenshot_path"])}</td>
        </tr>'''
    if not rows:
        rows = '<tr><td colspan="9" class="empty">No violations match filters</td></tr>'

    content = f'''
    <div class="page-header">
        <div class="page-title">Violations</div>
        <div class="page-sub">{len(all_v)} records shown</div>
    </div>
    <div class="section">
        <form method="get" class="filter-bar">
            <select name="type">
                <option value="">All Types</option>
                <option value="NO_HELMET" {"selected" if vtype=="NO_HELMET" else ""}>No Helmet</option>
                <option value="NO_VEST"   {"selected" if vtype=="NO_VEST"   else ""}>No Vest</option>
                <option value="UNAUTH_ZONE" {"selected" if vtype=="UNAUTH_ZONE" else ""}>Unauth Zone</option>
                <option value="FIRE"      {"selected" if vtype=="FIRE"      else ""}>Fire</option>
            </select>
            <select name="conf">
                <option value="">All Confidence</option>
                <option value="HIGH" {"selected" if conf=="HIGH" else ""}>HIGH</option>
                <option value="LOW"  {"selected" if conf=="LOW"  else ""}>LOW</option>
            </select>
            <input type="text" name="person" placeholder="Person name..." value="{person}" style="width:160px">
            <button type="submit" class="btn btn-primary">Filter</button>
            <a href="/violations" class="btn btn-ghost">Reset</a>
        </form>
        <div class="table-scroll" style="max-height:520px">
        <table>
            <thead><tr>
                <th>ID</th><th>Time</th><th>Person</th><th>Type</th>
                <th>Zone</th><th>Duration</th><th>Camera</th><th>Confidence</th><th>Screenshot</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
    </div>'''

    return base('Violations', content, 'violations')


# ─────────────────────────────────────────────────────────────────────
#  ROUTE: /charts — TRENDS
# ─────────────────────────────────────────────────────────────────────
@app.route('/charts')
@login_required
def charts():
    weekly  = db.get_weekly_summary()
    monthly = db.get_monthly_summary()
    persons = db.get_all_persons()

    # Weekly — violations per day total
    from collections import defaultdict
    days_7 = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(6,-1,-1)]
    day_labels_7 = [(datetime.now() - timedelta(days=i)).strftime('%a %d') for i in range(6,-1,-1)]

    weekly_totals = defaultdict(int)
    for w in weekly:
        weekly_totals[w['summary_date']] += w['count']
    weekly_data = [weekly_totals.get(d, 0) for d in days_7]

    # Monthly — per person per day
    person_colors = ['#f97316','#22d3ee','#22c55e','#a855f7','#eab308','#ef4444']
    days_30 = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(29,-1,-1)]
    day_labels_30 = [(datetime.now() - timedelta(days=i)).strftime('%d') for i in range(29,-1,-1)]

    person_datasets = []
    for idx, p in enumerate(persons):
        daily = defaultdict(int)
        for m in monthly:
            if m['person_id'] == p['aruco_id']:
                daily[m['summary_date']] += m['count']
        data = [daily.get(d, 0) for d in days_30]
        col = person_colors[idx % len(person_colors)]
        person_datasets.append({
            'label': p['name'],
            'data': data,
            'borderColor': col,
            'backgroundColor': col.replace(')', ',0.1)').replace('rgb','rgba') if 'rgb' in col else col + '1a',
            'tension': 0.4,
            'pointRadius': 3,
        })

    import json
    person_ds_json = json.dumps(person_datasets)

    # Violation type breakdown total
    all_v = db.get_all_violations(limit=1000)
    type_totals = defaultdict(int)
    for v in all_v:
        type_totals[v['violation_type']] += 1
    type_labels = list(type_totals.keys())
    type_data   = [type_totals[t] for t in type_labels]
    type_colors_list = [VIOLATION_COLORS.get(t, '#666') for t in type_labels]

    # Time of day heatmap data
    hour_counts = [0]*24
    for v in all_v:
        if v['started_at']:
            try:
                h = int(v['started_at'][11:13])
                hour_counts[h] += 1
            except: pass

    content = f'''
    <div class="page-header">
        <div class="page-title">Trends</div>
        <div class="page-sub">Weekly and monthly violation analysis</div>
    </div>

    <div class="grid-2" style="margin-bottom:24px">
        <div class="section">
            <div class="section-head"><span class="section-title">This Week — Total Violations</span></div>
            <div class="chart-wrap">
                <canvas id="weekChart"></canvas>
            </div>
        </div>
        <div class="section">
            <div class="section-head"><span class="section-title">All Time — By Type</span></div>
            <div class="chart-wrap">
                <canvas id="typeTotal"></canvas>
            </div>
        </div>
    </div>

    <div class="section" style="margin-bottom:24px">
        <div class="section-head"><span class="section-title">30-Day Trend Per Person (Improvement Over Time)</span></div>
        <div class="chart-wrap-tall">
            <canvas id="monthlyChart"></canvas>
        </div>
    </div>

    <div class="section">
        <div class="section-head"><span class="section-title">Violations by Hour of Day</span></div>
        <div class="chart-wrap">
            <canvas id="hourChart"></canvas>
        </div>
    </div>

    <script>
    Chart.defaults.color = '#5a6480';
    Chart.defaults.font.family = "'DM Mono', monospace";
    const border = '#252b3b';

    new Chart(document.getElementById('weekChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps(day_labels_7)},
            datasets: [{{
                label: 'Violations',
                data: {json.dumps(weekly_data)},
                backgroundColor: 'rgba(249,115,22,0.6)',
                borderColor: '#f97316',
                borderWidth: 1,
                borderRadius: 4,
            }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ grid: {{ color: border }} }},
                y: {{ grid: {{ color: border }}, beginAtZero: true, ticks: {{ stepSize: 1 }} }}
            }}
        }}
    }});

    new Chart(document.getElementById('typeTotal'), {{
        type: 'doughnut',
        data: {{
            labels: {json.dumps(type_labels)},
            datasets: [{{
                data: {json.dumps(type_data)},
                backgroundColor: {json.dumps(type_colors_list)},
                borderWidth: 0,
                hoverOffset: 8,
            }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 16, boxWidth: 12 }} }} }}
        }}
    }});

    new Chart(document.getElementById('monthlyChart'), {{
        type: 'line',
        data: {{
            labels: {json.dumps(day_labels_30)},
            datasets: {person_ds_json}
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ position: 'bottom', labels: {{ padding: 16, boxWidth: 12 }} }} }},
            scales: {{
                x: {{ grid: {{ color: border }} }},
                y: {{ grid: {{ color: border }}, beginAtZero: true, ticks: {{ stepSize: 1 }} }}
            }}
        }}
    }});

    new Chart(document.getElementById('hourChart'), {{
        type: 'bar',
        data: {{
            labels: {json.dumps([f'{h:02d}:00' for h in range(24)])},
            datasets: [{{
                label: 'Violations',
                data: {json.dumps(hour_counts)},
                backgroundColor: 'rgba(34,211,238,0.5)',
                borderColor: '#22d3ee',
                borderWidth: 1,
                borderRadius: 3,
            }}]
        }},
        options: {{
            responsive: true, maintainAspectRatio: false,
            plugins: {{ legend: {{ display: false }} }},
            scales: {{
                x: {{ grid: {{ color: border }} }},
                y: {{ grid: {{ color: border }}, beginAtZero: true, ticks: {{ stepSize: 1 }} }}
            }}
        }}
    }});
    </script>'''

    return base('Trends', content, 'charts')


# ─────────────────────────────────────────────────────────────────────
#  ROUTE: /review — MANUAL REVIEW QUEUE
# ─────────────────────────────────────────────────────────────────────
@app.route('/review', methods=['GET','POST'])
@login_required
def review():
    persons = db.get_all_persons()

    if request.method == 'POST':
        vid = int(request.form.get('violation_id', 0))
        pid = request.form.get('person_id', '')
        pid = int(pid) if pid.isdigit() else None
        db.mark_reviewed(vid, pid)
        return redirect('/review')

    unreviewed = db.get_unreviewed()

    person_options = '<option value="">— Unknown —</option>'
    for p in persons:
        person_options += f'<option value="{p["aruco_id"]}">{p["name"]} ({p["role"]})</option>'

    rows = ''
    for v in unreviewed:
        rows += f'''<tr>
            <td class="td-mono" style="color:var(--muted)">{v["id"]}</td>
            <td class="td-mono">{fmt_dt(v["started_at"])}</td>
            <td>{vtype_badge(v["violation_type"])}</td>
            <td style="color:var(--muted)">{v["zone_name"] or "—"}</td>
            <td class="td-mono">{fmt_dur(v["duration_sec"])}</td>
            <td>{conf_badge(v["confidence"])}</td>
            <td>{screenshot_link(v["screenshot_path"])}</td>
            <td>
                <form method="post" style="display:flex;gap:6px;align-items:center">
                    <input type="hidden" name="violation_id" value="{v["id"]}">
                    <select name="person_id" style="background:var(--surface);border:1px solid var(--border);
                            color:var(--text);padding:4px 8px;border-radius:4px;font-size:11px">
                        {person_options}
                    </select>
                    <button type="submit" class="btn btn-primary btn-sm">✓</button>
                </form>
            </td>
        </tr>'''
    if not rows:
        rows = '<tr><td colspan="8" class="empty">All violations reviewed ✓</td></tr>'

    content = f'''
    <div class="page-header">
        <div class="page-title">Manual Review</div>
        <div class="page-sub">{len(unreviewed)} violations need review</div>
    </div>
    <div class="section">
        <div class="section-head">
            <span class="section-title">Unreviewed — Unknown or Low Confidence</span>
        </div>
        <div class="table-scroll">
        <table>
            <thead><tr>
                <th>ID</th><th>Time</th><th>Type</th><th>Zone</th>
                <th>Duration</th><th>Confidence</th><th>Screenshot</th><th>Assign To</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
    </div>'''

    return base('Review', content, 'review')

# ─────────────────────────────────────────────────────────────────────
#  Delete all violation, attendance and summary data (keep persons)
# ─────────────────────────────────────────────────────────────────────
@app.route('/reset', methods=['GET', 'POST'])
@login_required
def reset_db():
    if request.method == 'POST':
        db.reset_all_data()
        return redirect('/')
    # Confirmation page
    html = (
        '<!DOCTYPE html><html><head>'
        '<meta charset="UTF-8"><title>Reset Database</title>'
        '<style>'
        '*{margin:0;padding:0;box-sizing:border-box}'
        'body{background:#0b0d12;display:flex;align-items:center;'
        'justify-content:center;min-height:100vh;font-family:monospace;color:#dde3f0}'
        '.box{background:#12151d;border:1px solid #252b3b;border-radius:10px;'
        'padding:40px 36px;width:380px;text-align:center}'
        '.title{font-size:20px;font-weight:700;color:#ef4444;margin-bottom:12px}'
        '.msg{font-size:13px;color:#5a6480;margin-bottom:28px;line-height:1.6}'
        '.btn-red{background:#ef4444;color:#fff;border:none;padding:11px 24px;'
        'border-radius:5px;font-size:14px;font-weight:700;cursor:pointer;margin-right:10px}'
        '.btn-red:hover{background:#dc2626}'
        '.btn-back{background:transparent;color:#5a6480;border:1px solid #252b3b;'
        'padding:11px 24px;border-radius:5px;font-size:14px;cursor:pointer;text-decoration:none}'
        '</style></head><body>'
        '<div class="box">'
        '<div class="title">⚠ Reset Database</div>'
        '<div class="msg">This will permanently delete all violations, attendance and summary data.<br><br>Persons registry will be kept.</div>'
        '<form method="post" style="display:inline">'
        '<button type="submit" class="btn-red">Yes, Reset Everything</button>'
        '</form>'
        '<a href="/" class="btn-back">Cancel</a>'
        '</div></body></html>'
    )
    return html

# ─────────────────────────────────────────────────────────────────────
#  ROUTE: /screenshots/<filename>
# ─────────────────────────────────────────────────────────────────────
@app.route('/screenshots/<path:filename>')
def serve_screenshot(filename):
    return send_from_directory(SCREENSHOT_DIR, filename)


# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 50)
    print("  AI Safety Surveillance Dashboard")
    print("  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)