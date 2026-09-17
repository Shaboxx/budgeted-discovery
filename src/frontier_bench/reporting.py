"""Offline HTML/SVG reports rebuilt solely from persisted artifacts."""
from __future__ import annotations
from pathlib import Path
from html import escape
import json
import hashlib
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from .runner import write_json, read_jsonl
from .evaluation import paired_summary

COLORS=["#267ca4","#aa4455","#558844","#995fa8","#c17f24","#188a87","#555d70"]

def validate_raw_run(path: Path):
    manifest=json.loads((path/"manifest.json").read_text())
    required={"summary.json","evaluation.jsonl","decisions.jsonl","executions.jsonl","outcomes.jsonl","initial.json","visible-final.json","curves.parquet"}
    if not required<=set(manifest["artifacts"]): raise ValueError("missing required raw artifact checksum")
    for name,expected in manifest["artifacts"].items():
        if Path(name).name!=name: raise ValueError("unsafe artifact name")
        artifact=path/name
        if not artifact.exists(): raise ValueError(f"missing raw artifact: {name}")
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=expected: raise ValueError(f"raw artifact checksum mismatch: {name}")
    return manifest

def chart(curves, axis):
    if not curves: return "<p>No completed curves.</p>"
    xmax=max([float(p.get(axis,0)) for _,curve in curves for p in curve]+[1.])
    ymax=max([p["reference_relevant_accounts"] for _,curve in curves for p in curve]+[1])
    parts=['<svg viewBox="0 0 720 280" role="img" aria-label="Reference-relevant discoveries by '+axis+'">', '<path d="M50 20V230H690" fill="none" stroke="#778"/>']
    for index,(label,curve) in enumerate(curves):
        points=" ".join(f'{50+float(p[axis])/xmax*640:.2f},{230-p["reference_relevant_accounts"]/ymax*205:.2f}' for p in curve)
        color=COLORS[index%len(COLORS)]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"><title>{escape(label)}</title></polyline>')
    parts.extend([f'<text x="48" y="250">0</text><text x="645" y="250">{xmax:g} {axis}</text>',f'<text x="8" y="26">{ymax}</text><text x="65" y="16">New reference-relevant accounts</text>','</svg>'])
    parts.append('<div class="legend">'+' '.join(f'<span style="color:{COLORS[i%7]}">{escape(label)}</span>' for i,(label,_) in enumerate(curves))+'</div>')
    return "".join(parts)

def report(root: str|Path) -> Path:
    root=Path(root)
    if (root/"comparison.json").exists():
        comparison=json.loads((root/"comparison.json").read_text()); summaries=[]
        failed_setup={f["job"] for f in comparison.get("failed",[])}
        for directory in sorted((root/"runs").iterdir()) if (root/"runs").exists() else []:
            if not directory.is_dir() or directory.name in failed_setup: continue
            validate_raw_run(directory)
            summaries.append({**json.loads((directory/"summary.json").read_text()),"run_path":directory.relative_to(root).as_posix()})
        if len(summaries)!=comparison["executed"]: raise ValueError("missing recorded run: executed artifact count mismatch")
        write_json(root/"summaries.json",summaries)
        if summaries: pq.write_table(pa.Table.from_pylist(summaries),root/"summaries.parquet")
        cfg=json.loads((root/"resolved-config.json").read_text())
        write_json(root/"statistics.json",paired_summary(summaries,samples=cfg["evaluation"]["bootstrap_samples"],confidence=cfg["evaluation"]["confidence"],seed=cfg["seed"]))
    elif (root/"summary.json").exists():
        validate_raw_run(root)
        summaries=[{**json.loads((root/"summary.json").read_text()),"run_path":"."}]; comparison={"status":summaries[0]["status"],"planned":1,"executed":1,"skipped":[],"failed":[]}
    else: raise ValueError("No recorded run/compare artifacts found")
    # DuckDB actually reads saved Parquet, not a separately invented summary.
    aggregations=[]
    if (root/"summaries.parquet").exists():
        with duckdb.connect() as conn:
            query="SELECT policy, count(*) AS runs, avg(reference_relevant_accounts) AS mean_relevant, avg(cpu_seconds) AS mean_cpu_seconds FROM read_parquet(?) WHERE status='completed' GROUP BY policy ORDER BY policy"
            cols=None; cursor=conn.execute(query,[str(root/"summaries.parquet")]); cols=[d[0] for d in cursor.description]
            aggregations=[dict(zip(cols,row)) for row in cursor.fetchall()]
        write_json(root/"derived-aggregate.json",aggregations)
    columns=[("policy","Policy"),("condition","Condition"),("arrangement","Target arrangement"),("seed","Seed"),("reference_relevant_accounts","Relevant reached"),("requests","Calls"),("cost","Cost"),("assessed_accounts","Assessed"),("pending_accounts","Pending"),("target_regions_reached","Regions"),("duplicate_rate","Duplicate rate"),("zero_rate","Zero rate"),("revisit_new_posts","Fresh revisit posts"),("cpu_seconds","CPU s"),("peak_rss_mb","Peak RSS MiB"),("status","Status")]
    def fmt(value):
        if value is None: return "unknown"
        return f"{value:.3f}" if isinstance(value,float) else str(value)
    table='<div class="table"><table><thead><tr>'+''.join(f'<th>{label}</th>' for _,label in columns)+'</tr></thead><tbody>'
    for row in summaries:
        table+='<tr>'+''.join('<td>'+escape(fmt(row.get(key)))+'</td>' for key,_ in columns)+'</tr>'
    table+='</tbody></table></div>'
    # One matched block per chart prevents averaging unmatched traces or hundreds
    # of unreadable curves; all other raw curves remain individually linked.
    exemplar=[]
    if summaries:
        first=summaries[0]; block=(first["seed"],first["condition"],first["arrangement"])
        for row in summaries:
            if (row["seed"],row["condition"],row["arrangement"])==block:
                path=root/row["run_path"]/"evaluation.jsonl"
                if path.exists(): exemplar.append((row["policy"],read_jsonl(path)))
    statistics=json.loads((root/"statistics.json").read_text()) if (root/"statistics.json").exists() else {}
    measurement_path=root/"measurement"/"measurement-diagnostics.json"
    measurement=json.loads(measurement_path.read_text()) if measurement_path.exists() else None
    links=''.join(f'<li><a href="{escape(row["run_path"],quote=True)}/manifest.json">{escape(row["run_path"])}</a> — manifest, JSONL receipts and Parquet curves in the same directory.</li>' for row in summaries)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Discovery Frontier research report</title>
<style>body{{font:16px/1.5 system-ui;background:#f5f7fa;color:#213247;max-width:1500px;margin:32px auto;padding:0 28px}}h1,h2{{line-height:1.2}}.note{{padding:18px;background:#e5eef4;border-left:4px solid #267ca4}}.table{{overflow:auto;background:white}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #d5dce3;text-align:left}}th{{background:#e5eef4}}svg{{width:100%;max-width:720px;background:white}}.charts{{display:flex;flex-wrap:wrap}}.charts>div{{flex:1;min-width:340px}}.legend span{{margin-right:15px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:16px}}a{{color:#126591}}</style>
<h1>Discovery Frontier: recorded experiment</h1><p class="note">Status: <strong>{escape(str(comparison['status']))}</strong>. {comparison['executed']} executed of {comparison['planned']} planned. Primary outcome: new unique reference-relevant accounts, excluding the common declared initial exposure. This is a bounded infrastructure demonstration, not evidence of general superiority.</p>
<p>Truth appears only in this post-run report. Policies receive separate observations and operational estimates. Assessed and pending are feedback states, not proof of true relevance. RSS is sampled process memory and may include caches retained across runs.</p>
<h2>Discovery against acquisition budget</h2><p>One paired world block is shown below; all runs are reported in the table and linked artifacts. Lines include WAIT steps with unchanged acquisition cost.</p><div class="charts"><div>{chart(exemplar,'requests')}</div><div>{chart(exemplar,'cost')}</div></div>
<h2>Every recorded run</h2>{table}<h2>Paired comparisons</h2><p>Intervals resample independent seed blocks, preserving condition and policy pairing. Missing pairs are reported. Very few seeds give descriptive intervals only.</p><pre>{escape(json.dumps(statistics,indent=2))}</pre>
<h2>Measurement diagnostic</h2><p>Discovery and temporal measurement are separate modules. Fixed-panel estimates are evaluated against the same panel's activity. Exact fixtures check logic; they do not estimate stochastic false-alarm rates.</p><pre>{escape(json.dumps(measurement,indent=2)) if measurement else 'Not included in this invocation.'}</pre>
<h2>Failures and resource exclusions</h2><pre>{escape(json.dumps({'failed':comparison.get('failed',[]),'skipped':comparison.get('skipped',[])},indent=2))}</pre>
<h2>Inspectable raw artifacts</h2><ul>{links}</ul><p>This file contains no external scripts, fonts, network requests, private production data, or attached papers.</p></html>'''
    destination=root/"report.html"; destination.write_text(body,encoding="utf-8"); return destination
