#!/usr/bin/env python3
"""
Add untrained-model comparison to simulation_3d.html.

Loads controller_lesson_0.pth, runs it from the exact same starting
positions as the trained trajectories already in the HTML, then injects:
  - const TRAJS_UNTRAINED = [...]  (raw data)
  - A second red truck mesh driven by the untrained data
  - Red trail dots for the untrained truck
  - Floating "TRAINED" / "UNTRAINED" billboard labels above each truck

Usage:
    python add_comparison.py
    python add_comparison.py --untrained_lesson 0 --max_steps 400
    python add_comparison.py --html simulation_3d.html
"""

import torch
import math
import json
import re
import os
import argparse


# ── Truck kinematics (must match simulate-3d.py) ──────────────────────────────

class TruckSim:
    L = 4.0; D = 4.0; S = -0.1

    def step(self, x, y, t0, t1, phi):
        x  += self.S * math.cos(t0)
        y  += self.S * math.sin(t0)
        t0 += self.S / self.L * math.tan(phi)
        t1 += self.S / self.D * math.sin(t0 - t1)
        return x, y, t0, t1

    def trailer_xy(self, x, y, t1):
        return x - self.D * math.cos(t1), y - self.D * math.sin(t1)

    def is_jackknifed(self, t0, t1):
        d = abs(math.degrees(t0 - t1))
        return min(d, abs(d - 360)) > 90

    def run_from(self, controller, x0, y0, t0_0, t1_0, phi0, max_steps=400):
        x, y, t0, t1 = x0, y0, t0_0, t1_0
        phi_t = torch.tensor([phi0], dtype=torch.float32)
        frames = []
        for _ in range(max_steps):
            if self.is_jackknifed(t0, t1):
                break
            tx, ty = self.trailer_xy(x, y, t1)
            frames.append({
                "x":  round(x,  4), "y":  round(y,  4),
                "t0": round(t0, 4), "t1": round(t1, 4),
                "phi": round(float(phi_t), 4),
                "tx": round(tx, 4), "ty": round(ty, 4),
            })
            with torch.no_grad():
                state = torch.tensor([x, y, t0, t1], dtype=torch.float32)
                phi_t = controller(torch.cat((phi_t, state)))
            x, y, t0, t1 = self.step(x, y, t0, t1, float(phi_t))
        return frames


# ── JavaScript injected into the HTML ────────────────────────────────────────

SECOND_TRUCK_JS = """
// @@COMPARISON_START@@
// ═══════════════════════════════════════════════════════════════════════════
// COMPARISON: Live ghost trucks — untrained model spawns every {branch_every} frames
// ═══════════════════════════════════════════════════════════════════════════

const _BRANCH_EVERY = {branch_every};

// Root group — toggling visible hides/shows all ghost trucks at once
const _ghostRoot = new THREE.Group();
scene.add(_ghostRoot);

const _liveGhosts  = [];   // {{cg, tg, branch, fi}}
const _spawnedSet  = new Set();
let _ghostTrajIdx  = -1, _ghostLastFrame = -1;

function _recolorGhost(root){{
  root.traverse(o => {{
    if(!o.isMesh) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    const nm = mats.map(m => {{
      const c = m.clone();
      c.color.setHex(0xcc2200);
      if(c.emissive) c.emissive.setHex(0x550000);
      c.transparent = false; c.opacity = 1.0; c.depthWrite = true;
      c.polygonOffset = true; c.polygonOffsetFactor = 4; c.polygonOffsetUnits = 4;
      return c;
    }});
    o.material = Array.isArray(o.material) ? nm : nm[0];
  }});
}}

function _spawnGhost(branch){{
  const {{cabGroup: cg, trailerGroup: tg}} = buildTruck();
  _recolorGhost(cg); _recolorGhost(tg);
  _ghostRoot.add(cg); _ghostRoot.add(tg);
  _liveGhosts.push({{cg, tg, branch, fi: 0}});
}}

function _placeGhost(g, fu){{
  const p = pos3(fu.x, fu.y);
  g.cg.position.set(p.x, 0, p.z); g.tg.position.set(p.x, 0, p.z);
  g.cg.rotation.y = fu.t0;        g.tg.rotation.y = fu.t1;
}}

function _clearGhosts(){{
  _liveGhosts.forEach(g => {{ _ghostRoot.remove(g.cg); _ghostRoot.remove(g.tg); }});
  _liveGhosts.length = 0;
  _spawnedSet.clear();
}}

// Toggle button
(function(){{
  const btn = document.createElement('button');
  btn.textContent = 'Hide Ghosts';
  btn.style.cssText = 'position:absolute;top:55px;right:16px;background:rgba(5,5,15,.88);color:#90caf9;border:1px solid rgba(255,255,255,.1);border-radius:9px;padding:7px 13px;font-size:11px;font-weight:600;cursor:pointer;backdrop-filter:blur(14px);letter-spacing:.5px;';
  btn.addEventListener('click', () => {{
    _ghostRoot.visible = !_ghostRoot.visible;
    btn.textContent = _ghostRoot.visible ? 'Hide Ghosts' : 'Show Ghosts';
  }});
  document.body.appendChild(btn);
}})();

const _applyFrameOrig = applyFrame;
applyFrame = function(f){{
  _applyFrameOrig(f);
  const branches = TRAJS_UNTRAINED_BRANCHES[trajIdx];

  // Reset on trajectory change or backward scrub
  if(_ghostTrajIdx !== trajIdx || frame < _ghostLastFrame){{
    _ghostTrajIdx = trajIdx;
    _clearGhosts();
  }}
  _ghostLastFrame = frame;

  // Spawn a new ghost truck when the main animation crosses each checkpoint
  if(branches){{
    for(let b = 0; b < branches.length; b++){{
      if(frame >= b * _BRANCH_EVERY && b < branches.length - 1 && !_spawnedSet.has(b)){{
        _spawnedSet.add(b);
        _spawnGhost(branches[b]);
      }}
    }}
  }}

  // Advance every live ghost one frame (freeze at last frame)
  _liveGhosts.forEach(g => {{
    if(!g.branch || g.branch.length === 0) return;
    _placeGhost(g, g.branch[g.fi]);
    if(g.fi < g.branch.length - 1) g.fi++;
  }});
}};
// @@COMPARISON_END@@
"""


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Add untrained-model comparison to simulation_3d.html")
    p.add_argument("--html",             default="simulation_3d.html")
    p.add_argument("--untrained_lesson", type=int, default=0)
    p.add_argument("--max_steps",        type=int, default=400)
    p.add_argument("--branch_every",     type=int, default=20)
    p.add_argument("--branch_steps",     type=int, default=80)
    args = p.parse_args()

    # ── Read HTML ──────────────────────────────────────────────────────────
    with open(args.html, encoding="utf-8") as f:
        html = f.read()

    # ── Extract trained TRAJS ──────────────────────────────────────────────
    m = re.search(r'const TRAJS = (\[[\s\S]*?\]);', html)
    if not m:
        raise RuntimeError("Could not find 'const TRAJS = [...]' in the HTML")
    trained_trajs = json.loads(m.group(1))
    print(f"Found {len(trained_trajs)} trained trajectories in {args.html}")

    # ── Load untrained controller ──────────────────────────────────────────
    model_path = f"models/controllers/controller_lesson_{args.untrained_lesson}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Loading untrained model: {model_path}")
    controller = torch.load(model_path, weights_only=False)
    controller.eval()

    # ── Run untrained from every Nth frame of each trained trajectory ─────
    sim = TruckSim()
    untrained_branches = []
    for i, traj in enumerate(trained_trajs):
        branches = []
        checkpoints = list(range(0, len(traj), args.branch_every))
        print(f"  Traj {i+1:2d}/{len(trained_trajs)}: {len(checkpoints)} branches ...", end=" ", flush=True)
        for j in checkpoints:
            f0 = traj[j]
            frames = sim.run_from(
                controller,
                f0["x"], f0["y"], f0["t0"], f0["t1"], f0["phi"],
                max_steps=args.branch_steps,
            )
            branches.append(frames)
        jk_count = sum(1 for b in branches if len(b) < args.branch_steps)
        print(f"{jk_count}/{len(branches)} jackknifed")
        untrained_branches.append(branches)

    branches_json = json.dumps(untrained_branches)

    # ── Inject / replace TRAJS_UNTRAINED_BRANCHES ─────────────────────────
    if "TRAJS_UNTRAINED_BRANCHES" in html:
        html = re.sub(
            r'const TRAJS_UNTRAINED_BRANCHES = \[[\s\S]*?\];',
            f'const TRAJS_UNTRAINED_BRANCHES = {branches_json};',
            html, count=1,
        )
        print("Replaced existing TRAJS_UNTRAINED_BRANCHES data.")
    else:
        html = re.sub(
            r'(const TRAJS = \[[\s\S]*?\];)',
            r'\1' + f'\nconst TRAJS_UNTRAINED_BRANCHES = {branches_json};',
            html, count=1,
        )
        print("Injected TRAJS_UNTRAINED_BRANCHES.")

    # ── Inject / replace second-truck JS ──────────────────────────────────
    js_block = SECOND_TRUCK_JS.format(branch_every=args.branch_every)
    if "// @@COMPARISON_START@@" in html and "// @@COMPARISON_END@@" in html:
        html = re.sub(
            r'// @@COMPARISON_START@@[\s\S]*?// @@COMPARISON_END@@',
            js_block.strip(),
            html, count=1,
        )
        print("Replaced existing second-truck JS block.")
    elif "// @@COMPARISON_START@@" not in html:
        idx = html.rfind('</script>')
        html = html[:idx] + js_block + '\n' + html[idx:]
        print("Injected second-truck JS.")
    else:
        print("Warning: found start marker but no end marker — re-injecting.")

    # ── Write output ──────────────────────────────────────────────────────
    with open(args.html, "w", encoding="utf-8") as f:
        f.write(html)
    size = os.path.getsize(args.html) // 1024
    print(f"\nDone → {os.path.abspath(args.html)}  ({size} KB)")


if __name__ == "__main__":
    main()
