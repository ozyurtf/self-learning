#!/usr/bin/env python3
"""
3D Realistic Truck Simulation
Usage:
  python simulate_3d.py
  python simulate_3d.py --num_trajectories 15 --test_lesson 10
  python simulate_3d.py --env_x_range 0 100 --env_y_range -30 30 \
                        --test_x_cab_range 40 90 --test_y_cab_range -25 25
"""

import torch
import math
import scipy.stats as stats
from random import seed, uniform
import json
import os
import argparse
import webbrowser
import urllib.request


# ─── Truck Dynamics ───────────────────────────────────────────────────────────

class TruckSim:
    W = 1.0; L = 1.0; D = 4.0; S = -0.1

    def __init__(self, env_x_range=(0, 40), env_y_range=(-15, 15)):
        self.box = [env_x_range[0], env_x_range[1], env_y_range[0], env_y_range[1]]

    def trailer_xy(self, x, y, t1):
        return x - self.D * math.cos(t1), y - self.D * math.sin(t1)

    def step(self, x, y, t0, t1, phi):
        x  += self.S * math.cos(t0);  y  += self.S * math.sin(t0)
        t0 += self.S / self.L * math.tan(phi)
        t1 += self.S / self.D * math.sin(t0 - t1)
        return x, y, t0, t1

    def is_jackknifed(self, t0, t1):
        d = abs(math.degrees(t0 - t1))
        return min(d, abs(d - 360)) > 90

    def is_offscreen(self, x, y, t0, t1):
        b = self.box
        x1 = x + 1.5 * self.L * math.cos(t0); y1 = y + 1.5 * self.L * math.sin(t0)
        x2, y2 = self.trailer_xy(x, y, t1)
        return not (b[0]<=x1<=b[1] and b[2]<=y1<=b[3] and b[0]<=x2<=b[1] and b[2]<=y2<=b[3])

    def is_valid(self, x, y, t0, t1):
        return not self.is_jackknifed(t0, t1) and not self.is_offscreen(x, y, t0, t1)

    def run(self, controller, test_seed, test_config):
        rng_seed = test_seed
        for _ in range(200):
            seed(rng_seed)
            x  = uniform(*test_config["x_range"]);  y  = uniform(*test_config["y_range"])
            t0 = math.radians(uniform(*test_config["t0_range"]))
            t1 = math.radians(uniform(*test_config["dt_range"])) + t0
            if self.is_valid(x, y, t0, t1): break
            rng_seed += 1
        phi_t = torch.tensor([math.radians(_random_deg())], dtype=torch.float32)
        frames = []
        while self.is_valid(x, y, t0, t1) and len(frames) < 3000:
            tx, ty = self.trailer_xy(x, y, t1)
            frames.append({"x":round(x,4),"y":round(y,4),"t0":round(t0,4),
                           "t1":round(t1,4),"phi":round(float(phi_t),4),
                           "tx":round(tx,4),"ty":round(ty,4)})
            state = torch.tensor([x, y, t0, t1], dtype=torch.float32)
            phi_t = controller(torch.cat((phi_t, state)))
            x, y, t0, t1 = self.step(x, y, t0, t1, float(phi_t))
        return frames


def _random_deg(mean=0, std=35, lo=-70, hi=70):
    a, b = (lo-mean)/std, (hi-mean)/std
    return float(stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=1)[0])


# ─── HTML Template ────────────────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Truck Backer-Upper — 3D</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#111;overflow:hidden;font-family:'Segoe UI',system-ui,sans-serif}
canvas{display:block}
#ui{
  position:absolute;top:16px;left:16px;color:#e0e0e0;
  background:rgba(5,5,15,.88);padding:15px 17px;border-radius:13px;
  backdrop-filter:blur(14px);border:1px solid rgba(255,255,255,.1);
  min-width:240px;user-select:none;
}
#ui h2{font-size:11px;font-weight:700;letter-spacing:1.3px;color:#90caf9;
  margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,.1);
  padding-bottom:7px;text-transform:uppercase}
.row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
label{font-size:11px;color:#888;min-width:68px}
select,input[type=range]{background:rgba(255,255,255,.07);color:#ddd;
  border:1px solid rgba(255,255,255,.14);border-radius:6px;
  padding:4px 8px;font-size:11px;cursor:pointer;flex:1}
input[type=range]{padding:2px 0}
button{background:rgba(255,255,255,.09);color:#ddd;
  border:1px solid rgba(255,255,255,.18);border-radius:7px;
  padding:6px 11px;font-size:11px;cursor:pointer;transition:background .15s;flex:1}
button:hover{background:rgba(255,255,255,.18)}
button.on{background:rgba(80,180,120,.25);border-color:rgba(80,180,120,.45);color:#a5d6a7}
#stats{font-size:11px;color:#666;margin-top:9px;
  border-top:1px solid rgba(255,255,255,.07);padding-top:8px;line-height:1.95}
#stats b{color:#bbb;font-weight:500}
#prog{height:3px;background:rgba(255,255,255,.07);border-radius:3px;margin-top:9px;overflow:hidden}
#pb{height:100%;background:linear-gradient(90deg,#4fc3f7,#81c784);border-radius:3px;width:0%}
#hint{position:absolute;bottom:14px;right:14px;color:#444;
  background:rgba(0,0,0,.6);padding:9px 13px;border-radius:9px;
  font-size:10px;line-height:2;border:1px solid rgba(255,255,255,.07)}
kbd{display:inline-block;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.2);
  border-radius:3px;padding:0 5px;font-size:9px;color:#bbb;margin:0 1px}
</style>
</head>
<body>
<div id="ui">
  <h2>Truck Backer-Upper 3D</h2>
  <div class="row"><label>Trajectory</label><select id="sel"></select></div>
  <div class="row">
    <label>Speed</label>
    <input type="range" id="spd" min="1" max="30" value="6">
    <span id="spdlbl" style="font-size:11px;color:#777;min-width:26px">6x</span>
  </div>
  <div class="row">
    <button id="playbtn">&#9654; Play</button>
    <button id="rstbtn">&#8635; Reset</button>
  </div>
  <div class="row">
    <button id="followbtn">Follow cam</button>
    <button id="freebtn" class="on">Free cam</button>
  </div>
  <div id="stats">
    Step&nbsp;<b id="s-step">0</b> / <b id="s-tot">0</b><br>
    Trailer (<b id="s-tx">—</b>, <b id="s-ty">—</b>)<br>
    Steer &nbsp;<b id="s-phi">—</b>°
  </div>
  <div id="prog"><div id="pb"></div></div>
</div>
<div id="hint">
  Left-drag: orbit &nbsp;|&nbsp; Right-drag: pan &nbsp;|&nbsp; Scroll: zoom<br>
  <kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd> move camera &nbsp;|&nbsp; <kbd>PgUp</kbd><kbd>PgDn</kbd> up/down &nbsp;|&nbsp; <kbd>Space</kbd> play/pause &nbsp;|&nbsp; <kbd>R</kbd> reset view
</div>

<script src="three.min.js"></script>
<script src="OrbitControls.js"></script>
<script>
if(typeof THREE==='undefined'){
  document.body.style.cssText='background:#111;display:flex;align-items:center;justify-content:center;height:100vh';
  document.body.innerHTML='<p style="color:#f66;font-size:18px;font-family:sans-serif;text-align:center">three.min.js not found — make sure all three files are in the same folder.</p>';
  throw new Error('THREE not defined');
}

const TRAJS = __TRAJS__;
const ENV_X = __ENV_X__;
const TRAIN_ZONE = __TRAIN_ZONE__;   // [x1, x2, y_min, y_max] truck coords
const S = 3.0;

// ── Renderer ──────────────────────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth,innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
document.body.appendChild(renderer.domElement);
addEventListener('resize',()=>{
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
});

// ── Scene / Camera ────────────────────────────────────────────────────────
const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0xbcd8ee, 80, 360);

const camera = new THREE.PerspectiveCamera(55,innerWidth/innerHeight,0.2,900);
camera.position.set(160,55,0);

// ── OrbitControls ─────────────────────────────────────────────────────────
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping  = true;
controls.dampingFactor  = 0.07;
controls.minDistance    = 5;
controls.maxDistance    = 400;
controls.maxPolarAngle  = Math.PI * 0.495;
controls.enabled        = true;

// ── Lighting ──────────────────────────────────────────────────────────────
scene.add(new THREE.HemisphereLight(0xa8cadf, 0x4a5f3a, 0.65));
const sun = new THREE.DirectionalLight(0xfff6d0, 1.6);
sun.position.set(90, 130, 70);
sun.castShadow = true;
sun.shadow.mapSize.width = sun.shadow.mapSize.height = 4096;
sun.shadow.camera.near=1; sun.shadow.camera.far=600;
sun.shadow.camera.left=sun.shadow.camera.bottom=-200;
sun.shadow.camera.right=sun.shadow.camera.top=200;
sun.shadow.bias=-0.0005;
scene.add(sun);
const fill = new THREE.DirectionalLight(0x8ab4d8, 0.4);
fill.position.set(-60,40,-60); scene.add(fill);

// ── Sky dome ──────────────────────────────────────────────────────────────
(function(){
  const c=document.createElement('canvas'); c.width=2; c.height=512;
  const ctx=c.getContext('2d'), g=ctx.createLinearGradient(0,0,0,512);
  g.addColorStop(0,   '#184e96');
  g.addColorStop(0.4, '#4a8ec4');
  g.addColorStop(0.72,'#8dbedd');
  g.addColorStop(0.9, '#bcd8ee');
  g.addColorStop(1,   '#d8eef8');
  ctx.fillStyle=g; ctx.fillRect(0,0,2,512);
  const sky=new THREE.Mesh(
    new THREE.SphereGeometry(500,24,12),
    new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(c),side:THREE.BackSide})
  );
  sky.rotation.x=Math.PI/2; scene.add(sky); scene.background=null;
})();

// ── Asphalt ground ────────────────────────────────────────────────────────
(function(){
  const sz=2048, c=document.createElement('canvas'); c.width=c.height=sz;
  const ctx=c.getContext('2d');
  ctx.fillStyle='#1e1e1e'; ctx.fillRect(0,0,sz,sz);
  const id=ctx.getImageData(0,0,sz,sz), d=id.data;
  for(let i=0;i<d.length;i+=4){
    const n=(Math.random()-.5)*24; d[i]=d[i+1]=d[i+2]=Math.max(0,Math.min(255,d[i]+n));
  }
  ctx.putImageData(id,0,0);
  // aggregate speckles
  for(let k=0;k<8000;k++){
    const px=Math.random()*sz,py=Math.random()*sz,br=170+Math.random()*70|0;
    ctx.fillStyle=`rgb(${br},${br},${br})`; ctx.fillRect(px,py,1+Math.random()*2,1+Math.random()*2);
  }
  const tex=new THREE.CanvasTexture(c);
  tex.wrapS=tex.wrapT=THREE.RepeatWrapping; tex.repeat.set(35,35);
  const g=new THREE.Mesh(
    new THREE.PlaneGeometry(900,900),
    new THREE.MeshStandardMaterial({map:tex,roughness:.96,metalness:0})
  );
  g.rotation.x=-Math.PI/2; g.receiveShadow=true; scene.add(g);
})();



// ── Training Zone ─────────────────────────────────────────────────────────
(function(){
  const tx1=TRAIN_ZONE[0]*S, tx2=TRAIN_ZONE[1]*S;
  const tz1=-TRAIN_ZONE[3]*S, tz2=-TRAIN_ZONE[2]*S; // y→z inversion
  const Y=0.04;

  // Dashed yellow border
  const pts=[
    new THREE.Vector3(tx1,Y,tz1),
    new THREE.Vector3(tx2,Y,tz1),
    new THREE.Vector3(tx2,Y,tz2),
    new THREE.Vector3(tx1,Y,tz2),
    new THREE.Vector3(tx1,Y,tz1),
  ];
  const lg=new THREE.BufferGeometry().setFromPoints(pts);
  const lm=new THREE.LineDashedMaterial({color:0xFFDD00,dashSize:S*.9,gapSize:S*.55,linewidth:2});
  const ln=new THREE.Line(lg,lm);
  ln.computeLineDistances();
  scene.add(ln);

  // "Training Zone" ground label
  const cw=512,ch=96;
  const can=document.createElement('canvas');
  can.width=cw; can.height=ch;
  const ctx=can.getContext('2d');
  ctx.clearRect(0,0,cw,ch);
  ctx.font='bold 54px "Segoe UI",sans-serif';
  ctx.textAlign='center';
  ctx.strokeStyle='rgba(0,0,0,.75)';
  ctx.lineWidth=8;
  ctx.strokeText('Training Zone',cw/2,ch*.74);
  ctx.fillStyle='#FFDD00';
  ctx.fillText('Training Zone',cw/2,ch*.74);
  const tex=new THREE.CanvasTexture(can);
  const pw=(tx2-tx1)*.65, ph=pw*(ch/cw);
  const pl=new THREE.Mesh(
    new THREE.PlaneGeometry(pw,ph),
    new THREE.MeshBasicMaterial({map:tex,transparent:true,depthWrite:false,side:THREE.DoubleSide})
  );
  pl.rotation.x=-Math.PI/2;
  pl.rotation.z=Math.PI;
  pl.position.set((tx1+tx2)/2, Y+.01, (tz1+tz2)/2);
  scene.add(pl);
})();


// ── Road markings ─────────────────────────────────────────────────────────
(function(){
  const yM=new THREE.MeshStandardMaterial({color:0xFFCC00,roughness:1});
  const wM=new THREE.MeshStandardMaterial({color:0xffffff,roughness:1});
  const xEnd=ENV_X[1]*S;
  for(let x=S*2;x<xEnd;x+=S*3.5){
    const m=new THREE.Mesh(new THREE.PlaneGeometry(S*2,.1),yM);
    m.rotation.x=-Math.PI/2; m.position.set(x,.005,0); scene.add(m);
  }
  [-S*1.9,S*1.9].forEach(z=>{
    const e=new THREE.Mesh(new THREE.PlaneGeometry(xEnd,.09),wM);
    e.rotation.x=-Math.PI/2; e.position.set(xEnd/2,.005,z); scene.add(e);
  });
})();

// ── Loading dock & Warehouse ─────────────────────────────────────────────
(function(){
  const concrM=new THREE.MeshStandardMaterial({color:0x686868,roughness:.92,metalness:.05});
  const wallM =new THREE.MeshStandardMaterial({color:0x8a9eac,roughness:.84,metalness:.14});
  const wallDk =new THREE.MeshStandardMaterial({color:0x6e8290,roughness:.86,metalness:.12});
  const roofM =new THREE.MeshStandardMaterial({color:0x3e4a52,roughness:.88,metalness:.28});
  const steelM=new THREE.MeshStandardMaterial({color:0xb0b8c0,roughness:.22,metalness:.88});
  const darkM =new THREE.MeshStandardMaterial({color:0x0e0e0e,roughness:.95});
  const intM  =new THREE.MeshStandardMaterial({color:0x181818,roughness:.9});
  const yellowM=new THREE.MeshStandardMaterial({color:0xffd600,roughness:.72});
  const redEM =new THREE.MeshStandardMaterial({color:0xff2200,roughness:.1,emissive:0xcc1100,emissiveIntensity:.9});
  const greenEM=new THREE.MeshStandardMaterial({color:0x22dd00,roughness:.1,emissive:0x11aa00,emissiveIntensity:.9});

  // Trailer reference dims (WC=S*1.18, FY+HT=S*1.96) — door is sized to fit
  const doorW=S*1.55, doorH=S*2.28;   // slightly wider/taller than trailer

  // ── Dock platform ──
  const dockH=S*.44, dockD=S*2.0, dockW=S*5.5;
  const dock=new THREE.Mesh(new THREE.BoxGeometry(dockD,dockH,dockW),concrM);
  dock.position.set(-dockD/2,dockH/2,0); dock.castShadow=true; dock.receiveShadow=true; scene.add(dock);

  // Hazard stripe on dock face
  const hc=document.createElement('canvas'); hc.width=512; hc.height=64;
  const hctx=hc.getContext('2d');
  for(let i=0;i<16;i++){hctx.fillStyle=i%2===0?'#ffd600':'#111111';hctx.fillRect(i*32,0,32,64);}
  const hazEdge=new THREE.Mesh(new THREE.BoxGeometry(S*.05,dockH,dockW),
    new THREE.MeshStandardMaterial({map:new THREE.CanvasTexture(hc),roughness:.85}));
  hazEdge.position.set(0,dockH/2,0); scene.add(hazEdge);



  // Dock bumpers (rubber)
  [-S*1.5,0,S*1.5].forEach(bz=>{
    const bmp=new THREE.Mesh(new THREE.BoxGeometry(S*.13,S*.26,S*.36),
      new THREE.MeshStandardMaterial({color:0x111111,roughness:.96}));
    bmp.position.set(S*.06,dockH*.55,bz); scene.add(bmp);
  });

  // Wheel chocks
  [-S*1.0,S*1.0].forEach(cz=>{
    const chk=new THREE.Mesh(new THREE.BoxGeometry(S*.28,S*.12,S*.18),yellowM);
    chk.position.set(S*.65,S*.06,cz); scene.add(chk);
  });

  // ── Warehouse ──
  const WHW=S*7.0, WHH=S*10.5, WHD=S*9.0;
  const wbx=-(dockD+WHD/2);  // front face at x = -dockD = -S*2

  // Main building shell
  const wbody=new THREE.Mesh(new THREE.BoxGeometry(WHD,WHH,WHW*2),wallM);
  wbody.position.set(wbx,WHH/2,0); wbody.castShadow=true; wbody.receiveShadow=true; scene.add(wbody);

  // Roof slab
  const wroofSlab=new THREE.Mesh(new THREE.BoxGeometry(WHD+S*.3,S*.45,WHW*2+S*.4),roofM);
  wroofSlab.position.set(wbx,WHH+S*.22,0); wroofSlab.castShadow=true; scene.add(wroofSlab);

  const wx=wbx+WHD/2;  // warehouse front face x

  // ── Sectional overhead loading dock door (open — panels stored on ceiling) ──

  // Warm lit interior visible through the open door
  const intLitM=new THREE.MeshStandardMaterial({color:0x4a3a2c,roughness:.88,emissive:0x1a1008,emissiveIntensity:.5});
  const interior=new THREE.Mesh(new THREE.BoxGeometry(S*4.5,doorH+S*.1,doorW+S*.02),intLitM);
  interior.position.set(wx-S*2.23,doorH/2,0); scene.add(interior);
  // Interior concrete floor (slightly lighter)
  const iFloor=new THREE.Mesh(new THREE.PlaneGeometry(S*4.5,doorW+S*.02),
    new THREE.MeshStandardMaterial({color:0x5c5040,roughness:.9,emissive:0x0c0a06,emissiveIntensity:.25}));
  iFloor.rotation.x=-Math.PI/2; iFloor.position.set(wx-S*2.23,S*.01,0); scene.add(iFloor);
  // Fluorescent tube lights on interior ceiling
  [-S*.6,S*.6].forEach(lz=>{
    const tube=new THREE.Mesh(new THREE.CylinderGeometry(S*.03,S*.03,S*.45,8),
      new THREE.MeshStandardMaterial({color:0xfffff8,roughness:.05,emissive:0xffffee,emissiveIntensity:1.4}));
    tube.rotation.z=Math.PI/2; tube.position.set(wx-S*1.1,doorH-S*.06,lz); scene.add(tube);
  });

  // Concrete dock leveller plate (bridge between dock platform and trailer floor)
  const leveller=new THREE.Mesh(new THREE.BoxGeometry(S*.55,S*.05,doorW-S*.1),concrM);
  leveller.position.set(wx+S*.2,dockH,0); scene.add(leveller);

  // ── Door frame: thick steel jambs and head ──
  const jamb=new THREE.MeshStandardMaterial({color:0x8a9298,roughness:.35,metalness:.72});
  // Left/right jambs
  [-doorW/2-S*.09,doorW/2+S*.09].forEach(jz=>{
    const j=new THREE.Mesh(new THREE.BoxGeometry(S*.18,doorH+S*.55,S*.18),jamb);
    j.position.set(wx,doorH/2,jz); j.castShadow=true; scene.add(j);
  });
  // Head beam across top of opening
  const head=new THREE.Mesh(new THREE.BoxGeometry(S*.18,S*.22,doorW+S*.36),jamb);
  head.position.set(wx,doorH+S*.11,0); head.castShadow=true; scene.add(head);
  // Threshold sill at bottom
  const sill=new THREE.Mesh(new THREE.BoxGeometry(S*.14,S*.10,doorW+S*.28),jamb);
  sill.position.set(wx,S*.05,0); scene.add(sill);

  // ── Door panel housing (box above opening where panels store when open) ──
  const housingM=new THREE.MeshStandardMaterial({color:0x7a8590,roughness:.38,metalness:.65});
  const housingH=doorH*0.20, housingD=S*.55;
  const housing=new THREE.Mesh(new THREE.BoxGeometry(housingD,housingH,doorW+S*.32),housingM);
  housing.position.set(wx-housingD/2+S*.03,doorH+S*.22+housingH/2,0);
  housing.castShadow=true; scene.add(housing);
  // Horizontal accent lines on housing face
  [.25,.5,.75].forEach(t=>{
    const line=new THREE.Mesh(new THREE.BoxGeometry(housingD+S*.01,S*.025,doorW+S*.34),
      new THREE.MeshStandardMaterial({color:0x5a6268,roughness:.4,metalness:.7}));
    line.position.set(wx-housingD/2+S*.04,doorH+S*.22+t*housingH,0); scene.add(line);
  });
  // End caps on housing
  [-1,1].forEach(side=>{
    const cap=new THREE.Mesh(new THREE.BoxGeometry(housingD,housingH,S*.10),
      new THREE.MeshStandardMaterial({color:0x6a7278,roughness:.4,metalness:.7}));
    cap.position.set(wx-housingD/2+S*.03,doorH+S*.22+housingH/2,side*(doorW/2+S*.15)); scene.add(cap);
  });

  // Vertical guide rails (angle iron channels in the jambs)
  [-doorW/2,doorW/2].forEach(rz=>{
    const rail=new THREE.Mesh(new THREE.BoxGeometry(S*.06,doorH+S*.1,S*.06),steelM);
    rail.position.set(wx+S*.04,doorH/2,rz); scene.add(rail);
  });

  // Dock light above door
  const dlt=new THREE.Mesh(new THREE.CylinderGeometry(S*.14,S*.12,S*.10,8),
    new THREE.MeshStandardMaterial({color:0xffffbb,roughness:.1,emissive:0xffff88,emissiveIntensity:.75}));
  dlt.position.set(wx,doorH+S*.55,0); scene.add(dlt);

  // Status lights (red/green)
  [-S*1.1,S*1.1].forEach((lz,li)=>{
    const lt=new THREE.Mesh(new THREE.CylinderGeometry(S*.09,S*.09,S*.08,8),li===0?redEM:greenEM);
    lt.position.set(wx+S*.06,doorH+S*.95,lz); scene.add(lt);
  });

  // Vertical corrugation (skip over door opening)
  for(let vz=-WHW+S*.4;vz<=WHW;vz+=S*1.05){
    if(Math.abs(vz)<doorW/2+S*.25) continue;
    const cr=new THREE.Mesh(new THREE.BoxGeometry(S*.1,WHH,S*.08),
      new THREE.MeshStandardMaterial({color:0x7e9098,roughness:.84}));
    cr.position.set(wbx,WHH/2,vz); scene.add(cr);
  }

  // Company sign band
  const signBand=new THREE.Mesh(new THREE.BoxGeometry(S*.12,S*1.8,WHW*2),
    new THREE.MeshStandardMaterial({color:0x0d2a4e,roughness:.8}));
  signBand.position.set(wx,WHH-S*2.2,0); scene.add(signBand);

  // Side edge trim
  [-1,1].forEach(side=>{
    const sw=new THREE.Mesh(new THREE.BoxGeometry(WHD,WHH,S*.3),wallDk);
    sw.position.set(wbx,WHH/2,side*WHW); sw.castShadow=true; scene.add(sw);
  });

  // Canopy over dock approach
  const canH=S*5.0, canD=S*4.2, canW=dockW+S*1.2;
  const canRoof=new THREE.Mesh(new THREE.BoxGeometry(canD,S*.22,canW),roofM);
  canRoof.position.set(-canD/2+S*.1,canH,0); canRoof.castShadow=true; scene.add(canRoof);
  [-canW*.43,canW*.43].forEach(cz=>{
    const col=new THREE.Mesh(new THREE.CylinderGeometry(S*.09,S*.09,canH,8),steelM);
    col.position.set(-canD+S*.2,canH/2,cz); col.castShadow=true; scene.add(col);
  });
})();

// ── Traffic cones ─────────────────────────────────────────────────────────
(function(){
  function cone(x,z){
    const g=new THREE.Group();
    const body=new THREE.Mesh(new THREE.ConeGeometry(.24,.95,8),
      new THREE.MeshStandardMaterial({color:0xff5500,roughness:.65}));
    body.position.y=.48; body.castShadow=true;
    const base=new THREE.Mesh(new THREE.CylinderGeometry(.34,.34,.09,8),
      new THREE.MeshStandardMaterial({color:0xff5500,roughness:.75}));
    const s1=new THREE.Mesh(new THREE.CylinderGeometry(.245,.245,.14,8),
      new THREE.MeshStandardMaterial({color:0xffffff,roughness:.65}));
    s1.position.y=.35;
    g.add(body,base,s1); g.position.set(x,0,z); scene.add(g);
  }
  const ce=S*1.9;
  for(let z=-ce;z<=ce;z+=S*.85){
    if(Math.abs(z)>.5){cone(S*1.85,z);cone(-S*1.85,z);}
  }
})();


// ── Lamp posts ────────────────────────────────────────────────────────────
(function(){
  const postM=new THREE.MeshStandardMaterial({color:0x888888,roughness:.5,metalness:.7});
  [S*6,S*14,S*22,S*30].forEach(x=>{
    [-S*4.8,S*4.8].forEach(z=>{
      const post=new THREE.Mesh(new THREE.CylinderGeometry(.12,.15,8.5,8),postM);
      post.position.set(x,4.25,z); post.castShadow=true; scene.add(post);
      const arm=new THREE.Mesh(new THREE.CylinderGeometry(.055,.055,2.2,6),postM);
      arm.rotation.z=Math.PI/2;
      arm.position.set(x+(z>0?1.1:-1.1),8.4,z); scene.add(arm);
      const head=new THREE.Mesh(new THREE.CylinderGeometry(.3,.2,.32,8),
        new THREE.MeshStandardMaterial({color:0xddddcc,roughness:.4,emissive:0xffffaa,emissiveIntensity:.35}));
      head.position.set(x+(z>0?2.2:-2.2),8.25,z); scene.add(head);
    });
  });
})();

// ── Truck model ───────────────────────────────────────────────────────────
// Rounded-box mesh: single ExtrudeGeometry mesh with genuinely smooth normals.
// Cross-section (YZ) is a rounded rectangle, extruded in X with bevelled front/back.
// All 12 edges are smooth — no seams, no separate touching meshes.
// d=depth(X), h=height(Y), w=width(Z), r=corner radius
function mkRndMesh(d,h,w,r,mat){
  const segs=10, hw=w/2, hh=h/2;
  const sh=new THREE.Shape();
  sh.moveTo(-hw+r,-hh);
  sh.lineTo( hw-r,-hh); sh.absarc( hw-r,-hh+r,r,-Math.PI/2,0,false);
  sh.lineTo( hw,   hh-r); sh.absarc( hw-r, hh-r,r,0,Math.PI/2,false);
  sh.lineTo(-hw+r, hh); sh.absarc(-hw+r, hh-r,r,Math.PI/2,Math.PI,false);
  sh.lineTo(-hw,  -hh+r); sh.absarc(-hw+r,-hh+r,r,Math.PI,3*Math.PI/2,false);
  const bd=Math.max(.01,d-2*r);
  const geo=new THREE.ExtrudeGeometry(sh,{
    depth:bd, bevelEnabled:true,
    bevelThickness:r, bevelSize:r, bevelSegments:segs
  });
  // Centre in Z, then rotate so depth runs along X
  geo.translate(0,0,-(bd/2));
  geo.applyMatrix4(new THREE.Matrix4().makeRotationY(Math.PI/2));
  const m=new THREE.Mesh(geo,mat);
  m.castShadow=m.receiveShadow=true;
  return m;
}
function addRndMesh(parent,d,h,w,r,mat,px,py,pz){
  const m=mkRndMesh(d,h,w,r,mat); m.position.set(px,py,pz); parent.add(m); return m;
}

function buildTruck(){

  // Materials
  const paint  = new THREE.MeshStandardMaterial({color:0xb82200,roughness:.38,metalness:.45});
  const paintD = new THREE.MeshStandardMaterial({color:0x8a1800,roughness:.50,metalness:.32});
  const chrome = new THREE.MeshStandardMaterial({color:0xd8d8d8,roughness:.12,metalness:.94});
  const glass  = new THREE.MeshStandardMaterial({color:0x192e42,roughness:.04,metalness:.55,transparent:true,opacity:.80});
  const rubber = new THREE.MeshStandardMaterial({color:0x0e0e0e,roughness:.95});
  const rim    = new THREE.MeshStandardMaterial({color:0xc0c0c0,roughness:.22,metalness:.88});
  const trlMat = new THREE.MeshStandardMaterial({color:0xefefef,roughness:.55,metalness:.08});
  const trlDrk = new THREE.MeshStandardMaterial({color:0xcccccc,roughness:.65,metalness:.08});
  const underM = new THREE.MeshStandardMaterial({color:0x1a1a1a,roughness:.9,metalness:.4});
  const redEM  = new THREE.MeshStandardMaterial({color:0xff1500,roughness:.08,emissive:0xcc1000,emissiveIntensity:.7});
  const ambEM  = new THREE.MeshStandardMaterial({color:0xff7700,roughness:.08,emissive:0xcc5500,emissiveIntensity:.6});
  const headEM = new THREE.MeshStandardMaterial({color:0xffffdd,roughness:.04,emissive:0xffffaa,emissiveIntensity:.8});

  function mkWheel(r,tireW){
    const g=new THREE.Group();
    const tire=new THREE.Mesh(new THREE.CylinderGeometry(r,r,tireW,24),rubber);
    tire.rotation.x=Math.PI/2; tire.castShadow=true;
    const hub=new THREE.Mesh(new THREE.CylinderGeometry(r*.56,r*.56,tireW+.01,14),rim);
    hub.rotation.x=Math.PI/2;
    const lugR=r*.40;
    for(let a=0;a<8;a++){
      const lug=new THREE.Mesh(new THREE.CylinderGeometry(r*.045,r*.045,tireW+.02,6),chrome);
      lug.rotation.x=Math.PI/2;
      lug.position.set(lugR*Math.cos(a*Math.PI/4),lugR*Math.sin(a*Math.PI/4),0);
      g.add(lug);
    }
    g.add(tire,hub); return g;
  }

  // Key dimensions
  const WC=S*1.18, HC=S*1.60, LC=S*1.12;
  const CR=S*.07;    // cab body corner radius
  const FY=S*0.38;   // frame floor height above ground (chassis clearance)
  const GAP=S*0.26;  // visual gap between cab rear and trailer front

  // ── CAB ──────────────────────────────────────────────────────────────────
  const cabGroup=new THREE.Group();

  // cabShell: all body panels — lifted FY off ground, shifted GAP forward
  const cabShell=new THREE.Group();
  cabShell.position.set(GAP,FY,0);
  cabGroup.add(cabShell);

  // Main cab body (rounded)
  addRndMesh(cabShell, LC,HC,WC, CR, paint, LC/2,HC/2,0);

  // Sleeper box behind cab
  addRndMesh(cabShell, S*.38,HC*.88,WC*.97, S*.04, paintD, S*.19,HC*.44,0);


  // Chrome front surround frame
  const frontSurround=new THREE.Mesh(new THREE.BoxGeometry(S*.055,HC*.92,WC*1.04),chrome);
  frontSurround.position.set(LC+S*.028,HC*.46,0); cabShell.add(frontSurround);

  // Windshield — top 36% of front face; gap below shows red hood area
  const ws=new THREE.Mesh(new THREE.BoxGeometry(S*.07,HC*.34,WC*.86),glass);
  ws.position.set(LC+S*.005,HC*.77,0); ws.rotation.z=-.11; cabShell.add(ws);
  [-1,1].forEach(side=>{
    const pl=new THREE.Mesh(new THREE.BoxGeometry(S*.065,HC*.38,S*.055),chrome);
    pl.position.set(LC+S*.005,HC*.77,side*WC*.45); cabShell.add(pl);
  });
  const wsTop=new THREE.Mesh(new THREE.BoxGeometry(S*.065,S*.05,WC*.86),chrome);
  wsTop.position.set(LC+S*.005,HC*.95,0); cabShell.add(wsTop);

  // Grill — compact, only bottom 22% of front face
  // Large red hood area (HC*.24 to HC*.59) visible between grill top and windshield bottom
  const grillBG=new THREE.Mesh(new THREE.BoxGeometry(S*.055,HC*.22,WC*.82),
    new THREE.MeshStandardMaterial({color:0x111111,roughness:.6,metalness:.5}));
  grillBG.position.set(LC+S*.005,HC*.12,0); cabShell.add(grillBG);
  for(let gi=0;gi<4;gi++){
    const bar=new THREE.Mesh(new THREE.BoxGeometry(S*.06,S*.022,WC*.80),chrome);
    bar.position.set(LC+S*.008,S*.034+gi*S*.048,0); cabShell.add(bar);
  }

  // Bumper (rounded chrome, at very bottom of cab shell)
  addRndMesh(cabShell, S*.20,S*.28,WC*1.06, S*.05, chrome, LC+S*.10,S*.14,0);

  // Headlights + turn signals
  [-1,1].forEach(side=>{
    const hlight=new THREE.Mesh(new THREE.BoxGeometry(S*.07,S*.26,S*.32),headEM);
    hlight.position.set(LC+S*.04,HC*.28,side*WC*.40); cabShell.add(hlight);
    const hframe=new THREE.Mesh(new THREE.BoxGeometry(S*.06,S*.30,S*.36),chrome);
    hframe.position.set(LC+S*.035,HC*.28,side*WC*.40); cabShell.add(hframe);
    const ts=new THREE.Mesh(new THREE.BoxGeometry(S*.06,S*.12,S*.18),ambEM);
    ts.position.set(LC+S*.04,HC*.12,side*WC*.40); cabShell.add(ts);
  });

  // Side mirrors — compact stub-and-housing (not oversized)
  [-1,1].forEach(side=>{
    const arm=new THREE.Mesh(new THREE.BoxGeometry(S*.05,S*.05,S*.22),chrome);
    arm.position.set(LC*.82,HC*.73,side*(WC/2+S*.11)); cabShell.add(arm);
    addRndMesh(cabShell, S*.052,S*.22,S*.14, S*.026,
      new THREE.MeshStandardMaterial({color:0x222222,roughness:.5,metalness:.5}),
      LC*.80,HC*.73,side*(WC/2+S*.24));
    const mFace=new THREE.Mesh(new THREE.BoxGeometry(S*.036,S*.18,S*.10),
      new THREE.MeshStandardMaterial({color:0x1a2a38,roughness:.04,metalness:.7,transparent:true,opacity:.9}));
    mFace.position.set(LC*.81,HC*.73,side*(WC/2+S*.24)); cabShell.add(mFace);
  });

  // Entry steps
  [-1,1].forEach(side=>{
    [S*.10,S*.27].forEach(ys=>{
      const step=new THREE.Mesh(new THREE.BoxGeometry(S*.20,S*.05,S*.16),chrome);
      step.position.set(LC+S*.04,ys,side*(WC/2+S*.06)); cabShell.add(step);
    });
    const bar=new THREE.Mesh(new THREE.CylinderGeometry(S*.018,S*.018,S*.26,6),chrome);
    bar.position.set(LC+S*.04,S*.23,side*(WC/2+S*.08)); cabShell.add(bar);
  });

  // Door windows
  [-1,1].forEach(side=>{
    const dWin=new THREE.Mesh(new THREE.BoxGeometry(S*.02,S*.36,S*.50),glass);
    dWin.position.set(LC*.40,HC*.78,side*(WC/2+S*.01)); cabShell.add(dWin);
  });

  // Fifth-wheel plate (frame floor level)
  const fw5=new THREE.Mesh(new THREE.CylinderGeometry(S*.32,S*.32,S*.09,16),chrome);
  fw5.position.set(0,S*.05,0); cabShell.add(fw5);

  // ── Wheels in cabGroup (grounded — y=wheel_radius keeps them on ground) ──

  // Single front steer axle — positioned at ~65% of cab length (under driver)
  [-1,1].forEach(side=>{
    const w=mkWheel(S*.33,S*.20); w.position.set(GAP+LC*.65,S*.33,side*(WC/2+S*.14)); cabGroup.add(w);
  });

  // ── TRAILER ──────────────────────────────────────────────────────────────
  const trailerGroup=new THREE.Group();
  const TL=S*4, HT=S*1.58, TR=S*.06;

  // trlShell: trailer body panels lifted FY
  const trlShell=new THREE.Group();
  trlShell.position.y=FY;
  trailerGroup.add(trlShell);

  // Main trailer body (rounded)
  addRndMesh(trlShell, TL,HT,WC*.99, TR, trlMat, -TL/2,HT/2,0);

  // Front cap + rear door face
  const tFront=new THREE.Mesh(new THREE.BoxGeometry(S*.07,HT,WC*.99),trlDrk);
  tFront.position.set(0,HT/2,0); trlShell.add(tFront);
  const tRear=new THREE.Mesh(new THREE.BoxGeometry(S*.07,HT,WC*.99),
    new THREE.MeshStandardMaterial({color:0xaaaaaa,roughness:.65}));
  tRear.position.set(-TL,HT/2,0); trlShell.add(tRear);

  // Horizontal ribs
  for(let ri=0;ri<8;ri++){
    const ry=S*.10+ri*S*.18;
    [-1,1].forEach(side=>{
      const hr=new THREE.Mesh(new THREE.BoxGeometry(TL+.08,S*.050,S*.06),trlDrk);
      hr.position.set(-TL/2,ry,side*(WC/2+S*.03)); trlShell.add(hr);
    });
  }
  // Vertical ribs
  for(let vx=0;vx<=TL;vx+=S*0.88){
    [-1,1].forEach(side=>{
      const vr=new THREE.Mesh(new THREE.BoxGeometry(S*.07,HT,S*.06),trlDrk);
      vr.position.set(-vx,HT/2,side*(WC/2+S*.03)); trlShell.add(vr);
    });
  }

  // Rear door panel + split
  const dPanel=new THREE.Mesh(new THREE.BoxGeometry(S*.04,S*1.4,WC*.46),
    new THREE.MeshStandardMaterial({color:0xe0e0e0,roughness:.6}));
  dPanel.position.set(-TL-.02,HT/2,0); trlShell.add(dPanel);
  const dSplit=new THREE.Mesh(new THREE.BoxGeometry(S*.08,HT,S*.04),
    new THREE.MeshStandardMaterial({color:0x888888,roughness:.7}));
  dSplit.position.set(-TL-.03,HT/2,0); trlShell.add(dSplit);

  // Rear lights
  [-1,1].forEach(side=>{
    const rl=new THREE.Mesh(new THREE.BoxGeometry(S*.075,S*.28,S*.28),redEM);
    rl.position.set(-TL-.03,S*.55,side*WC*.40); trlShell.add(rl);
    const al=new THREE.Mesh(new THREE.BoxGeometry(S*.075,S*.20,S*.20),ambEM);
    al.position.set(-TL-.03,S*.22,side*WC*.40); trlShell.add(al);
  });

  // Side markers
  [.15,.35,.55,.75,.90].forEach(frac=>{
    [-1,1].forEach(side=>{
      const ml=new THREE.Mesh(new THREE.BoxGeometry(S*.04,S*.09,S*.09),
        new THREE.MeshStandardMaterial({color:0xff8800,roughness:.1,emissive:0xcc5500,emissiveIntensity:.4}));
      ml.position.set(-TL*frac,HT*.54,side*(WC/2+S*.02)); trlShell.add(ml);
    });
  });

  // Reflective stripe
  [-1,1].forEach(side=>{
    const stripe=new THREE.Mesh(new THREE.BoxGeometry(TL+.08,S*.08,S*.055),
      new THREE.MeshStandardMaterial({color:0xff6600,roughness:.1,emissive:0xcc4000,emissiveIntensity:.3}));
    stripe.position.set(-TL/2,S*.18,side*(WC/2+S*.03)); trlShell.add(stripe);
  });

  // Rear underride guard
  const guard=new THREE.Mesh(new THREE.BoxGeometry(S*.10,S*.38,WC*1.06),
    new THREE.MeshStandardMaterial({color:0x555555,roughness:.7,metalness:.5}));
  guard.position.set(-TL-.045,S*.19,0); trlShell.add(guard);

  // ── Trailer wheels in trailerGroup (grounded) ─────────────────────────────
  [.50,.96].forEach(ax=>{
    [-1,1].forEach(side=>{
      const wi=mkWheel(S*.31,S*.15); wi.position.set(-(TL-S*ax),S*.31,side*(WC/2+S*.06)); trailerGroup.add(wi);
      const wo=mkWheel(S*.31,S*.15); wo.position.set(-(TL-S*ax),S*.31,side*(WC/2+S*.26)); trailerGroup.add(wo);
    });
  });

  // Landing legs (span ground → frame floor)
  const legH=FY-S*.08;
  [-1,1].forEach(side=>{
    const leg=new THREE.Mesh(new THREE.BoxGeometry(S*.11,legH,S*.11),underM);
    leg.position.set(-S*.7,legH/2+S*.04,side*WC*.34); trailerGroup.add(leg);
    const foot=new THREE.Mesh(new THREE.BoxGeometry(S*.26,S*.06,S*.26),underM);
    foot.position.set(-S*.7,S*.03,side*WC*.34); trailerGroup.add(foot);
  });

  // Kingpin support post (ground to fifth-wheel)
  const kp=new THREE.Mesh(new THREE.CylinderGeometry(S*.09,S*.09,FY+S*.10,12),chrome);
  kp.position.set(0,(FY+S*.10)/2,0); trailerGroup.add(kp);

  scene.add(cabGroup);
  scene.add(trailerGroup);
  return {cabGroup, trailerGroup};
}

const {cabGroup, trailerGroup} = buildTruck();

// ── State & Camera ────────────────────────────────────────────────────────
let trajIdx=0, frame=0, playing=false, follow=false, spd=6;
let camPhi=0.28, camR=52;
const camSmoothTgt = new THREE.Vector3();

function pos3(x2,y2){return{x:x2*S, z:-y2*S};}

function applyFrame(f){
  if(!f) return;
  const p=pos3(f.x,f.y);
  cabGroup.position.set(p.x,0,p.z);
  trailerGroup.position.set(p.x,0,p.z);
  cabGroup.rotation.y     = f.t0;
  trailerGroup.rotation.y = f.t1;
  const traj=TRAJS[trajIdx];
  document.getElementById('s-step').textContent = frame;
  document.getElementById('s-tot').textContent  = traj.length;
  document.getElementById('s-tx').textContent   = f.tx.toFixed(2);
  document.getElementById('s-ty').textContent   = f.ty.toFixed(2);
  document.getElementById('s-phi').textContent  = (f.phi*180/Math.PI).toFixed(1);
  document.getElementById('pb').style.width     = (100*frame/traj.length)+'%';
}

function initCam(f){
  const p=pos3(f.x,f.y);
  // Place camera in front of the cab (in the direction the truck faces) so you
  // see the nose head-on — this is the "front view" as the truck backs toward dock
  const fx=Math.cos(f.t0), fz=-Math.sin(f.t0);
  camera.position.set(
    p.x+camR*fx*Math.cos(camPhi),
    camR*Math.sin(camPhi)+4,
    p.z+camR*fz*Math.cos(camPhi)
  );
  camSmoothTgt.set(p.x,0,p.z);
  controls.target.set(p.x,3.5,p.z);
  camera.lookAt(p.x,3.5,p.z);
}

function updateFollowCam(){
  const f=TRAJS[trajIdx][Math.min(frame,TRAJS[trajIdx].length-1)];
  const p=pos3(f.x,f.y);
  camSmoothTgt.lerp(new THREE.Vector3(p.x,0,p.z),.06);
  const fx=Math.cos(f.t0), fz=-Math.sin(f.t0);
  camera.position.lerp(new THREE.Vector3(
    camSmoothTgt.x+camR*fx*Math.cos(camPhi),
    camR*Math.sin(camPhi)+4,
    camSmoothTgt.z+camR*fz*Math.cos(camPhi)
  ),.045);
  controls.target.lerp(new THREE.Vector3(camSmoothTgt.x,3.5,camSmoothTgt.z),.06);
  camera.lookAt(controls.target);
}

// ── UI ────────────────────────────────────────────────────────────────────
const sel=document.getElementById('sel');
TRAJS.forEach((_,i)=>{
  const o=document.createElement('option');
  o.value=i; o.textContent=`Trajectory ${i+1}  (${TRAJS[i].length} steps)`;
  sel.appendChild(o);
});
sel.addEventListener('change',()=>{
  trajIdx=+sel.value; frame=0;
  applyFrame(TRAJS[trajIdx][0]); initCam(TRAJS[trajIdx][0]);
});

function togglePlay(){
  playing=!playing;
  document.getElementById('playbtn').textContent=playing?'⏸ Pause':'▶ Play';
}
document.getElementById('playbtn').addEventListener('click',togglePlay);
document.getElementById('rstbtn').addEventListener('click',()=>{
  frame=0; applyFrame(TRAJS[trajIdx][0]);
  if(follow) initCam(TRAJS[trajIdx][0]);
});
document.getElementById('followbtn').addEventListener('click',()=>{
  follow=true; controls.enabled=false;
  document.getElementById('followbtn').classList.add('on');
  document.getElementById('freebtn').classList.remove('on');
  initCam(TRAJS[trajIdx][Math.min(frame,TRAJS[trajIdx].length-1)]);
});
document.getElementById('freebtn').addEventListener('click',()=>{
  follow=false; controls.enabled=true;
  document.getElementById('freebtn').classList.add('on');
  document.getElementById('followbtn').classList.remove('on');
});
const spdSlider=document.getElementById('spd');
spdSlider.addEventListener('input',()=>{
  spd=+spdSlider.value;
  document.getElementById('spdlbl').textContent=spd+'x';
});
// ── Keyboard state ─────────────────────────────────────────────────────────
const keys={};
addEventListener('keydown',e=>{
  keys[e.code]=true;
  if(e.code==='Space'){e.preventDefault();togglePlay();}
  if(e.code==='KeyR'){frame=0;applyFrame(TRAJS[trajIdx][0]);initCam(TRAJS[trajIdx][0]);}
  if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight','PageUp','PageDown'].includes(e.code)) e.preventDefault();
});
addEventListener('keyup',e=>{keys[e.code]=false;});
renderer.domElement.addEventListener('wheel',e=>{
  if(follow) camR=Math.max(8,Math.min(200,camR+e.deltaY*.08));
},{passive:true});

// ── Render loop ────────────────────────────────────────────────────────────
const clock=new THREE.Clock(); let accum=0;
function applyKeyboardCamera(dt){
  if(follow) return;
  const speed=30*dt;
  // Forward direction = where camera is looking, projected flat onto XZ plane
  const fwd=new THREE.Vector3();
  camera.getWorldDirection(fwd); fwd.y=0; fwd.normalize();
  const right=new THREE.Vector3().crossVectors(fwd,new THREE.Vector3(0,1,0)).normalize();
  if(keys['ArrowUp'])    {camera.position.addScaledVector(fwd, speed); controls.target.addScaledVector(fwd, speed);}
  if(keys['ArrowDown'])  {camera.position.addScaledVector(fwd,-speed); controls.target.addScaledVector(fwd,-speed);}
  if(keys['ArrowLeft'])  {camera.position.addScaledVector(right,-speed); controls.target.addScaledVector(right,-speed);}
  if(keys['ArrowRight']) {camera.position.addScaledVector(right, speed); controls.target.addScaledVector(right, speed);}
  if(keys['PageUp'])   {camera.position.y+=speed; controls.target.y+=speed;}
  if(keys['PageDown']) {camera.position.y-=speed; controls.target.y-=speed;}
}
function animate(){
  requestAnimationFrame(animate);
  const dt=clock.getDelta();
  if(playing){
    accum+=dt*30*(spd/6);
    while(accum>=1){
      accum-=1;
      const traj=TRAJS[trajIdx];
      if(frame<traj.length-1){frame++;applyFrame(traj[frame]);}
      else{playing=false;document.getElementById('playbtn').textContent='▶ Play';break;}
    }
  }
  applyKeyboardCamera(dt);
  follow ? updateFollowCam() : controls.update();
  renderer.render(scene,camera);
}

applyFrame(TRAJS[0][0]);
initCam(TRAJS[0][0]);
animate();
</script>
</body>
</html>
"""


# ─── Download helpers ─────────────────────────────────────────────────────────

_LIBS = [
    ("three.min.js",     ["https://cdn.jsdelivr.net/npm/three@0.134.0/build/three.min.js",
                          "https://unpkg.com/three@0.134.0/build/three.min.js"]),
    ("OrbitControls.js", ["https://cdn.jsdelivr.net/npm/three@0.134.0/examples/js/controls/OrbitControls.js",
                          "https://unpkg.com/three@0.134.0/examples/js/controls/OrbitControls.js"]),
]


def _download(urls, label):
    for url in urls:
        try:
            print(f"  Fetching {label} ...", end=" ", flush=True)
            with urllib.request.urlopen(url, timeout=20) as r:
                src = r.read().decode("utf-8")
            print(f"ok ({len(src)//1024} KB)")
            return src
        except Exception as e:
            print(f"failed ({e})")
    return None


def generate_html(trajectories, env_x, env_y, train_zone, out_path):
    out_dir = os.path.dirname(os.path.abspath(out_path))
    for fname, urls in _LIBS:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            print(f"  Cached  {fname}")
        else:
            src = _download(urls, fname)
            if src is None:
                raise RuntimeError(f"Could not download {fname}. Check internet connection.")
            with open(fpath, "w") as f:
                f.write(src)
    html = _HTML
    html = html.replace("__TRAJS__", json.dumps(trajectories))
    html = html.replace("__ENV_X__", json.dumps(list(env_x)))
    html = html.replace("__TRAIN_ZONE__", json.dumps(list(train_zone)))
    with open(out_path, "w") as f:
        f.write(html)
    size = os.path.getsize(out_path) // 1024
    print(f"Saved  → {os.path.abspath(out_path)}  ({size} KB)")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="3D Truck Backer-Upper Simulation")
    p.add_argument("--test_lesson",      type=int, default=10)
    p.add_argument("--num_trajectories", type=int, default=15)
    p.add_argument("--env_x_range",      type=int, nargs=2, default=(0, 100))
    p.add_argument("--env_y_range",      type=int, nargs=2, default=(-30, 30))
    # Start farther from target by default
    p.add_argument("--test_x_cab_range", type=int, nargs=2, default=(40, 90))
    p.add_argument("--test_y_cab_range", type=int, nargs=2, default=(-25, 25))
    p.add_argument("--test_cab_angle_range",              type=int, nargs=2, default=(-180, 180))
    p.add_argument("--test_cab_trailer_angle_diff_range", type=int, nargs=2, default=(-45, 45))
    p.add_argument("--train_x_range", type=float, nargs=2, default=(10, 35),
                   help="Training zone x cab range (truck coords)")
    p.add_argument("--train_y_range", type=float, nargs=2, default=(-7, 7),
                   help="Training zone y cab range (truck coords)")
    p.add_argument("--output",      type=str,  default="simulation_3d.html")
    p.add_argument("--no_browser",  action="store_true")
    args = p.parse_args()

    test_config = {
        "x_range":  tuple(args.test_x_cab_range),
        "y_range":  tuple(args.test_y_cab_range),
        "t0_range": tuple(args.test_cab_angle_range),
        "dt_range": tuple(args.test_cab_trailer_angle_diff_range),
    }

    model_path = f"models/controllers/controller_lesson_{args.test_lesson}.pth"
    print(f"Loading {model_path} ...")
    controller = torch.load(model_path, weights_only=False)
    controller.eval()

    sim = TruckSim(env_x_range=args.env_x_range, env_y_range=args.env_y_range)
    all_trajs = []
    for i in range(1, args.num_trajectories + 1):
        print(f"  Traj {i:2d}/{args.num_trajectories} ...", end=" ", flush=True)
        with torch.no_grad():
            frames = sim.run(controller, test_seed=i, test_config=test_config)
        last = frames[-1] if frames else {}
        jk = last and sim.is_jackknifed(last["t0"], last["t1"])
        print(f"{len(frames):4d} steps  {'[jackknifed]' if jk else '[ok]'}")
        all_trajs.append(frames)

    train_zone = [args.train_x_range[0], args.train_x_range[1],
                  args.train_y_range[0], args.train_y_range[1]]
    generate_html(all_trajs, args.env_x_range, args.env_y_range, train_zone, args.output)
    if not args.no_browser:
        url = f"file://{os.path.abspath(args.output)}"
        print(f"Opening  → {url}")
        webbrowser.open(url)


if __name__ == "__main__":
    main()
