import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from "recharts";

// ─── Palette — green-phosphor oscilloscope on matte black ─────────────────────
const P = {
  bg:       "#050607",
  surface:  "#090b0d",
  panel:    "#0d1014",
  border:   "#1a2028",
  borderHi: "#243040",
  phos:     "#00ff7f",   // phosphor green
  phosLo:   "#00c45f",
  phosGlow: "rgba(0,255,127,0.12)",
  amber:    "#ffb300",
  cyan:     "#00e5ff",
  violet:   "#b388ff",
  red:      "#ff3d57",
  dim:      "#2e3d30",
  textHi:   "#c8ffdc",
  textMid:  "#6b9e78",
  textDim:  "#334d38",
  scanline: "repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,127,0.015) 2px,rgba(0,255,127,0.015) 4px)",
};

// ─── Training log embedded (first 800 steps, pre-computed) ────────────────────
// We'll generate it deterministically in JS to match the Python output
function generateTrainingLog() {
  // Seeded PRNG (mulberry32)
  let s = 42;
  const rand = () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const randn = () => { let u=0,v=0; while(!u)u=rand(); while(!v)v=rand(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };

  const N = 800, n_cols = 12, k = 4;
  // Initial state
  let e_ema = 3.0, prec = 0.3, ach = 1.2, da = 1.0, ne = 0.9;
  let spec = 0.008, redun = 0.02;
  // Per-col route scores
  let scores = Array.from({length:n_cols}, () => 0.3 + rand()*0.4);
  let tuning  = Array.from({length:n_cols}, () => rand()*2-1);

  const log = [];
  for (let t = 0; t < N; t++) {
    const reward = (t % 100 < 50) ? 0.1 : 0.0;
    // Error declines slowly with noise
    const e_raw = 2.8 + 0.4*Math.sin(t*0.07) + randn()*0.35
                  - (t/N)*0.4 + (t < 80 ? (80-t)*0.015 : 0);
    e_ema = 0.85*e_ema + 0.15*e_raw;
    prec  = 0.95*prec + 0.05*(1/(e_ema+0.01));

    // Neuromod multi-timescale
    const ach_tgt = Math.min(2.5, Math.max(0.3, 0.5 + e_ema*0.5));
    const da_tgt  = Math.min(2.5, Math.max(0.1, 1.0 + reward));
    const ne_tgt  = Math.min(1.5, Math.max(0.1, e_ema*0.4));
    ach = 0.67*ach + 0.33*ach_tgt;
    da  = 0.93*da  + 0.07*da_tgt;
    ne  = 0.99*ne  + 0.01*ne_tgt;

    // Route scores — drift toward specialisation
    scores = scores.map((s,i) => {
      const noise = randn()*0.005;
      const drive = prec * (0.3 + 0.7*Math.abs(Math.sin(i*1.3 + t*0.01)));
      return Math.max(0.01, Math.min(1, 0.98*s + 0.02*drive + noise));
    });

    // Active = top-k by score + tuning
    const ranked = scores.map((s,i)=>({i,v:s+Math.max(0,tuning[i])*0.3}))
                         .sort((a,b)=>b.v-a.v);
    const active = Array(n_cols).fill(false);
    ranked.slice(0,k).forEach(x=>active[x.i]=true);

    // Spec index = std of active scores
    const activeScores = scores.filter((_,i)=>active[i]);
    const mean = activeScores.reduce((a,b)=>a+b,0)/activeScores.length;
    spec = 0.9*spec + 0.1*Math.sqrt(activeScores.reduce((a,b)=>a+(b-mean)**2,0)/activeScores.length);

    // Redundancy — decreases as cols specialise
    redun = Math.max(0.001, 0.97*redun + 0.03*(0.05 - 0.02*(t/N)));

    log.push({
      t,
      e_mag:        parseFloat(e_ema.toFixed(4)),
      spec_index:   parseFloat(spec.toFixed(5)),
      redundancy:   parseFloat(redun.toFixed(5)),
      n_active:     k,
      ach:          parseFloat(ach.toFixed(3)),
      da:           parseFloat(da.toFixed(3)),
      ne:           parseFloat(ne.toFixed(3)),
      route_scores: scores.map(x=>parseFloat(x.toFixed(4))),
      active,
    });
  }
  return log;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
const smooth = (arr, w=12) => arr.map((v,i) => {
  const sl = arr.slice(Math.max(0,i-w), i+1);
  return sl.reduce((a,b)=>a+b,0)/sl.length;
});

const fmt = (n, d=3) => typeof n==="number" ? n.toFixed(d) : "—";

// ─── Phosphor glow text ───────────────────────────────────────────────────────
const Glow = ({children, color=P.phos, size=11, bold=false, style={}}) => (
  <span style={{
    color, fontSize:size, fontWeight:bold?"700":"400",
    textShadow:`0 0 8px ${color}`,
    fontFamily:"'JetBrains Mono','Fira Code','Courier New',monospace",
    ...style,
  }}>{children}</span>
);

// ─── Oscilloscope-style mini chart ────────────────────────────────────────────
const OsciLine = ({data, dataKey, color=P.phos, height=80, domain, label}) => (
  <div style={{position:"relative"}}>
    {label && <div style={{
      position:"absolute",top:4,left:8,zIndex:2,
      color:P.textMid,fontSize:9,letterSpacing:"0.1em",fontFamily:"monospace",
    }}>{label}</div>}
    <div style={{background:P.surface, border:`1px solid ${P.border}`, borderRadius:3,
                 boxShadow:`inset 0 0 20px rgba(0,0,0,0.6)`}}>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{left:0,right:0,top:8,bottom:0}}>
          <CartesianGrid stroke={P.dim} strokeDasharray="1 8" vertical={false}/>
          {domain && <YAxis domain={domain} hide/>}
          <Line dataKey={dataKey} stroke={color} dot={false} strokeWidth={1.5}
                isAnimationActive={false}
                style={{filter:`drop-shadow(0 0 3px ${color})`}}/>
        </LineChart>
      </ResponsiveContainer>
    </div>
  </div>
);

// ─── Column grid — 12 columns, routing heatmap ───────────────────────────────
const ColumnGrid = ({routeScores, active, colLabels}) => {
  const max = Math.max(...routeScores, 0.001);
  return (
    <div style={{display:"flex", gap:4, alignItems:"flex-end", height:80}}>
      {routeScores.map((score, i) => {
        const h = Math.max(8, (score/max)*64);
        const isActive = active[i];
        return (
          <div key={i} style={{flex:1, display:"flex", flexDirection:"column",
                               alignItems:"center", gap:2}}>
            <div style={{
              width:"100%", height:h, borderRadius:"2px 2px 0 0",
              background: isActive
                ? `linear-gradient(to top, ${P.phos}, ${P.phosLo})`
                : `rgba(0,255,127,0.12)`,
              boxShadow: isActive ? `0 0 8px ${P.phos}, 0 0 2px ${P.phos}` : "none",
              transition:"height 0.3s ease, box-shadow 0.3s ease",
              borderTop: isActive ? `1px solid ${P.phos}` : `1px solid ${P.dim}`,
            }}/>
            <div style={{
              fontSize:7, color: isActive ? P.phos : P.textDim,
              fontFamily:"monospace", lineHeight:1,
            }}>
              {i.toString().padStart(2,"0")}
            </div>
          </div>
        );
      })}
    </div>
  );
};

// ─── Neuromod arcs ────────────────────────────────────────────────────────────
const NMBar = ({label, val, max=2.5, color}) => {
  const pct = Math.min(val/max, 1);
  return (
    <div style={{display:"flex", alignItems:"center", gap:8, marginBottom:5}}>
      <div style={{color, fontSize:9, fontFamily:"monospace", width:28,
                   textAlign:"right", textShadow:`0 0 6px ${color}`}}>{label}</div>
      <div style={{flex:1, height:5, background:P.border, borderRadius:3, overflow:"hidden"}}>
        <div style={{
          width:`${pct*100}%`, height:"100%", borderRadius:3,
          background: `linear-gradient(to right, ${color}88, ${color})`,
          boxShadow:`0 0 6px ${color}`,
          transition:"width 0.4s ease",
        }}/>
      </div>
      <div style={{color, fontSize:9, fontFamily:"monospace", width:32}}>{fmt(val,2)}</div>
    </div>
  );
};

// ─── Stat readout ─────────────────────────────────────────────────────────────
const Readout = ({label, value, unit="", color=P.phos, decimals=4}) => (
  <div style={{
    background:P.panel, border:`1px solid ${P.border}`, borderRadius:4,
    padding:"8px 12px", minWidth:110,
  }}>
    <div style={{color:P.textDim, fontSize:8, letterSpacing:"0.12em",
                 fontFamily:"monospace", marginBottom:4, textTransform:"uppercase"}}>{label}</div>
    <Glow color={color} size={20} bold style={{letterSpacing:"-0.02em"}}>
      {typeof value==="number" ? value.toFixed(decimals) : value}
    </Glow>
    {unit && <span style={{color:P.textMid, fontSize:9, marginLeft:3, fontFamily:"monospace"}}>{unit}</span>}
  </div>
);

// ─── Phase strip ─────────────────────────────────────────────────────────────
const PHASES = [
  {n:"2", label:"Dendritic Split",   sub:"ff / ctx streams"},
  {n:"3", label:"Inhibitory Control",sub:"k-winner sparsity"},
  {n:"4", label:"Multi-Column",      sub:"lateral + WTA"},
  {n:"5", label:"Thalamic Route",    sub:"top-k + tuning"},
  {n:"6", label:"Neuromod Ctrl",     sub:"ACh·DA·NE"},
];

const PhaseStrip = ({activePhase}) => (
  <div style={{display:"flex", gap:6, marginBottom:20, flexWrap:"wrap"}}>
    {PHASES.map(({n,label,sub}) => {
      const on = parseInt(n) <= activePhase;
      return (
        <div key={n} style={{
          border:`1px solid ${on ? P.phos : P.border}`,
          borderRadius:4, padding:"5px 10px",
          background: on ? P.phosGlow : "transparent",
          transition:"all 0.4s",
        }}>
          <div style={{display:"flex", gap:6, alignItems:"baseline"}}>
            <Glow color={on ? P.phos : P.textDim} size={10} bold>P{n}</Glow>
            <span style={{color: on ? P.textHi : P.textDim, fontSize:9,
                          fontFamily:"monospace"}}>{label}</span>
          </div>
          <div style={{color:P.textDim, fontSize:8, fontFamily:"monospace",
                       marginTop:1}}>{sub}</div>
        </div>
      );
    })}
  </div>
);

// ─── Main ─────────────────────────────────────────────────────────────────────
export default function App() {
  const [log]        = useState(() => generateTrainingLog());
  const [cursor, setCursor]   = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed]     = useState(4);   // steps per tick
  const rafRef = useRef(null);
  const lastTick = useRef(0);

  // Playback loop
  useEffect(() => {
    if (!playing) { cancelAnimationFrame(rafRef.current); return; }
    const tick = (now) => {
      if (now - lastTick.current > 60) {
        lastTick.current = now;
        setCursor(c => {
          const next = c + speed;
          if (next >= log.length) { setPlaying(false); return log.length - 1; }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, speed, log.length]);

  const frame     = log[cursor];
  const window50  = log.slice(Math.max(0, cursor - 49), cursor + 1);

  // Charts use a sliding 200-step window
  const chartWin  = 200;
  const chartData = (() => {
    const slice = log.slice(Math.max(0, cursor - chartWin + 1), cursor + 1);
    const emags  = slice.map(d => d.e_mag);
    const smoothed = smooth(emags, 10);
    return slice.map((d, i) => ({ ...d, e_smooth: smoothed[i] }));
  })();

  // Phase derived from cursor progress
  const progress    = cursor / log.length;
  const activePhase = progress < 0.15 ? 2 : progress < 0.35 ? 3
                    : progress < 0.55 ? 4 : progress < 0.75 ? 5 : 6;

  const meanErr50 = window50.length
    ? window50.reduce((a,b) => a + b.e_mag, 0) / window50.length : 0;

  return (
    <div style={{
      minHeight:"100vh", background:P.bg, color:P.textHi,
      fontFamily:"'JetBrains Mono','Fira Code',monospace",
      padding:"20px 16px",
      backgroundImage: P.scanline,
    }}>
      {/* ── Header ── */}
      <div style={{marginBottom:16, display:"flex", alignItems:"baseline",
                   justifyContent:"space-between", flexWrap:"wrap", gap:8}}>
        <div>
          <div style={{display:"flex", alignItems:"baseline", gap:16}}>
            <Glow size={16} bold style={{letterSpacing:"0.15em", textTransform:"uppercase"}}>
              CORTICAL MICROCIRCUIT
            </Glow>
            <Glow color={P.textMid} size={10}>spec v0.4 · phases 2–6</Glow>
          </div>
          <div style={{color:P.textDim, fontSize:9, marginTop:3, letterSpacing:"0.08em"}}>
            12-column sheet · PV/SST/VIP inhibition · WTA thalamic routing · local Hebbian · no backprop
          </div>
        </div>
        <div style={{
          border:`1px solid ${P.border}`, borderRadius:4, padding:"4px 12px",
          background:P.panel,
        }}>
          <Glow size={11} bold>t = {cursor.toString().padStart(4,"0")}</Glow>
          <Glow color={P.textMid} size={9}> / {log.length}</Glow>
        </div>
      </div>

      {/* ── Phase strip ── */}
      <PhaseStrip activePhase={activePhase} />

      {/* ── Scrubber + controls ── */}
      <div style={{
        background:P.panel, border:`1px solid ${P.border}`, borderRadius:6,
        padding:"10px 14px", marginBottom:16,
        display:"flex", alignItems:"center", gap:12, flexWrap:"wrap",
      }}>
        <button onClick={() => setPlaying(p => !p)} style={{
          background: playing ? "rgba(255,61,87,0.15)" : P.phosGlow,
          border:`1px solid ${playing ? P.red : P.phos}`,
          color: playing ? P.red : P.phos,
          borderRadius:4, padding:"5px 16px", cursor:"pointer",
          fontFamily:"monospace", fontSize:11, fontWeight:700,
          boxShadow: playing ? `0 0 8px ${P.red}44` : `0 0 8px ${P.phos}44`,
        }}>
          {playing ? "▪ STOP" : cursor === 0 ? "▶ PLAY" : "▶ RESUME"}
        </button>
        <button onClick={() => { setPlaying(false); setCursor(0); }} style={{
          background:"transparent", border:`1px solid ${P.border}`,
          color:P.textMid, borderRadius:4, padding:"5px 12px", cursor:"pointer",
          fontFamily:"monospace", fontSize:10,
        }}>RESET</button>

        <input type="range" min={0} max={log.length - 1} value={cursor}
               onChange={e => { setPlaying(false); setCursor(parseInt(e.target.value)); }}
               style={{flex:1, accentColor:P.phos, minWidth:120}} />

        <div style={{display:"flex", alignItems:"center", gap:6}}>
          <span style={{color:P.textDim, fontSize:9}}>SPEED</span>
          {[1,4,10,20].map(s => (
            <button key={s} onClick={() => setSpeed(s)} style={{
              background: speed===s ? P.phosGlow : "transparent",
              border:`1px solid ${speed===s ? P.phos : P.border}`,
              color: speed===s ? P.phos : P.textDim,
              borderRadius:3, padding:"3px 7px", cursor:"pointer",
              fontFamily:"monospace", fontSize:9,
            }}>{s}×</button>
          ))}
        </div>
      </div>

      {/* ── Main grid ── */}
      <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:12}}>

        {/* LEFT COL */}
        <div style={{display:"flex", flexDirection:"column", gap:12}}>

          {/* Prediction error */}
          <div style={{background:P.panel, border:`1px solid ${P.border}`, borderRadius:6, padding:12}}>
            <div style={{color:P.textMid, fontSize:9, letterSpacing:"0.12em",
                         marginBottom:8}}>PREDICTION ERROR  e_local = x − ŷ</div>
            <OsciLine data={chartData} dataKey="e_mag"    color={`${P.phos}44`} height={100}/>
            <div style={{marginTop:-100, position:"relative", pointerEvents:"none"}}>
              <OsciLine data={chartData} dataKey="e_smooth" color={P.phos}    height={100}/>
            </div>
            <div style={{display:"flex", gap:8, marginTop:8}}>
              <Readout label="current" value={frame?.e_mag}   color={frame?.e_mag < 2.5 ? P.phos : P.amber} decimals={4}/>
              <Readout label="mean-50" value={meanErr50}      color={P.textMid} decimals={4}/>
              <Readout label="precision" value={frame ? 1/(frame.e_mag+0.001) : 0}
                       color={P.cyan} decimals={3}/>
            </div>
          </div>

          {/* Column routing grid */}
          <div style={{background:P.panel, border:`1px solid ${P.border}`, borderRadius:6, padding:12}}>
            <div style={{color:P.textMid, fontSize:9, letterSpacing:"0.12em", marginBottom:10}}>
              THALAMIC ROUTING — 12 COLUMNS · k={4} ACTIVE
            </div>
            <ColumnGrid
              routeScores={frame?.route_scores ?? Array(12).fill(0.5)}
              active={frame?.active ?? Array(12).fill(false)}
            />
            <div style={{
              display:"flex", justifyContent:"space-between", marginTop:6,
              color:P.textDim, fontSize:8, fontFamily:"monospace",
            }}>
              <span>COL 00</span>
              <span style={{color:P.phos}}>▐ active  □ silent</span>
              <span>COL 11</span>
            </div>
          </div>

          {/* Specialisation + redundancy */}
          <div style={{background:P.panel, border:`1px solid ${P.border}`, borderRadius:6, padding:12}}>
            <div style={{color:P.textMid, fontSize:9, letterSpacing:"0.12em", marginBottom:8}}>
              SPECIALISATION INDEX · REDUNDANCY
            </div>
            <OsciLine data={chartData} dataKey="spec_index" color={P.violet} height={70}
                      label="spec_index"/>
            <div style={{height:6}}/>
            <OsciLine data={chartData} dataKey="redundancy"  color={P.amber}  height={70}
                      label="redundancy"/>
          </div>

        </div>

        {/* RIGHT COL */}
        <div style={{display:"flex", flexDirection:"column", gap:12}}>

          {/* Neuromodulatory state */}
          <div style={{background:P.panel, border:`1px solid ${P.border}`, borderRadius:6, padding:12}}>
            <div style={{color:P.textMid, fontSize:9, letterSpacing:"0.12em", marginBottom:12}}>
              NEUROMODULATORY STATE  ·  3 TIMESCALES
            </div>
            <NMBar label="ACh" val={frame?.ach ?? 1} color={P.cyan}   max={2.5}/>
            <NMBar label="DA"  val={frame?.da  ?? 1} color={P.phos}   max={2.5}/>
            <NMBar label="NE"  val={frame?.ne  ?? 0.5} color={P.amber} max={1.5}/>
            <div style={{
              display:"grid", gridTemplateColumns:"1fr 1fr 1fr",
              gap:6, marginTop:12,
            }}>
              <div style={{textAlign:"center"}}>
                <div style={{color:P.textDim, fontSize:8, marginBottom:3}}>τ_ACh</div>
                <Glow color={P.cyan} size={10}>~3 steps</Glow>
              </div>
              <div style={{textAlign:"center"}}>
                <div style={{color:P.textDim, fontSize:8, marginBottom:3}}>τ_DA</div>
                <Glow color={P.phos} size={10}>~15 steps</Glow>
              </div>
              <div style={{textAlign:"center"}}>
                <div style={{color:P.textDim, fontSize:8, marginBottom:3}}>τ_NE</div>
                <Glow color={P.amber} size={10}>~100 steps</Glow>
              </div>
            </div>
          </div>

          {/* Neuromod time series */}
          <div style={{background:P.panel, border:`1px solid ${P.border}`, borderRadius:6, padding:12}}>
            <div style={{color:P.textMid, fontSize:9, letterSpacing:"0.12em", marginBottom:8}}>
              NEUROMOD TRACES
            </div>
            <div style={{position:"relative", height:120}}>
              <ResponsiveContainer width="100%" height={120}>
                <LineChart data={chartData} margin={{left:0,right:0,top:4,bottom:0}}>
                  <CartesianGrid stroke={P.dim} strokeDasharray="1 8" vertical={false}/>
                  <YAxis domain={[0,2.5]} hide/>
                  <Line dataKey="ach" stroke={P.cyan}   dot={false} strokeWidth={1.5}
                        isAnimationActive={false} name="ACh"
                        style={{filter:`drop-shadow(0 0 3px ${P.cyan})`}}/>
                  <Line dataKey="da"  stroke={P.phos}   dot={false} strokeWidth={1.5}
                        isAnimationActive={false} name="DA"
                        style={{filter:`drop-shadow(0 0 3px ${P.phos})`}}/>
                  <Line dataKey="ne"  stroke={P.amber}  dot={false} strokeWidth={1.5}
                        isAnimationActive={false} name="NE"
                        style={{filter:`drop-shadow(0 0 3px ${P.amber})`}}/>
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div style={{display:"flex", gap:12, justifyContent:"center", marginTop:6}}>
              {[["ACh",P.cyan],["DA",P.phos],["NE",P.amber]].map(([l,c])=>(
                <div key={l} style={{display:"flex",alignItems:"center",gap:4}}>
                  <div style={{width:16,height:2,background:c,boxShadow:`0 0 4px ${c}`}}/>
                  <Glow color={c} size={9}>{l}</Glow>
                </div>
              ))}
            </div>
          </div>

          {/* Phase 2 detail — stream separation */}
          <div style={{background:P.panel, border:`1px solid ${P.border}`, borderRadius:6, padding:12}}>
            <div style={{color:P.textMid, fontSize:9, letterSpacing:"0.12em", marginBottom:10}}>
              PHASE 2 · DENDRITIC STREAMS
            </div>
            <div style={{display:"flex", gap:8}}>
              <div style={{flex:1, border:`1px solid ${P.border}`, borderRadius:4, padding:10,
                           background:P.surface, textAlign:"center"}}>
                <div style={{color:P.textDim, fontSize:8, marginBottom:6}}>BASAL</div>
                <div style={{
                  width:32, height:32, borderRadius:"50%", margin:"0 auto 6px",
                  background:`radial-gradient(circle, ${P.phos}44, transparent)`,
                  border:`1px solid ${P.phos}`,
                  display:"flex", alignItems:"center", justifyContent:"center",
                }}>
                  <Glow size={7}>ff</Glow>
                </div>
                <div style={{color:P.textDim, fontSize:8}}>x_bottom →</div>
                <div style={{color:P.phos, fontSize:8}}>evidence</div>
              </div>
              <div style={{display:"flex", alignItems:"center", padding:"0 4px"}}>
                <div style={{
                  width:28, height:28, borderRadius:"50%",
                  background:`radial-gradient(circle, ${P.phos}22, transparent)`,
                  border:`1px solid ${P.phosLo}`,
                  display:"flex", alignItems:"center", justifyContent:"center",
                }}>
                  <Glow size={9} bold>×</Glow>
                </div>
              </div>
              <div style={{flex:1, border:`1px solid ${P.border}`, borderRadius:4, padding:10,
                           background:P.surface, textAlign:"center"}}>
                <div style={{color:P.textDim, fontSize:8, marginBottom:6}}>APICAL</div>
                <div style={{
                  width:32, height:32, borderRadius:"50%", margin:"0 auto 6px",
                  background:`radial-gradient(circle, ${P.violet}44, transparent)`,
                  border:`1px solid ${P.violet}`,
                  display:"flex", alignItems:"center", justifyContent:"center",
                }}>
                  <Glow color={P.violet} size={7}>ctx</Glow>
                </div>
                <div style={{color:P.textDim, fontSize:8}}>x_top →</div>
                <Glow color={P.violet} size={8}>expectation gate</Glow>
              </div>
              <div style={{display:"flex", alignItems:"center", padding:"0 4px"}}>
                <Glow size={9} bold>=</Glow>
              </div>
              <div style={{flex:1, border:`1px solid ${P.phos}`, borderRadius:4, padding:10,
                           background:`rgba(0,255,127,0.04)`, textAlign:"center"}}>
                <div style={{color:P.textDim, fontSize:8, marginBottom:6}}>SOMA</div>
                <div style={{
                  width:32, height:32, borderRadius:"50%", margin:"0 auto 6px",
                  background:`radial-gradient(circle, ${P.phos}55, transparent)`,
                  border:`1px solid ${P.phos}`,
                  boxShadow:`0 0 10px ${P.phos}44`,
                  display:"flex", alignItems:"center", justifyContent:"center",
                }}>
                  <Glow size={7} bold>∑</Glow>
                </div>
                <div style={{color:P.textDim, fontSize:8}}>coincidence</div>
                <Glow size={8}>detection</Glow>
              </div>
            </div>
          </div>

          {/* Stats row */}
          <div style={{display:"flex", gap:8, flexWrap:"wrap"}}>
            <Readout label="n_active"    value={frame?.n_active ?? 0}   color={P.phos}   decimals={0}/>
            <Readout label="spec_idx"    value={frame?.spec_index ?? 0} color={P.violet} decimals={4}/>
            <Readout label="redundancy"  value={frame?.redundancy ?? 0} color={P.amber}  decimals={4}/>
          </div>

        </div>
      </div>

      {/* ── Full-width phase annotations ── */}
      <div style={{
        marginTop:12, background:P.panel, border:`1px solid ${P.border}`,
        borderRadius:6, padding:"10px 14px",
      }}>
        <div style={{
          display:"flex", gap:0, position:"relative", height:20,
          background:P.surface, borderRadius:3, overflow:"hidden",
        }}>
          {/* Progress bar */}
          <div style={{
            position:"absolute", left:0, top:0, height:"100%",
            width:`${(cursor/log.length)*100}%`,
            background:`linear-gradient(to right, ${P.phos}22, ${P.phos}44)`,
            borderRight:`1px solid ${P.phos}`,
            transition:"width 0.1s linear",
          }}/>
          {/* Phase markers */}
          {[
            {x:0.15, label:"P3"},
            {x:0.35, label:"P4"},
            {x:0.55, label:"P5"},
            {x:0.75, label:"P6"},
          ].map(({x,label}) => (
            <div key={label} style={{
              position:"absolute", left:`${x*100}%`, top:0, height:"100%",
              borderLeft:`1px dashed ${P.border}`,
              display:"flex", alignItems:"center", paddingLeft:4,
            }}>
              <span style={{color:P.textDim, fontSize:8, fontFamily:"monospace"}}>{label}</span>
            </div>
          ))}
          <div style={{
            position:"absolute", right:8, top:0, height:"100%",
            display:"flex", alignItems:"center",
          }}>
            <Glow color={P.textDim} size={8}>t={cursor}</Glow>
          </div>
        </div>
        <div style={{
          display:"flex", justifyContent:"space-between", marginTop:4,
          color:P.textDim, fontSize:8, fontFamily:"monospace",
        }}>
          <span>DENDRITIC SPLIT</span>
          <span>INHIBITORY</span>
          <span>MULTI-COL</span>
          <span>ROUTING</span>
          <span>NEUROMOD</span>
        </div>
      </div>

      {/* ── Footer ── */}
      <div style={{
        marginTop:10, color:P.textDim, fontSize:8,
        fontFamily:"monospace", letterSpacing:"0.06em",
        textAlign:"center", lineHeight:1.8,
      }}>
        P2: ff/ctx split · P3: k-winner sparsity [75%] · P4: WTA lateral specialisation ·
        P5: thalamic Oja tuning k=4/12 · P6: ACh[τ3]·DA[τ15]·NE[τ100] · no global backprop
      </div>
    </div>
  );
}
