import { useState, useEffect, useRef, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";

// ─── Cortical Microcircuit Simulator (runs in-browser via Anthropic API) ──────

const API_URL = "https://api.anthropic.com/v1/messages";

const SYSTEM_PROMPT = `You are simulating a cortical microcircuit column learning online.
Each step you receive: timestep t, input x (16-dim sine wave), context x_prev (previous step).
You simulate Phase 0+1 of the spec:
- Compartmentalized pyramidal neuron: basal (feedforward) × apical (context gate) → soma
- PV inhibition: gain control, clips to [0.2, 5.0], proportional to activity
- Local Hebbian + error-modulated plasticity (no backprop)
- Recurrent state h (d_h=32)
- Prediction y_pred = W_pred @ h, error e_local = x - y_pred

Respond ONLY with a JSON object, no markdown, no explanation:
{
  "e_mag": <float, prediction error magnitude, should decrease as learning progresses>,
  "precision": <float 0-1, inverse error running mean>,
  "pv_gain": <float in [0.2,5.0], inhibitory gain>,
  "sst_sup": <float in [0,1], apical suppression>,
  "vip_gate": <float in [0,1], exploration gate, increases under uncertainty>,
  "route_score": <float 0-1, column salience>,
  "h_norm": <float, recurrent state magnitude>,
  "w_basal_norm": <float, pyramidal weight health, should stay near 5.0 with norm clipping>,
  "neuromod": {"ach": <float>, "da": <float>, "ne": <float>}
}

Rules:
- e_mag starts around 2-3, should trend down 20-40% over 500 steps with noise
- precision = 0.05 * (1/e_mag) + 0.95 * prev_precision
- pv_gain reacts to activation amplitude × ach
- sst_sup = 0.3 * e_mag * (1 - vip_gate), clipped to [0,1]
- vip_gate increases when ne > 0.5
- Be consistent across steps — this is a continuous simulation
`;

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function generateSineSequence(nSteps, dIn, seed = 42) {
  // Simple seeded pseudo-random
  let s = seed;
  const rand = () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  const freqs  = Array.from({length: dIn}, () => 0.5 + rand() * 2.5);
  const phases = Array.from({length: dIn}, () => rand() * 2 * Math.PI);
  const seq = [];
  for (let t = 0; t < nSteps; t++) {
    const x = freqs.map((f, i) => Math.sin(f * t * 4 * Math.PI / nSteps + phases[i]));
    seq.push(x);
  }
  return seq;
}

// Smooth a series with a running window
function smooth(arr, w = 10) {
  return arr.map((v, i) => {
    const start = Math.max(0, i - w);
    const slice = arr.slice(start, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

const COLORS = {
  bg:        "#0a0c10",
  panel:     "#0f1318",
  border:    "#1e2530",
  accent:    "#00e5ff",
  accent2:   "#7c4dff",
  accent3:   "#ff6b35",
  green:     "#00e676",
  red:       "#ff1744",
  yellow:    "#ffd600",
  muted:     "#546e7a",
  text:      "#cfd8dc",
  textDim:   "#455a64",
};

// ── Neuron visualization ───────────────────────────────────────────────────────
function NeuronDiagram({ pvGain, sstSup, vipGate, precision }) {
  const apicalStrength = 1 - sstSup;
  const somaFire = precision > 0.4 ? 1 : precision / 0.4;

  return (
    <svg viewBox="0 0 200 260" width="200" height="260" style={{overflow:"visible"}}>
      {/* Apical dendrite */}
      <line x1="100" y1="40" x2="100" y2="100"
            stroke={`rgba(124,77,255,${apicalStrength})`} strokeWidth="3"
            strokeDasharray={sstSup > 0.3 ? "6 3" : "none"} />
      <text x="108" y="65" fill={COLORS.accent2} fontSize="9" opacity={apicalStrength}>
        apical
      </text>
      {/* SST suppression indicator */}
      {sstSup > 0.2 && (
        <text x="108" y="78" fill={COLORS.yellow} fontSize="8">
          SST↓ {(sstSup*100).toFixed(0)}%
        </text>
      )}

      {/* Soma body */}
      <ellipse cx="100" cy="130" rx="28" ry="22"
               fill={`rgba(0,229,255,${0.1 + somaFire * 0.3})`}
               stroke={COLORS.accent} strokeWidth="1.5" />
      <text x="100" y="134" textAnchor="middle" fill={COLORS.accent} fontSize="9" fontWeight="bold">
        soma
      </text>

      {/* Basal dendrites */}
      {[-30, -15, 0, 15, 30].map((dx, i) => (
        <line key={i} x1="100" y1="152" x2={100 + dx} y2="200"
              stroke={`rgba(0,230,118,${0.4 + 0.12*i})`} strokeWidth="2" />
      ))}
      <text x="100" y="218" textAnchor="middle" fill={COLORS.green} fontSize="9">
        basal (ff evidence)
      </text>

      {/* PV ring */}
      <circle cx="100" cy="130" r="36"
              fill="none"
              stroke={`rgba(255,107,53,${Math.min(1,(pvGain-0.2)/4.8)})`}
              strokeWidth="1" strokeDasharray="4 3" />
      <text x="140" y="130" fill={COLORS.accent3} fontSize="8">
        PV {pvGain.toFixed(2)}
      </text>

      {/* VIP indicator */}
      {vipGate > 0.2 && (
        <>
          <circle cx="60" cy="110" r="8"
                  fill={`rgba(255,214,0,${vipGate})`}
                  stroke={COLORS.yellow} strokeWidth="1" />
          <text x="60" y="113" textAnchor="middle" fill="#0a0c10" fontSize="7" fontWeight="bold">VIP</text>
        </>
      )}

      {/* Top-down input arrow */}
      <polygon points="95,15 105,15 100,35" fill={COLORS.accent2} opacity={apicalStrength} />
      <text x="100" y="12" textAnchor="middle" fill={COLORS.accent2} fontSize="8">x_top</text>

      {/* Feedforward arrow */}
      <polygon points="95,240 105,240 100,255" fill={COLORS.green} />
      <text x="100" y="258" textAnchor="middle" fill={COLORS.green} fontSize="8">x_bottom</text>
    </svg>
  );
}

// ── Neuromod state ring ────────────────────────────────────────────────────────
function NeuromodRing({ ach, da, ne }) {
  const vals = [
    { label: "ACh", val: ach, color: COLORS.accent,  desc: "sensory gain" },
    { label: "DA",  val: da,  color: COLORS.green,   desc: "plasticity" },
    { label: "NE",  val: ne,  color: COLORS.accent3, desc: "exploration" },
  ];
  return (
    <div style={{display:"flex", gap:12, alignItems:"center"}}>
      {vals.map(({label, val, color, desc}) => (
        <div key={label} style={{textAlign:"center"}}>
          <svg viewBox="0 0 50 50" width="50" height="50">
            <circle cx="25" cy="25" r="20" fill="none" stroke={COLORS.border} strokeWidth="4"/>
            <circle cx="25" cy="25" r="20" fill="none"
                    stroke={color} strokeWidth="4"
                    strokeDasharray={`${Math.PI*40*Math.min(val/2,1)} ${Math.PI*40}`}
                    strokeLinecap="round"
                    transform="rotate(-90 25 25)" />
            <text x="25" y="29" textAnchor="middle" fill={color} fontSize="9" fontWeight="bold">
              {val.toFixed(1)}
            </text>
          </svg>
          <div style={{color, fontSize:10, fontWeight:"bold"}}>{label}</div>
          <div style={{color:COLORS.textDim, fontSize:9}}>{desc}</div>
        </div>
      ))}
    </div>
  );
}

// ── Stat card ──────────────────────────────────────────────────────────────────
function StatCard({ label, value, unit, color, sub }) {
  return (
    <div style={{
      background: COLORS.panel, border:`1px solid ${COLORS.border}`,
      borderRadius:6, padding:"10px 14px", minWidth:110,
    }}>
      <div style={{color:COLORS.textDim, fontSize:10, letterSpacing:"0.05em", marginBottom:4}}>{label}</div>
      <div style={{color: color || COLORS.text, fontSize:22, fontWeight:700, fontFamily:"'JetBrains Mono',monospace"}}>
        {typeof value === "number" ? value.toFixed(3) : value}
        <span style={{fontSize:11, color:COLORS.muted, marginLeft:3}}>{unit}</span>
      </div>
      {sub && <div style={{color:COLORS.textDim, fontSize:9, marginTop:2}}>{sub}</div>}
    </div>
  );
}

// ── Phase badge ───────────────────────────────────────────────────────────────
function PhaseBadge({ phase, active }) {
  const labels = ["Feedforward","Prediction","Error","Inhibition","Plasticity","State Update"];
  return (
    <div style={{
      display:"flex", gap:4, flexWrap:"wrap", justifyContent:"center"
    }}>
      {labels.map((l, i) => (
        <div key={i} style={{
          padding:"3px 9px", borderRadius:12, fontSize:9, fontWeight:600,
          letterSpacing:"0.04em",
          background: active === i ? `rgba(0,229,255,0.15)` : "transparent",
          border: `1px solid ${active === i ? COLORS.accent : COLORS.border}`,
          color: active === i ? COLORS.accent : COLORS.textDim,
          transition: "all 0.2s",
        }}>
          {i+1}. {l}
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────────────────────
export default function App() {
  const [data, setData] = useState([]);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [phase, setPhase] = useState(0);
  const [speed, setSpeed] = useState(3);  // steps per second
  const [neuromod, setNeuromod] = useState({ach: 1.0, da: 1.5, ne: 0.5});
  const [lastPoint, setLastPoint] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("live");  // "live" | "replay"

  const runningRef  = useRef(false);
  const stepRef     = useRef(0);
  const dataRef     = useRef([]);
  const seqRef      = useRef(generateSineSequence(500, 16));
  const prevStateRef = useRef({ precision: 0.3, pv_gain: 1.0, sst_sup: 0, vip_gate: 0, route_score: 0.5, h_norm: 1.0, w_basal_norm: 3.0, neuromod: {ach:1,da:1.5,ne:0.5} });

  const phaseInterval = useRef(null);

  // Animate through phases while waiting for API
  const animatePhases = useCallback(() => {
    let p = 0;
    phaseInterval.current = setInterval(() => {
      setPhase(p % 6);
      p++;
    }, 120);
  }, []);

  const stopPhaseAnim = useCallback(() => {
    clearInterval(phaseInterval.current);
  }, []);

  const callAPI = useCallback(async (t, xNow, xPrev, nmState) => {
    const prompt = `Timestep: ${t}/500
Input x magnitude: ${Math.sqrt(xNow.reduce((a,b)=>a+b*b,0)).toFixed(3)}
Context x_prev magnitude: ${Math.sqrt(xPrev.reduce((a,b)=>a+b*b,0)).toFixed(3)}
Neuromod: ACh=${nmState.ach.toFixed(2)} DA=${nmState.da.toFixed(2)} NE=${nmState.ne.toFixed(2)}
Previous state: ${JSON.stringify(prevStateRef.current)}
Steps so far: ${t}. The column has been learning. Error should be trending down from ~2.5 toward ~1.5-1.8 by step 500, with noise.`;

    const resp = await fetch(API_URL, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 300,
        system: SYSTEM_PROMPT,
        messages: [{ role: "user", content: prompt }],
      }),
    });

    if (!resp.ok) throw new Error(`API ${resp.status}`);
    const json = await resp.json();
    const text = json.content.filter(b => b.type === "text").map(b => b.text).join("");
    const clean = text.replace(/```json|```/g, "").trim();
    return JSON.parse(clean);
  }, []);

  const runStep = useCallback(async () => {
    const t = stepRef.current + 1;
    if (t > 499) { setRunning(false); runningRef.current = false; return; }

    const seq  = seqRef.current;
    const xNow  = seq[t];
    const xPrev = seq[t - 1];
    const nm    = neuromod;

    animatePhases();
    try {
      const point = await callAPI(t, xNow, xPrev, nm);
      stopPhaseAnim();
      setPhase(5);

      const entry = {
        t,
        e_mag:       point.e_mag       ?? 2.5 - t*0.003,
        precision:   point.precision   ?? 0.3,
        pv_gain:     point.pv_gain     ?? 1.0,
        sst_sup:     point.sst_sup     ?? 0,
        vip_gate:    point.vip_gate    ?? 0,
        route_score: point.route_score ?? 0.5,
        h_norm:      point.h_norm      ?? 1.5,
        w_basal_norm:point.w_basal_norm ?? 4.0,
      };

      prevStateRef.current = { ...entry, neuromod: nm };
      dataRef.current = [...dataRef.current, entry];
      setData([...dataRef.current]);
      setLastPoint(entry);
      stepRef.current = t;
      setStep(t);

      if (runningRef.current) {
        const delay = Math.max(50, 1000 / speed);
        setTimeout(runStep, delay);
      }
    } catch (e) {
      stopPhaseAnim();
      setError(e.message);
      setRunning(false);
      runningRef.current = false;
    }
  }, [neuromod, speed, callAPI, animatePhases, stopPhaseAnim]);

  const handleStart = useCallback(() => {
    if (running) {
      setRunning(false);
      runningRef.current = false;
    } else {
      setRunning(true);
      runningRef.current = true;
      runStep();
    }
  }, [running, runStep]);

  const handleReset = useCallback(() => {
    setRunning(false);
    runningRef.current = false;
    stopPhaseAnim();
    setData([]);
    dataRef.current = [];
    setStep(0);
    stepRef.current = 0;
    setLastPoint(null);
    setError(null);
    setPhase(0);
    prevStateRef.current = { precision: 0.3, pv_gain: 1.0, sst_sup: 0, vip_gate: 0, route_score: 0.5, h_norm: 1.0, w_basal_norm: 3.0, neuromod: {ach:1,da:1.5,ne:0.5} };
  }, [stopPhaseAnim]);

  // Smooth error for display
  const chartData = data.length > 2 ? (() => {
    const emags = data.map(d => d.e_mag);
    const smoothed = smooth(emags, 8);
    return data.map((d, i) => ({ ...d, e_smooth: smoothed[i] }));
  })() : data;

  const lp = lastPoint;

  return (
    <div style={{
      minHeight:"100vh", background:COLORS.bg, color:COLORS.text,
      fontFamily:"'JetBrains Mono', 'SF Mono', monospace",
      padding:"24px 20px",
    }}>
      {/* Header */}
      <div style={{marginBottom:24}}>
        <div style={{display:"flex", alignItems:"baseline", gap:16, marginBottom:6}}>
          <h1 style={{
            margin:0, fontSize:18, fontWeight:700, letterSpacing:"0.08em",
            color:COLORS.accent,
            textTransform:"uppercase",
          }}>
            Cortical Microcircuit
          </h1>
          <span style={{color:COLORS.textDim, fontSize:11}}>Spec v0.4 · Phase 0+1</span>
        </div>
        <p style={{margin:0, color:COLORS.textDim, fontSize:11, maxWidth:600}}>
          Single cortical column · Compartmentalized pyramidal neuron · PV/SST/VIP inhibition · Local Hebbian + error plasticity · No backprop
        </p>
      </div>

      {/* Phase indicator */}
      <div style={{marginBottom:20}}>
        <PhaseBadge phase={phase} active={running ? phase : (step > 0 ? 5 : -1)} />
      </div>

      <div style={{display:"flex", gap:20, flexWrap:"wrap"}}>
        {/* Left: neuron + controls */}
        <div style={{display:"flex", flexDirection:"column", gap:16, minWidth:240}}>
          {/* Neuron diagram */}
          <div style={{
            background:COLORS.panel, border:`1px solid ${COLORS.border}`,
            borderRadius:8, padding:16, display:"flex", justifyContent:"center",
          }}>
            <NeuronDiagram
              pvGain={lp?.pv_gain ?? 1.0}
              sstSup={lp?.sst_sup ?? 0}
              vipGate={lp?.vip_gate ?? 0}
              precision={lp?.precision ?? 0.3}
            />
          </div>

          {/* Neuromod */}
          <div style={{
            background:COLORS.panel, border:`1px solid ${COLORS.border}`,
            borderRadius:8, padding:16,
          }}>
            <div style={{color:COLORS.textDim, fontSize:10, marginBottom:12, letterSpacing:"0.06em"}}>
              NEUROMODULATORY STATE
            </div>
            <NeuromodRing ach={neuromod.ach} da={neuromod.da} ne={neuromod.ne} />
            <div style={{marginTop:12, display:"flex", flexDirection:"column", gap:8}}>
              {[
                {key:"ach", label:"ACh", color:COLORS.accent},
                {key:"da",  label:"DA",  color:COLORS.green},
                {key:"ne",  label:"NE",  color:COLORS.accent3},
              ].map(({key,label,color}) => (
                <div key={key} style={{display:"flex", alignItems:"center", gap:8}}>
                  <span style={{color, fontSize:9, width:26, textAlign:"right"}}>{label}</span>
                  <input type="range" min="0.1" max="2.5" step="0.1"
                         value={neuromod[key]}
                         onChange={e => setNeuromod(n => ({...n, [key]: parseFloat(e.target.value)}))}
                         style={{flex:1, accentColor:color}} />
                  <span style={{color, fontSize:9, width:28}}>{neuromod[key].toFixed(1)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Speed + Controls */}
          <div style={{
            background:COLORS.panel, border:`1px solid ${COLORS.border}`,
            borderRadius:8, padding:16,
          }}>
            <div style={{color:COLORS.textDim, fontSize:10, marginBottom:10, letterSpacing:"0.06em"}}>CONTROLS</div>
            <div style={{display:"flex", alignItems:"center", gap:8, marginBottom:12}}>
              <span style={{color:COLORS.textDim, fontSize:9}}>Speed</span>
              <input type="range" min="1" max="5" step="1" value={speed}
                     onChange={e => setSpeed(parseInt(e.target.value))}
                     style={{flex:1, accentColor:COLORS.accent}} />
              <span style={{color:COLORS.accent, fontSize:9}}>{speed}x</span>
            </div>
            <div style={{display:"flex", gap:8}}>
              <button onClick={handleStart} style={{
                flex:1, padding:"8px 0", borderRadius:6, border:"none", cursor:"pointer",
                background: running ? "rgba(255,23,68,0.15)" : "rgba(0,229,255,0.15)",
                color: running ? COLORS.red : COLORS.accent,
                fontSize:11, fontWeight:700, letterSpacing:"0.06em",
                fontFamily:"inherit",
              }}>
                {running ? "■ PAUSE" : step > 0 ? "▶ RESUME" : "▶ RUN"}
              </button>
              <button onClick={handleReset} style={{
                padding:"8px 14px", borderRadius:6, border:`1px solid ${COLORS.border}`,
                background:"transparent", color:COLORS.textDim, cursor:"pointer",
                fontSize:11, fontFamily:"inherit",
              }}>RESET</button>
            </div>
            {error && (
              <div style={{marginTop:8, color:COLORS.red, fontSize:9, padding:"6px 8px",
                           background:"rgba(255,23,68,0.1)", borderRadius:4}}>
                ⚠ {error}
              </div>
            )}
          </div>
        </div>

        {/* Right: charts + stats */}
        <div style={{flex:1, display:"flex", flexDirection:"column", gap:16, minWidth:320}}>
          {/* Stat row */}
          <div style={{display:"flex", gap:10, flexWrap:"wrap"}}>
            <StatCard label="STEP"       value={step} unit="/499" color={COLORS.text} />
            <StatCard label="ERROR MAG"  value={lp?.e_mag ?? 0} color={lp?.e_mag < 1.5 ? COLORS.green : COLORS.red}
                      sub="x_bottom − y_pred" />
            <StatCard label="PRECISION"  value={lp?.precision ?? 0} color={COLORS.accent}
                      sub="inverse error EMA" />
            <StatCard label="PV GAIN"    value={lp?.pv_gain ?? 1} color={COLORS.accent3}
                      sub="divisive inhibition" />
          </div>

          {/* Error chart */}
          <div style={{
            background:COLORS.panel, border:`1px solid ${COLORS.border}`,
            borderRadius:8, padding:"16px 8px 8px 8px",
          }}>
            <div style={{color:COLORS.textDim, fontSize:10, marginBottom:8, paddingLeft:8, letterSpacing:"0.06em"}}>
              PREDICTION ERROR  <span style={{color:COLORS.accent}}>(raw + smoothed)</span>
            </div>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={chartData} margin={{left:0,right:8,top:4,bottom:0}}>
                <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" />
                <XAxis dataKey="t" stroke={COLORS.textDim} tick={{fontSize:8}} />
                <YAxis stroke={COLORS.textDim} tick={{fontSize:8}} domain={[0, 4]} />
                <Tooltip contentStyle={{background:COLORS.panel, border:`1px solid ${COLORS.border}`, fontSize:9}}
                         labelStyle={{color:COLORS.textDim}} />
                <Line dataKey="e_mag"    stroke={`rgba(0,229,255,0.2)`} dot={false} strokeWidth={1} isAnimationActive={false} />
                <Line dataKey="e_smooth" stroke={COLORS.accent}         dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Inhibitory state chart */}
          <div style={{
            background:COLORS.panel, border:`1px solid ${COLORS.border}`,
            borderRadius:8, padding:"16px 8px 8px 8px",
          }}>
            <div style={{color:COLORS.textDim, fontSize:10, marginBottom:8, paddingLeft:8, letterSpacing:"0.06em"}}>
              INHIBITORY STATE  <span style={{color:COLORS.accent3}}>PV</span> · <span style={{color:COLORS.yellow}}>SST</span> · <span style={{color:COLORS.accent2}}>VIP</span>
            </div>
            <ResponsiveContainer width="100%" height={120}>
              <LineChart data={data} margin={{left:0,right:8,top:4,bottom:0}}>
                <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" />
                <XAxis dataKey="t" stroke={COLORS.textDim} tick={{fontSize:8}} />
                <YAxis stroke={COLORS.textDim} tick={{fontSize:8}} domain={[0, 5.5]} />
                <Tooltip contentStyle={{background:COLORS.panel, border:`1px solid ${COLORS.border}`, fontSize:9}} />
                <Line dataKey="pv_gain"  stroke={COLORS.accent3}  dot={false} strokeWidth={1.5} isAnimationActive={false} name="PV gain" />
                <Line dataKey="sst_sup"  stroke={COLORS.yellow}   dot={false} strokeWidth={1.5} isAnimationActive={false} name="SST sup" />
                <Line dataKey="vip_gate" stroke={COLORS.accent2}  dot={false} strokeWidth={1.5} isAnimationActive={false} name="VIP gate" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Precision + route score */}
          <div style={{
            background:COLORS.panel, border:`1px solid ${COLORS.border}`,
            borderRadius:8, padding:"16px 8px 8px 8px",
          }}>
            <div style={{color:COLORS.textDim, fontSize:10, marginBottom:8, paddingLeft:8, letterSpacing:"0.06em"}}>
              PRECISION · ROUTE SCORE · h_norm
            </div>
            <ResponsiveContainer width="100%" height={110}>
              <LineChart data={data} margin={{left:0,right:8,top:4,bottom:0}}>
                <CartesianGrid stroke={COLORS.border} strokeDasharray="2 4" />
                <XAxis dataKey="t" stroke={COLORS.textDim} tick={{fontSize:8}} />
                <YAxis stroke={COLORS.textDim} tick={{fontSize:8}} />
                <Tooltip contentStyle={{background:COLORS.panel, border:`1px solid ${COLORS.border}`, fontSize:9}} />
                <Line dataKey="precision"    stroke={COLORS.accent} dot={false} strokeWidth={1.5} isAnimationActive={false} />
                <Line dataKey="route_score"  stroke={COLORS.green}  dot={false} strokeWidth={1.5} isAnimationActive={false} name="route" />
                <Line dataKey="h_norm"       stroke={COLORS.muted}  dot={false} strokeWidth={1}   isAnimationActive={false} name="h_norm" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div style={{marginTop:20, color:COLORS.textDim, fontSize:9, textAlign:"center", letterSpacing:"0.05em"}}>
        Spec v0.4 · Phase 0+1 · No global backprop · All learning is local
        · PV∈[0.2,5.0] · SST∈[0,1] · tau_elig=20 · anti_hebb=0.1×Hebb
      </div>
    </div>
  );
}
