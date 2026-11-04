/* Honest Orb — WebGL vibranium-energy orb. No libraries.
 *
 * Charged purple energy alive inside a dark crystal sphere: deep royal
 * purple depths, electric violet / hot magenta internal currents, glowing
 * filaments drifting through the inside, near-white lavender highlights,
 * violet bloom bleeding into the black, and a thin warm gold rim light on
 * one edge. Same round shape, same motion and states as before.
 * Falls back to a CSS orb (see .no-webgl) when WebGL is unavailable.
 *
 * window.Orb: { setState(name), setEmergency(bool), setLevel(v), ready }
 * States: idle | listening | thinking | speaking  (~400ms eased blends)
 */

(function () {
  "use strict";

  var canvas = document.getElementById("orbCanvas");
  var reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var VERT =
    "attribute vec2 a_pos;\n" +
    "void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }";

  // 2D simplex noise (Ashima), fbm kept to 3 octaves for mid-range phones.
  var FRAG =
    "precision mediump float;\n" +
    "uniform vec2 u_res;\n" +
    "uniform float u_time;\n" +
    "uniform float u_level;\n" +   // 0..1 live amplitude (mic or answer audio)
    "uniform float u_flow;\n" +    // internal flow speed
    "uniform float u_swirl;\n" +   // inward rotation
    "uniform float u_glow;\n" +    // outer glow size
    "uniform float u_bright;\n" +  // overall brightness
    "uniform float u_red;\n" +     // 0 violet .. 1 emergency red
    "uniform float u_fil;\n" +     // filament intensity (0 on low-end)
    "uniform float u_gold;\n" +    // gold rim-light amount
    "vec3 mod289(vec3 x){ return x - floor(x * (1.0/289.0)) * 289.0; }\n" +
    "vec2 mod289(vec2 x){ return x - floor(x * (1.0/289.0)) * 289.0; }\n" +
    "vec3 permute(vec3 x){ return mod289(((x*34.0)+1.0)*x); }\n" +
    "float snoise(vec2 v){\n" +
    "  const vec4 C = vec4(0.211324865405187, 0.366025403784439,\n" +
    "    -0.577350269189626, 0.024390243902439);\n" +
    "  vec2 i = floor(v + dot(v, C.yy));\n" +
    "  vec2 x0 = v - i + dot(i, C.xx);\n" +
    "  vec2 i1 = (x0.x > x0.y) ? vec2(1.0,0.0) : vec2(0.0,1.0);\n" +
    "  vec4 x12 = x0.xyxy + C.xxzz;\n" +
    "  x12.xy -= i1;\n" +
    "  i = mod289(i);\n" +
    "  vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0))\n" +
    "    + i.x + vec3(0.0, i1.x, 1.0));\n" +
    "  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),\n" +
    "    dot(x12.zw,x12.zw)), 0.0);\n" +
    "  m = m*m; m = m*m;\n" +
    "  vec3 x = 2.0 * fract(p * C.www) - 1.0;\n" +
    "  vec3 h = abs(x) - 0.5;\n" +
    "  vec3 ox = floor(x + 0.5);\n" +
    "  vec3 a0 = x - ox;\n" +
    "  m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);\n" +
    "  vec3 g;\n" +
    "  g.x = a0.x * x0.x + h.x * x0.y;\n" +
    "  g.yz = a0.yz * x12.xz + h.yz * x12.yw;\n" +
    "  return 130.0 * dot(m, g);\n" +
    "}\n" +
    "float fbm(vec2 p){\n" +
    "  float v = 0.0; float a = 0.5;\n" +
    "  for(int i=0;i<3;i++){ v += a * snoise(p); p *= 2.03; a *= 0.5; }\n" +
    "  return v;\n" +
    "}\n" +
    "void main(){\n" +
    "  vec2 uv = (gl_FragCoord.xy / u_res) * 2.0 - 1.0;\n" +
    "  uv.x *= u_res.x / u_res.y;\n" +
    "  float r = length(uv);\n" +
    "  float ang = u_swirl * u_time * 0.6;\n" +
    "  float ca = cos(ang), sa = sin(ang);\n" +
    "  vec2 p = mat2(ca,-sa,sa,ca) * uv;\n" +
    "  vec2 flow = vec2(u_time * u_flow, -u_time * u_flow * 0.7);\n" +
    "  float warp = fbm(p * 1.6 + flow * 0.6);\n" +
    "  float n = fbm(p * 1.9 + flow + warp * 0.9);\n" +
    // live ripple: rings that brighten with mic/answer amplitude
    "  float ripple = sin(r * 18.0 - u_time * 7.0 + n * 4.0) * 0.5 + 0.5;\n" +
    "  float body = smoothstep(1.0, 0.15, r + n * 0.12);\n" +
    "  float core = smoothstep(0.62, 0.0, r + n * 0.18);\n" +
    // vibranium palette: royal depths, violet/magenta currents,
    // near-white lavender highlights, warm gold rim catch-light
    "  vec3 deep = vec3(0.231, 0.039, 0.561);\n" +   // #3B0A8F
    "  vec3 violet = vec3(0.482, 0.184, 0.968);\n" + // #7B2FF7
    "  vec3 magenta = vec3(0.757, 0.231, 1.0);\n" +  // #C13BFF
    "  vec3 lavHi = vec3(0.914, 0.847, 1.0);\n" +   // #E9D8FF
    "  vec3 goldRim = vec3(0.910, 0.690, 0.294);\n" +// #E8B04B
    "  vec3 redCore = vec3(1.0, 0.42, 0.34);\n" +
    "  vec3 redEdge = vec3(0.45, 0.05, 0.04);\n" +
    "  vec3 hi = mix(lavHi, redCore, u_red);\n" +
    "  vec3 mid = mix(mix(deep, violet, 0.55) + magenta * 0.22,\n" +
    "    redEdge * 2.2, u_red);\n" +
    "  vec3 edge = mix(deep * 0.55, redEdge, u_red);\n" +
    "  vec3 col = mix(edge, mid, body);\n" +
    "  col = mix(col, hi, core * core);\n" +
    // liquid streaks inside the sphere
    "  col += mid * (n * 0.5 + 0.5) * body * 0.55;\n" +
    "  col += hi * ripple * u_level * body * 0.7;\n" +
    // fine glowing filaments / veins drifting through the inside
    "  if (u_fil > 0.003) {\n" +
    "    vec2 fp = p * 2.6 - flow * 1.35 + warp * 0.7;\n" +
    "    float v1 = 1.0 - abs(fbm(fp));\n" +
    "    float v2 = 1.0 - abs(fbm(fp * 1.7 + 4.7));\n" +
    "    float fil = pow(v1 * v2, 3.0);\n" +
    "    col += mix(lavHi, magenta, 0.45) * fil * body * u_fil;\n" +
    "  }\n" +
    // thin warm gold rim light on one edge (suppressed in emergency)
    "  vec2 dir = r > 0.001 ? uv / r : vec2(1.0, 0.0);\n" +
    "  float rimBand = smoothstep(0.30, 0.62, r) * (1.0 - smoothstep(0.62, 0.86, r));\n" +
    "  float rimSide = pow(max(dot(dir, normalize(vec2(0.75, -0.45))), 0.0), 3.0);\n" +
    "  col += goldRim * rimBand * rimSide * u_gold * (1.0 - u_red);\n" +
    "  col *= (0.92 + 0.35 * u_level);\n" +
    // strong soft violet bloom + faint wider halo bleeding into the black
    "  float g = exp(-max(r - 0.52, 0.0) * (5.5 - u_glow * 3.0));\n" +
    "  col += mix(violet, redEdge * 2.6, u_red) * g * (0.34 + 0.62 * u_glow);\n" +
    "  float halo = exp(-max(r - 0.5, 0.0) * 2.1);\n" +
    "  col += mix(violet * 0.55, redEdge, u_red) * halo * 0.16;\n" +
    "  col *= u_bright;\n" +
    "  vec3 bg = vec3(0.039, 0.039, 0.039);\n" +
    "  float alpha = smoothstep(1.9, 0.4, r);\n" +
    "  gl_FragColor = vec4(mix(bg, col, alpha), 1.0);\n" +
    "}";

  // Per-state shader + scale targets. Blended with ~400ms easing in JS.
  // fil = filament intensity, gold = rim-light amount (motion untouched).
  var STATES = {
    idle:      { flow: 0.16, swirl: 0.10, glow: 0.45, bright: 1.00, scale: 1.00, fil: 0.50, gold: 0.35 },
    listening: { flow: 0.55, swirl: 0.35, glow: 0.62, bright: 1.12, scale: 1.05, fil: 0.55, gold: 0.35 },
    thinking:  { flow: 0.32, swirl: 1.60, glow: 0.34, bright: 0.88, scale: 0.99, fil: 0.95, gold: 0.60 },
    speaking:  { flow: 0.42, swirl: 0.30, glow: 0.85, bright: 1.18, scale: 1.04, fil: 0.70, gold: 0.45 }
  };

  var api = {
    ready: false,
    state: "idle",
    setState: function () {},
    setEmergency: function () {},
    setLevel: function () {}
  };

  if (!canvas) { window.Orb = api; return; }

  var gl = null;
  try {
    gl = canvas.getContext("webgl", {
      antialias: false, alpha: false, depth: false, stencil: false
    });
  } catch (e) { gl = null; }

  if (!gl) {
    document.documentElement.classList.add("no-webgl");
    api.ready = true;
    window.Orb = api;
    return;
  }

  function compile(type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      throw new Error("shader: " + gl.getShaderInfoLog(s));
    }
    return s;
  }

  var prog;
  try {
    prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error("link");
  } catch (e) {
    document.documentElement.classList.add("no-webgl");
    api.ready = true;
    window.Orb = api;
    return;
  }
  gl.useProgram(prog);

  var buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER,
    new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  var loc = gl.getAttribLocation(prog, "a_pos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  var U = {};
  ["u_res", "u_time", "u_level", "u_flow", "u_swirl",
   "u_glow", "u_bright", "u_red", "u_fil", "u_gold"].forEach(function (n) {
    U[n] = gl.getUniformLocation(prog, n);
  });

  // Low-end phones: skip the filament pass, render at lower resolution.
  var lowEnd = reduceMotion ||
    ((navigator.hardwareConcurrency || 8) <= 4 &&
     Math.min(window.screen.width || 9999, window.screen.height || 9999) < 760);

  var cur = { flow: 0.16, swirl: 0.10, glow: 0.45, bright: 1.0,
              fil: 0.5, gold: 0.35 };
  var tgt = STATES.idle;
  var scaleCur = 1, scaleTgt = 1;
  var level = 0, levelTgt = 0;
  var redCur = 0, redTgt = 0;
  var timeScale = reduceMotion ? 0.12 : 1;

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, lowEnd ? 1.25 : 2);
    var size = Math.floor(canvas.clientWidth * dpr) || 512;
    if (canvas.width !== size || canvas.height !== size) {
      canvas.width = size;
      canvas.height = size;
      gl.viewport(0, 0, size, size);
    }
  }
  window.addEventListener("resize", resize);

  var t0 = performance.now();
  var last = t0;
  var breathPhase = 0;
  var running = true;
  document.addEventListener("visibilitychange", function () {
    running = !document.hidden;
    if (running) { last = performance.now(); requestAnimationFrame(frame); }
  });

  function frame(now) {
    if (!running) return;
    requestAnimationFrame(frame);
    var dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    resize();

    // ~400ms critically-damped-ish easing toward the state target.
    var k = 1 - Math.exp(-dt / 0.16);
    cur.flow += (tgt.flow - cur.flow) * k;
    cur.swirl += (tgt.swirl - cur.swirl) * k;
    cur.glow += (tgt.glow - cur.glow) * k;
    cur.bright += (tgt.bright - cur.bright) * k;
    cur.fil += (tgt.fil - cur.fil) * k;
    cur.gold += (tgt.gold - cur.gold) * k;
    scaleCur += (scaleTgt - scaleCur) * k;
    level += (levelTgt - level) * (1 - Math.exp(-dt / 0.09));
    redCur += (redTgt - redCur) * k;

    var t = (now - t0) / 1000 * timeScale;
    breathPhase += dt * (Math.PI * 2 / 4); // ±3% breathing over ~4s
    var breathe = 1 + 0.03 * Math.sin(breathPhase);
    canvas.style.transform = "scale(" + (scaleCur * breathe).toFixed(4) + ")";

    var shimmer = (api.state === "thinking")
      ? 1 + 0.06 * Math.sin(t * 9.0) : 1;

    gl.uniform2f(U.u_res, canvas.width, canvas.height);
    gl.uniform1f(U.u_time, t);
    gl.uniform1f(U.u_level, level);
    gl.uniform1f(U.u_flow, cur.flow);
    gl.uniform1f(U.u_swirl, cur.swirl);
    gl.uniform1f(U.u_glow, cur.glow);
    gl.uniform1f(U.u_bright, cur.bright * shimmer);
    gl.uniform1f(U.u_red, redCur);
    gl.uniform1f(U.u_fil, lowEnd ? 0 : cur.fil);
    gl.uniform1f(U.u_gold, cur.gold);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  api.ready = true;
  api.setState = function (name) {
    if (STATES[name]) { api.state = name; tgt = STATES[name]; scaleTgt = tgt.scale; }
  };
  api.setEmergency = function (on) { redTgt = on ? 1 : 0; };
  api.setLevel = function (v) {
    levelTgt = Math.max(0, Math.min(1, v || 0));
  };

  resize();
  requestAnimationFrame(frame);
  window.Orb = api;
})();
