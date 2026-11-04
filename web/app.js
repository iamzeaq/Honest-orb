/* Honest Orb app: tap-to-talk, answer card, type fallback. */

(function () {
  "use strict";

  var orbButton = document.getElementById("orbButton");
  var orbLine = document.getElementById("orbLine");
  var orbError = document.getElementById("orbError");
  var voiceNote = document.getElementById("voiceNote");

  // Dev mode (?dev=1): show the server's actual failure reason in small
  // text under the orb. Off by default — users only see the friendly line.
  var DEV = /(?:\?|&)dev=1(?:&|$)/.test(location.search);
  var typeToggle = document.getElementById("typeToggle");
  var typeForm = document.getElementById("typeForm");
  var typeInput = document.getElementById("typeInput");
  var card = document.getElementById("card");
  var scrim = document.getElementById("scrim");
  var cardRight = document.getElementById("cardRight");
  var cardSay = document.getElementById("cardSay");
  var cardDo = document.getElementById("cardDo");
  var cardCites = document.getElementById("cardCites");
  var callButton = document.getElementById("callButton");

  var SILENCE_MS = 1500;   // auto-stop after ~1.5s of silence
  var MAX_MS = 20000;      // hard cap on a recording
  var SILENCE_LEVEL = 0.02;

  var mode = "idle";       // idle | recording | working | speaking
  var recorder = null;
  var chunks = [];
  var audioCtx = null;
  var analyser = null;
  var micStream = null;
  var rafId = 0;
  var silenceSince = 0;
  var recordStarted = 0;
  var answerAudio = null;
  var answerCtx = null;
  var answerAnalyser = null;
  var answerRaf = 0;
  var answerQueue = [];      // {url, sentence}[] in speaking order
  var queuePos = 0;          // next item in answerQueue to play
  var queuePlaying = false;  // an Audio element is active
  var streamOpen = false;    // more sentences may still arrive
  var playToken = 0;         // invalidates stale ended handlers
  var playOk = false;        // any clip actually started playing
  var voiceTurn = false;     // this turn started by voice (auto-listen after)
  var pendingAnswer = null;  // staged answer behind "Show details"
  var ackAudio = null;       // short acknowledgement clip masking brain latency
  var detailsButton = document.getElementById("detailsButton");

  // One conversation id per browser, kept across visits: the server's
  // session memory (recent turns, summary, state/topic facts) keys on it.
  var sessionId = null;
  try {
    sessionId = localStorage.getItem("orb_session_v1");
    if (!sessionId) {
      sessionId = (crypto.randomUUID ? crypto.randomUUID() :
        String(Date.now()) + "-" + String(Math.random()).slice(2));
      localStorage.setItem("orb_session_v1", sessionId);
    }
  } catch (e) {
    sessionId = String(Date.now());
  }
  function withSession(url) {
    return url + (url.indexOf("?") >= 0 ? "&" : "?") +
      "session_id=" + encodeURIComponent(sessionId);
  }

  // Play the tiny pre-generated ack clip in the orb voice the moment speech
  // ends. It masks brain latency; the real answer stops it on arrival.
  function playAck(url) {
    stopAck();
    try {
      ackAudio = new Audio(url || "/audio/ack.wav");
      ackAudio.volume = 0.9;
      ackAudio.play().catch(function () { ackAudio = null; });
    } catch (e) { ackAudio = null; }
  }
  function stopAck() {
    if (ackAudio) {
      try { ackAudio.pause(); } catch (e) {}
      ackAudio = null;
    }
  }

  // Inline base64 data URIs play from the answer itself, so they keep
  // working even when /audio/ would land on another container. Plain
  // /audio/ URLs stay as the fallback for older responses.
  function clipUrl(msg) {
    return (msg && (msg.audio_b64 || msg.audio_url)) || "";
  }

  function showVoiceNote() {
    if (!voiceNote) return;
    voiceNote.textContent = "Voice unavailable — text answer above.";
    voiceNote.hidden = false;
  }

  function hideVoiceNote() {
    if (!voiceNote) return;
    voiceNote.hidden = true;
  }

  function orb() { return window.Orb || null; }

  // ---------- thinking sound: Web Audio only, no files ----------
  // Quiet machine hum + soft high ticks while the orb is thinking, from
  // end-of-speech until the first reply audio starts (or any error/idle).
  // The mic is never open during thinking (tracks stop in stopRecording
  // before the thinking state, and startRecording kills the sound first),
  // so the orb cannot hear itself.
  var thinkCtx = null;
  var thinkMaster = null;
  var thinkNodes = null;
  var thinkTimer = 0;
  var thinkNextTick = 0;
  var thinkToken = 0;
  var thinkMuteBtn = document.getElementById("thinkMute");

  function thinkMuted() {
    try { return localStorage.getItem("orb_think_muted") === "1"; }
    catch (e) { return false; }
  }
  function paintThinkMute() {
    if (thinkMuteBtn) {
      thinkMuteBtn.textContent =
        thinkMuted() ? "Thinking hum: off" : "Thinking hum: on";
    }
  }
  function ensureThinkCtx() {
    if (!thinkCtx) {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      thinkCtx = new AC();
      thinkMaster = thinkCtx.createGain();
      thinkMaster.gain.value = 0;
      thinkMaster.connect(thinkCtx.destination);
    }
    if (thinkCtx.state === "suspended") {
      thinkCtx.resume().catch(function () {});
    }
    return thinkCtx;
  }
  function startThinkSound() {
    if (thinkMuted() || thinkNodes) return;
    var ctx = ensureThinkCtx();
    if (!ctx) return;
    var token = ++thinkToken;
    var t = ctx.currentTime;
    thinkMaster.gain.cancelScheduledValues(t);
    thinkMaster.gain.setValueAtTime(Math.max(thinkMaster.gain.value, 0), t);
    thinkMaster.gain.linearRampToValueAtTime(1, t + 0.15); // fade in ~150ms

    // Low hum (~78 Hz sine) with a slow gentle wobble, plus a faint shimmer.
    var hum = ctx.createOscillator();
    hum.type = "sine";
    hum.frequency.value = 78;
    var humGain = ctx.createGain();
    humGain.gain.value = 0.05; // ~15-20% of reply voice loudness
    var wobble = ctx.createOscillator();
    wobble.type = "sine";
    wobble.frequency.value = 0.35;
    var wobbleAmt = ctx.createGain();
    wobbleAmt.gain.value = 6;
    wobble.connect(wobbleAmt);
    wobbleAmt.connect(hum.frequency);
    hum.connect(humGain);
    humGain.connect(thinkMaster);
    hum.start(t);
    wobble.start(t);

    var shimmer = ctx.createOscillator();
    shimmer.type = "sine";
    shimmer.frequency.value = 8400;
    var shimmerGain = ctx.createGain();
    shimmerGain.gain.value = 0.005;
    shimmer.connect(shimmerGain);
    shimmerGain.connect(thinkMaster);
    shimmer.start(t);

    // Soft high ticks (~3.5 kHz, ~2ms) at ~4-5/s with loose timing, nudged
    // by the orb's thinking pulse so it feels alive, not like a clock.
    var clickLen = Math.floor(ctx.sampleRate * 0.012);
    var clickBuf = ctx.createBuffer(1, clickLen, ctx.sampleRate);
    var data = clickBuf.getChannelData(0);
    for (var i = 0; i < clickLen; i++) {
      var env = Math.exp(-i / (ctx.sampleRate * 0.002));
      data[i] = (Math.random() * 2 - 1) * env;
    }
    var band = ctx.createBiquadFilter();
    band.type = "bandpass";
    band.frequency.value = 3500;
    band.Q.value = 1.1;
    var tickGain = ctx.createGain();
    tickGain.gain.value = 0.25; // soft clicks under the hum, never harsh
    band.connect(tickGain);
    tickGain.connect(thinkMaster);

    thinkNodes = { hum: hum, wobble: wobble, shimmer: shimmer, band: band };
    thinkNextTick = t + 0.1;
    thinkTimer = setInterval(function () {
      if (token !== thinkToken || !thinkNodes) return;
      var ahead = ctx.currentTime + 0.3;
      while (thinkNextTick < ahead) {
        // Loose sync with the orb's thinking pulse (~9 rad/s shimmer).
        var pulse = 0.5 + 0.5 * Math.sin(thinkNextTick * 9.0);
        if (Math.random() > 0.25 * pulse) {
          var src = ctx.createBufferSource();
          src.buffer = clickBuf;
          src.connect(band);
          src.start(thinkNextTick);
        }
        thinkNextTick += 0.22 * (0.75 + Math.random() * 0.5);
      }
    }, 90);
  }
  function stopThinkSound(immediate) {
    if (!thinkNodes && !thinkTimer) return;
    var token = ++thinkToken;
    if (thinkTimer) { clearInterval(thinkTimer); thinkTimer = 0; }
    var nodes = thinkNodes;
    thinkNodes = null;
    thinkNextTick = 0;
    if (!thinkCtx || !thinkMaster) return;
    var t = thinkCtx.currentTime;
    thinkMaster.gain.cancelScheduledValues(t);
    if (immediate) {
      thinkMaster.gain.setValueAtTime(0, t);
    } else {
      thinkMaster.gain.setValueAtTime(Math.max(thinkMaster.gain.value, 0), t);
      thinkMaster.gain.linearRampToValueAtTime(0, t + 0.2); // fade out ~200ms
    }
    setTimeout(function () {
      if (token !== thinkToken) return;
      try {
        nodes.hum.stop(); nodes.wobble.stop(); nodes.shimmer.stop();
        nodes.hum.disconnect(); nodes.wobble.disconnect();
        nodes.shimmer.disconnect(); nodes.band.disconnect();
      } catch (e) { /* already torn down */ }
    }, immediate ? 0 : 260);
  }

  function setLine(text) {
    orbError.hidden = true; // a new state clears any stale reason
    orbLine.classList.add("fade");
    setTimeout(function () {
      orbLine.textContent = text;
      orbLine.classList.remove("fade");
    }, 180);
  }

  // Server error bodies carry {detail}; surface the reason for dev mode.
  function httpError(res) {
    return res.json().then(function (body) {
      var reason = (body && body.detail) || ("HTTP " + res.status);
      return new Error(String(reason).slice(0, 200));
    }).catch(function () {
      return new Error("HTTP " + res.status);
    });
  }

  function setState(name) {
    var o = orb();
    if (o && o.setState) o.setState(name);
    orbButton.setAttribute("aria-label",
      name === "idle" ? "Tap and talk" :
      name === "listening" ? "Listening. Tap to stop." :
      name === "thinking" ? "Thinking" : "Speaking");
    orbButton.setAttribute("aria-pressed", name === "listening" ? "true" : "false");
    // Thinking hum follows the orb state: on in thinking, off on idle or
    // re-listen. Speaking keeps it until the first reply audio actually
    // starts (the play handler below stops it then).
    if (name === "thinking") startThinkSound();
    else if (name === "idle" || name === "listening") stopThinkSound();
  }

  function setEmergency(on) {
    var o = orb();
    if (o && o.setEmergency) o.setEmergency(on);
    document.documentElement.classList.toggle("emergency", !!on);
  }

  function firstSentence(text) {
    var m = /^.*?[.!?](\s|$)/.exec(text || "");
    return (m ? m[0] : (text || "")).trim();
  }

  // ---------- recording ----------

  function ensureAudioCtx() {
    if (!audioCtx) {
      var AC = window.AudioContext || window.webkitAudioContext;
      audioCtx = new AC();
    }
    if (audioCtx.state === "suspended") audioCtx.resume();
    return audioCtx;
  }

  function micLevelLoop() {
    var data = new Uint8Array(analyser.fftSize);
    var o = orb();
    var tick = function () {
      if (mode !== "recording") return;
      analyser.getByteTimeDomainData(data);
      var sum = 0;
      for (var i = 0; i < data.length; i++) {
        var v = (data[i] - 128) / 128;
        sum += v * v;
      }
      var level = Math.min(1, Math.sqrt(sum / data.length) * 3);
      if (o && o.setLevel) o.setLevel(level);
      var now = performance.now();
      if (level < SILENCE_LEVEL) {
        if (!silenceSince) silenceSince = now;
        if (now - silenceSince > SILENCE_MS || now - recordStarted > MAX_MS) {
          stopRecording();
          return;
        }
      } else {
        silenceSince = 0;
        if (now - recordStarted > MAX_MS) { stopRecording(); return; }
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }

  async function startRecording() {
    dismissCard();
    stopThinkSound(true); // never let the mic hear the hum's fade tail
    try {
      var ctx = ensureAudioCtx();
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      setLine("Mic blocked — allow access, or type instead.");
      return;
    }
    var mime = "";
    ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].forEach(function (m) {
      if (!mime && window.MediaRecorder && MediaRecorder.isTypeSupported(m)) mime = m;
    });
    try {
      recorder = mime ? new MediaRecorder(micStream, { mimeType: mime })
                      : new MediaRecorder(micStream);
    } catch (e) {
      setLine("Recording isn't supported here — type instead.");
      micStream.getTracks().forEach(function (t) { t.stop(); });
      return;
    }
    chunks = [];
    recorder.ondataavailable = function (e) {
      if (e.data && e.data.size) chunks.push(e.data);
    };
    recorder.onstop = sendRecording;
    var src = ctx.createMediaStreamSource(micStream);
    analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    src.connect(analyser);
    recorder.start(250);
    mode = "recording";
    silenceSince = 0;
    recordStarted = performance.now();
    setState("listening");
    setLine("Listening…");
    micLevelLoop();
  }

  function stopRecording() {
    if (mode !== "recording") return;
    mode = "working";
    cancelAnimationFrame(rafId);
    var o = orb();
    if (o && o.setLevel) o.setLevel(0);
    try { recorder.stop(); } catch (e) { /* already stopped */ }
    if (micStream) micStream.getTracks().forEach(function (t) { t.stop(); });
    setState("thinking");
    setLine("Thinking…");
    playAck(); // instant orb-voice acknowledgement while the brain runs
  }

  async function sendRecording() {
    var blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
    chunks = [];
    if (!blob.size) {
      resetToIdle("I didn't catch that — tap and try again.");
      return;
    }
    voiceTurn = true; // voice turns auto-listen again after speaking
    var ext = (blob.type.indexOf("mp4") >= 0) ? "m4a" : "webm";
    var form = new FormData();
    form.append("audio", blob, "question." + ext);
    try {
      // Prefer the streaming endpoint shape: meta first, then one audio
      // line per sentence, so the first sentence starts playing while
      // later sentences are still rendering server-side. session_id keeps
      // the conversation (state, topic, summary) across turns.
      var res = await fetch(withSession("/api/ask"), {
        method: "POST",
        body: form,
        headers: { "Accept": "application/x-ndjson" }
      });
      if (!res.ok) throw await httpError(res);
      var ctype = res.headers.get("content-type") || "";
      if (res.body && res.body.getReader && ctype.indexOf("ndjson") >= 0) {
        await readAnswerStream(res.body);
        return;
      }
      // No stream support on this response: plain JSON answer instead.
      if (res.body && res.body.cancel) {
        try { await res.body.cancel(); } catch (e) { /* already read */ }
      }
      var res2 = await fetch(withSession("/api/ask?format=json"), {
        method: "POST",
        body: form
      });
      if (!res2.ok) throw await httpError(res2);
      handleAnswer(await res2.json());
    } catch (e) {
      resetToIdle("Something went wrong — tap to try again.", e);
    }
  }

  // ---------- typed fallback ----------

  async function sendText(text) {
    dismissCard();
    voiceTurn = false; // typed turns don't auto-listen afterwards
    mode = "working";
    setState("thinking");
    setLine("Thinking…");
    try {
      // Session chat stream: meta first (answer + caption), then one audio
      // line per sentence. session_id carries the conversation server-side.
      var payload = JSON.stringify({ text: text, session_id: sessionId });
      var res = await fetch("/api/chat_stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload
      });
      if (!res.ok) throw await httpError(res);
      if (res.body && res.body.getReader) {
        await readAnswerStream(res.body);
        return;
      }
      // Ancient browser without stream support: plain JSON answer instead.
      var res2 = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload
      });
      if (!res2.ok) throw await httpError(res2);
      handleAnswer(await res2.json());
    } catch (e) {
      resetToIdle("Something went wrong — try again.", e);
    }
  }

  // NDJSON stream: a meta line, then one audio line per sentence.
  // Sentences play in order as they arrive.
  async function readAnswerStream(body) {
    var reader = body.getReader();
    var decoder = new TextDecoder();
    var buf = "";
    for (;;) {
      var chunk = await reader.read();
      if (chunk.value) buf += decoder.decode(chunk.value, { stream: !chunk.done });
      var lines = buf.split("\n");
      buf = chunk.done ? "" : lines.pop();
      for (var i = 0; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        var msg = null;
        try { msg = JSON.parse(lines[i]); } catch (e) { continue; }
        onStreamMessage(msg);
        if (mode === "idle") return; // failed mid-stream; UI already reset
      }
      if (chunk.done) break;
    }
    streamOpen = false;
    if (!queuePlaying) finishSpeaking();
  }

  // The brain answered but the voice failed: keep the text card and say
  // so in small text instead of dropping back to idle silently.
  function voiceFailed(detail) {
    streamOpen = false;
    if (mode === "speaking" && !card.hidden) {
      showVoiceNote();
      if (!queuePlaying) finishSpeaking();
    } else {
      resetToIdle("Something went wrong — try again.",
        new Error(detail || "tts"));
    }
  }

  // Voice-first: no card pops up automatically. One short caption under
  // the orb, plus "Show details" when there is substance behind it.
  function stageAnswer(ans) {
    pendingAnswer = ans;
    var caption = firstSentence(ans.spoken) || "Here's what I found.";
    setLine(caption);
    var hasDetails = !!(ans.right || ans.say_this || ans.do_this ||
      (ans.citations && ans.citations.length));
    if (detailsButton) detailsButton.hidden = !hasDetails;
  }

  function onStreamMessage(msg) {
    if (msg.type === "meta") {
      var ans = msg.answer || {};
      stopAck();
      setEmergency(ans.status === "emergency");
      mode = "speaking";
      setState("speaking");
      stageAnswer(ans);
      startQueue();
    } else if (msg.type === "audio") {
      var url = clipUrl(msg);
      if (!url) return; // unplayable line; the queue finish shows the note
      answerQueue.push({ url: url, sentence: msg.sentence || "" });
      if (!queuePlaying) playNextInQueue();
    } else if (msg.type === "error") {
      voiceFailed(msg.detail);
    }
  }

  // ---------- answer ----------

  function handleAnswer(data) {
    var ans = data.answer || {};
    var emergency = ans.status === "emergency";
    stopAck();
    setEmergency(emergency);
    mode = "speaking";
    setState("speaking");
    stageAnswer(ans);
    // Prefer the per-sentence queue; fall back to the single full clip.
    // Inline base64 first, /audio/ URL second (see clipUrl).
    var urls = data.audio_urls || [];
    var b64s = data.audio_b64s || [];
    var sents = data.sentences || [];
    var items = [];
    var n = Math.max(urls.length, b64s.length);
    for (var i = 0; i < n; i++) {
      var url = clipUrl({ audio_b64: b64s[i], audio_url: urls[i] });
      if (url) items.push({ url: url, sentence: sents[i] || "" });
    }
    if (!items.length) {
      var last = clipUrl(data);
      if (last) items.push({ url: last, sentence: firstSentence(ans.spoken) });
    }
    if (!items.length) {
      showVoiceNote(); // text answer only, no playable audio at all
      stopThinkSound();
      return;
    }
    startQueue(items);
  }

  function startQueue(items) {
    stopAnswerAudio();
    playToken++;
    playOk = false;
    hideVoiceNote();
    answerQueue = items || [];
    queuePos = 0;
    streamOpen = !items; // stream mode: meta arrived, audio lines pending
    queuePlaying = false;
    playNextInQueue();
  }

  // Play answerQueue[queuePos..] in order. The orb line and the pulse
  // always follow whichever sentence is currently playing.
  function playNextInQueue() {
    if (queuePos >= answerQueue.length) {
      if (!streamOpen) finishSpeaking();
      return; // waiting on more stream lines
    }
    var item = answerQueue[queuePos++];
    var token = playToken;
    queuePlaying = true;
    setLine(item.sentence || "Speaking…");
    answerAudio = new Audio(item.url);
    answerAudio.playsInline = true;
    var o = orb();
    answerAudio.addEventListener("ended", function () {
      if (token !== playToken) return;
      queuePlaying = false;
      stopAnswerLevel();
      playNextInQueue();
    });
    answerAudio.addEventListener("error", function () {
      if (token !== playToken) return;
      queuePlaying = false;
      stopAnswerLevel();
      playNextInQueue(); // skip the broken clip, keep the answer going
    });
    // Pulse the orb with this sentence's amplitude (best effort).
    try {
      if (!answerCtx) {
        var AC = window.AudioContext || window.webkitAudioContext;
        answerCtx = new AC();
      }
      if (answerCtx.state === "suspended") answerCtx.resume();
      var src = answerCtx.createMediaElementSource(answerAudio);
      answerAnalyser = answerCtx.createAnalyser();
      answerAnalyser.fftSize = 512;
      src.connect(answerAnalyser);
      answerAnalyser.connect(answerCtx.destination);
      var data = new Uint8Array(answerAnalyser.fftSize);
      var tick = function () {
        if (!answerAudio || answerAudio.paused) return;
        answerAnalyser.getByteTimeDomainData(data);
        var sum = 0;
        for (var i = 0; i < data.length; i++) {
          var v = (data[i] - 128) / 128;
          sum += v * v;
        }
        if (o && o.setLevel) o.setLevel(Math.min(1, Math.sqrt(sum / data.length) * 3.2));
        answerRaf = requestAnimationFrame(tick);
      };
      answerAudio.addEventListener("play", function onPlay() {
        answerAudio.removeEventListener("play", onPlay);
        if (token === playToken) playOk = true;
        stopThinkSound(); // first reply audio begins: fade the hum out
        answerRaf = requestAnimationFrame(tick);
      });
    } catch (e) { /* plain playback without the pulse */ }
    answerAudio.play().catch(function () {
      // Rejected play (autoplay guard or undecodable clip): skip to the
      // next clip like the error handler instead of stalling silently.
      if (token !== playToken) return;
      queuePlaying = false;
      stopAnswerLevel();
      playNextInQueue();
    });
  }

  function finishSpeaking() {
    stopAnswerLevel();
    var o = orb();
    if (o && o.setLevel) o.setLevel(0);
    // The queue drained but no clip ever played: every clip was broken
    // (or the voice failed mid-stream), so say so under the orb line.
    if (!playOk && queuePos > 0) showVoiceNote();
    // Continuous conversation: a voice turn flows straight back into
    // listening so the user can keep talking. Tapping the orb ends it.
    if (voiceTurn && mode === "speaking") {
      startRecording();
      return;
    }
    mode = "idle";
    setState("idle");
  }

  function stopAnswerLevel() {
    cancelAnimationFrame(answerRaf);
  }

  function stopAnswerAudio() {
    stopAnswerLevel();
    playToken++;
    answerQueue = [];
    queuePos = 0;
    queuePlaying = false;
    streamOpen = false;
    if (answerAudio) {
      try { answerAudio.pause(); } catch (e) {}
      answerAudio = null;
    }
  }

  function showCard(ans) {
    cardRight.textContent = ans.right || "";
    cardSay.textContent = ans.say_this || "—";
    cardDo.textContent = ans.do_this || "—";
    callButton.hidden = ans.status !== "emergency";
    cardCites.innerHTML = "";
    (ans.citations || []).forEach(function (c) {
      var chip = document.createElement("button");
      chip.className = "cite-chip";
      chip.type = "button";
      chip.textContent = c.label;
      chip.setAttribute("aria-expanded", "false");
      chip.addEventListener("click", function () {
        var open = chip.getAttribute("aria-expanded") === "true";
        var next = chip.nextSibling;
        if (!open && !(next && next.classList &&
            next.classList.contains("cite-text"))) {
          fetch("/api/section/" + encodeURIComponent(c.id))
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (s) {
              if (!s) return;
              var box = document.createElement("div");
              box.className = "cite-text";
              var title = document.createElement("strong");
              title.textContent = s.label + " — " + s.title;
              box.appendChild(title);
              box.appendChild(document.createTextNode(" " + s.text));
              chip.after(box);
              chip.setAttribute("aria-expanded", "true");
            })
            .catch(function () {});
        } else if (next && next.classList &&
            next.classList.contains("cite-text")) {
          next.remove();
          chip.setAttribute("aria-expanded", "false");
        }
      });
      cardCites.appendChild(chip);
    });
    card.hidden = false;
    scrim.hidden = false;
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { card.classList.add("open"); });
    });
  }

  function dismissCard() {
    if (card.hidden) return;
    card.classList.remove("open");
    scrim.hidden = true;
    setTimeout(function () { card.hidden = true; }, 480);
  }

  function resetToIdle(line, err) {
    mode = "idle";
    voiceTurn = false;
    pendingAnswer = null;
    stopAck();
    stopAnswerAudio();
    hideVoiceNote();
    if (detailsButton) detailsButton.hidden = true;
    dismissCard();
    setEmergency(false);
    setState("idle");
    var o = orb();
    if (o && o.setLevel) o.setLevel(0);
    if (line) setLine(line);
    else setLine("Tap and talk");
    if (err && DEV) {
      var reason = String((err && err.message) || err).slice(0, 200);
      setTimeout(function () {
        if (mode === "idle") { // still relevant, not superseded
          orbError.textContent = reason;
          orbError.hidden = false;
        }
      }, 220);
    }
  }

  // ---------- wiring ----------

  if (detailsButton) {
    detailsButton.addEventListener("click", function () {
      if (pendingAnswer) {
        detailsButton.hidden = true;
        showCard(pendingAnswer);
      }
    });
  }

  orbButton.addEventListener("click", function () {
    ensureThinkCtx(); // create/resume audio on the tap (iPhone/Safari rule)
    if (mode === "recording") stopRecording();
    else if (mode === "idle") { voiceTurn = true; startRecording(); }
    else { voiceTurn = false; resetToIdle(); } // tap the orb to end
  });

  // Swipe down on the card, or tap outside it, to dismiss.
  var touchY = null;
  card.addEventListener("touchstart", function (e) {
    touchY = e.touches[0].clientY;
  }, { passive: true });
  card.addEventListener("touchend", function (e) {
    if (touchY === null) return;
    var dy = e.changedTouches[0].clientY - touchY;
    touchY = null;
    if (dy > 60) resetToIdle();
  }, { passive: true });
  scrim.addEventListener("click", function () { resetToIdle(); });

  typeToggle.addEventListener("click", function () {
    ensureThinkCtx(); // create/resume audio on the tap (iPhone/Safari rule)
    var opening = typeForm.hidden;
    typeForm.hidden = !opening;
    if (opening) typeInput.focus();
  });
  typeForm.addEventListener("submit", function (e) {
    e.preventDefault();
    ensureThinkCtx();
    var text = typeInput.value.trim();
    if (!text || mode === "working" || mode === "recording") return;
    typeInput.value = "";
    typeForm.hidden = true;
    sendText(text);
  });
  if (thinkMuteBtn) {
    paintThinkMute();
    thinkMuteBtn.addEventListener("click", function () {
      try {
        localStorage.setItem("orb_think_muted", thinkMuted() ? "0" : "1");
      } catch (e) {}
      paintThinkMute();
      if (thinkMuted()) stopThinkSound(true);
      else if (mode === "working") startThinkSound();
    });
  }

  if ("serviceWorker" in navigator) {
    window.addEventListener("load", function () {
      navigator.serviceWorker.register("/sw.js").catch(function () {});
    });
  }
})();
