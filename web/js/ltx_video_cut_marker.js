import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

// LTX Video Cut Marker (Scenes).
//
// Cuts mark SCENE BOUNDARIES (the start of the next scene/chunk), snapped to
// the LTX latent grid. The widget emits the sampler's `scene_lengths`
// schedule: pipe-separated pixel-frame counts per scene (multiples of 8),
// including the final scene to the video's end. Cuts are time-anchored and
// computed in emit_fps frame space (default 25 — the VHS force_rate 25 case),
// so the schedule lives in the pipeline's frame space, not the file's.
//
// All placement is hard-snapped: scene lengths must be multiples of 8, so
// off-grid boundaries don't exist in this mode. Alt only affects playhead
// stepping (fine, per-frame).

const GRID_COLOR    = "#3a3a3a";
const TICK_COLOR    = "#555";
const CUT_COLOR     = "#4caf50";
const CUT_SELECTED  = "#a5d6a7";
const END_COLOR     = "#ef5350";
const END_SELECTED  = "#ff8a80";
const DEAD_REGION   = "rgba(0,0,0,0.55)";
const SCENE_LABEL   = "#777";
const PLAYHEAD      = "#e0e0e0";
const BAR_BG        = "#1c1c1c";
const WAVEFORM      = "#2e4b40";

const END_SEL = -2;   // `selected` sentinel for the end marker

function frameToLatent(f) {
    return f <= 0 ? 0 : Math.floor((f - 1) / 8) + 1;
}

app.registerExtension({
    name: "LTXAVTools.VideoCutMarker",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LTXVideoCutMarker") return;

        // Fires after serialized widget values are applied on workflow
        // load/refresh — reload the media with the RESTORED filename so the
        // saved schedule is restored against the right file.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            this._cutMarkerReload?.();
            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            const videoWidget  = node.widgets.find((w) => w.name === "video");
            const scenesWidget = node.widgets.find((w) => w.name === "scene_lengths");
            const fpsWidget    = node.widgets.find((w) => w.name === "emit_fps");

            // ---- state -----------------------------------------------------
            // cuts: [{ t: seconds }] — each marks the start of the next scene.
            let cuts = [];
            let selected = -1;      // cut index, or END_SEL for the end marker
            let endLat = null;      // usable latent count when an end marker is set
            let detectedFps = null;
            let scheduleMismatch = false;   // stored schedule longer than loaded media
            let loadedName = null;          // media the widget currently shows
            let pendingReset = false;       // user picked NEW media -> clear cuts on load

            const fps = () => (fpsWidget?.value > 0 ? fpsWidget.value : 25.0);
            const duration = () => (isFinite(video.duration) ? video.duration : 0);
            const totalFrames = () => Math.max(1, Math.round(duration() * fps()));
            const lastFrame = () => totalFrames() - 1;
            // usable latents (incomplete tail latent dropped)
            const totalLatents = () => Math.max(1, Math.floor((totalFrames() - 1) / 8) + 1);
            const timeToFrame = (t) => Math.max(0, Math.min(lastFrame(), Math.round(t * fps() - 0.5)));
            const frameToTime = (f) => (f + 0.5) / fps();

            // Effective schedule length: capped by the end marker when present.
            const effLatents = () =>
                endLat !== null ? Math.min(endLat, totalLatents()) : totalLatents();
            // End marker sits on the LAST pixel of latent E: frame 8(E-1).
            const endFrame = () => 8 * (effLatents() - 1);
            const endTime  = () => frameToTime(endFrame());
            const lastBoundaryLat = () => {
                let m = 0;
                for (const c of cuts) m = Math.max(m, boundaryLatent(snapBoundary(timeToFrame(c.t))));
                return m;
            };

            // Boundary grid: first pixel of latent k (k>=1) is 1+8(k-1). A
            // boundary at latent L needs 1 <= L <= effLatents-1 so the final
            // scene keeps at least one latent.
            function snapBoundary(f) {
                let k = Math.round((f - 1) / 8);            // latent index - 1
                k = Math.max(0, Math.min(effLatents() - 2, k));
                return 1 + 8 * k;
            }
            const boundaryLatent = (p) => Math.floor((p - 1) / 8) + 1;

            // ---- DOM -------------------------------------------------------
            const root = document.createElement("div");
            root.style.cssText =
                "display:flex;flex-direction:column;gap:4px;width:100%;" +
                "font-family:monospace;font-size:11px;color:#ccc;outline:none;";
            root.tabIndex = 0;

            const video = document.createElement("video");
            video.style.cssText = "width:100%;max-height:260px;background:#000;";
            video.muted = false;
            video.preload = "auto";
            root.appendChild(video);

            const bar = document.createElement("div");
            bar.style.cssText = "display:flex;gap:4px;align-items:center;flex-wrap:wrap;";
            const mkBtn = (label, title, fn) => {
                const b = document.createElement("button");
                b.textContent = label;
                b.title = title;
                b.style.cssText =
                    "background:#333;color:#ddd;border:1px solid #555;border-radius:3px;" +
                    "padding:2px 8px;cursor:pointer;font-family:monospace;font-size:11px;";
                b.addEventListener("click", (e) => { e.preventDefault(); fn(); root.focus(); });
                bar.appendChild(b);
                return b;
            };
            const playBtn = mkBtn("▶", "Play/pause (Space)", () => togglePlay());
            mkBtn("⏮", "Seek to start", () => seekFrame(0));
            mkBtn("−8", "Back one latent (←)", () => stepPlayhead(-8));
            mkBtn("−1", "Back one frame (Alt+←)", () => stepPlayhead(-1));
            mkBtn("+1", "Forward one frame (Alt+→)", () => stepPlayhead(1));
            mkBtn("+8", "Forward one latent (→)", () => stepPlayhead(8));
            mkBtn("✂ cut", "New scene starts at playhead (C)", () => addCutAtPlayhead());
            mkBtn("⏹ end", "Video ends at playhead (E) — schedule stops here; "
                          + "frame_count/frame_load_cap cover only up to this marker",
                  () => setEndAtPlayhead());
            mkBtn("✕ del", "Delete selected marker (X / right-click)", () => deleteSelected());
            const autoInput = document.createElement("input");
            autoInput.type = "number";
            autoInput.min = "0.5";
            autoInput.step = "0.5";
            autoInput.value = "12";
            autoInput.title = "Auto-cut interval in seconds";
            autoInput.style.cssText =
                "width:44px;background:#222;color:#ddd;border:1px solid #555;" +
                "border-radius:3px;font-family:monospace;font-size:11px;padding:2px 4px;";
            bar.appendChild(autoInput);
            mkBtn("⚡ auto", "Replace ALL cuts with one every N seconds "
                          + "(snapped to the latent grid, stops at the end marker)",
                  () => autoPlaceCuts());
            mkBtn("⧉ copy", "Copy scene_lengths string", () => {
                navigator.clipboard?.writeText(scenesWidget.value || "");
            });
            const upBtn = mkBtn("⇧ upload", "Upload a video to the input directory", () => fileInput.click());
            upBtn.style.marginLeft = "auto";
            root.appendChild(bar);

            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.accept = "video/*,audio/*";
            fileInput.style.display = "none";
            root.appendChild(fileInput);

            const canvas = document.createElement("canvas");
            canvas.height = 72;
            canvas.style.cssText = "width:100%;height:72px;cursor:crosshair;";
            root.appendChild(canvas);

            const readout = document.createElement("div");
            readout.style.cssText = "white-space:pre;color:#aaa;line-height:1.5;";
            root.appendChild(readout);

            node.addDOMWidget("cut_timeline", "div", root, { serialize: false });
            node.size = [Math.max(node.size[0], 420), Math.max(node.size[1], 480)];

            // ---- video source ----------------------------------------------
            let audioCtx = null;
            let peaks = null;   // { mins, maxs, n } — waveform min/max buckets

            async function loadWaveform(url) {
                peaks = null;
                try {
                    const resp = await fetch(url);
                    const buf = await resp.arrayBuffer();
                    audioCtx = audioCtx ||
                        new (window.AudioContext || window.webkitAudioContext)();
                    const decoded = await audioCtx.decodeAudioData(buf);
                    const ch = decoded.getChannelData(0);
                    const N = 2048;
                    const mins = new Float32Array(N);
                    const maxs = new Float32Array(N);
                    const per = Math.max(1, Math.floor(ch.length / N));
                    for (let i = 0; i < N; i++) {
                        let mn = 1, mx = -1;
                        const s0 = i * per, s1 = Math.min(ch.length, s0 + per);
                        for (let s = s0; s < s1; s += 16) {
                            const v = ch[s];
                            if (v < mn) mn = v;
                            if (v > mx) mx = v;
                        }
                        mins[i] = mn; maxs[i] = mx;
                    }
                    peaks = { mins, maxs, n: N };
                    console.debug("[CutMarker] waveform decoded:",
                                  decoded.duration.toFixed(2), "s");
                } catch (e) {
                    // No decodable audio track (or unsupported container) —
                    // timeline simply renders without a waveform.
                    console.debug("[CutMarker] no waveform:", e?.message);
                    peaks = null;
                }
                lastSig = "";   // force a repaint either way
            }

            function loadVideo() {
                const name = videoWidget?.value;
                if (!name) return;
                const parts = name.split("/");
                const file = parts.pop();
                const subfolder = parts.join("/");
                const url = api.apiURL(
                    `/view?filename=${encodeURIComponent(file)}` +
                    `&type=input&subfolder=${encodeURIComponent(subfolder)}`
                );
                video.src = url;
                detectedFps = null;
                loadedName = name;
                video.load();
                loadWaveform(url);
            }

            const prevVideoCb = videoWidget?.callback;
            if (videoWidget) {
                videoWidget.callback = function (...args) {
                    const rr = prevVideoCb?.apply(this, args);
                    // A combo change to a DIFFERENT file is a user decision to
                    // work on new media: the old schedule is meaningless there,
                    // so clear it once metadata arrives. (Workflow restore goes
                    // through _cutMarkerReload, never through this callback, so
                    // saved schedules still survive refresh.)
                    if (videoWidget.value !== loadedName) pendingReset = true;
                    loadVideo();
                    return rr;
                };
            }

            async function uploadFile(f) {
                if (!f) return;
                const form = new FormData();
                form.append("image", f);           // /upload/image accepts any file
                form.append("type", "input");
                const resp = await api.fetchApi("/upload/image", { method: "POST", body: form });
                if (resp.status === 200) {
                    const data = await resp.json();
                    const name = (data.subfolder ? data.subfolder + "/" : "") + data.name;
                    if (videoWidget) {
                        if (!videoWidget.options.values.includes(name)) {
                            videoWidget.options.values.push(name);
                        }
                        videoWidget.value = name;
                    }
                    pendingReset = true;   // fresh upload = fresh schedule
                    loadVideo();
                } else {
                    console.error("[LTXVideoCutMarker] upload failed", resp.status);
                }
            }

            fileInput.addEventListener("change", async () => {
                await uploadFile(fileInput.files?.[0]);
                fileInput.value = "";
            });

            ["dragenter", "dragover"].forEach((ev) =>
                root.addEventListener(ev, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    root.style.outline = "2px dashed #4caf50";
                })
            );
            ["dragleave", "drop"].forEach((ev) =>
                root.addEventListener(ev, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    root.style.outline = "none";
                })
            );
            root.addEventListener("drop", async (e) => {
                const f = [...(e.dataTransfer?.files || [])].find(
                    (x) => x.type.startsWith("video/") || x.type.startsWith("audio/") ||
                        /\.(mp4|webm|mkv|mov|avi|m4v|mp3|wav|flac|ogg|m4a|aac|opus)$/i.test(x.name)
                );
                if (f) await uploadFile(f);
            });

            // detected source fps (info only) via frame-callback deltas
            function sampleFps() {
                if (!video.requestVideoFrameCallback) return;
                const samples = [];
                let lastMediaTime = -1;
                const cb = (_now, meta) => {
                    if (lastMediaTime >= 0) {
                        const dt = meta.mediaTime - lastMediaTime;
                        if (dt > 0.001) samples.push(dt);
                    }
                    lastMediaTime = meta.mediaTime;
                    if (samples.length < 12 && !video.paused) {
                        video.requestVideoFrameCallback(cb);
                    } else if (samples.length >= 4) {
                        samples.sort((a, b) => a - b);
                        detectedFps = 1 / samples[Math.floor(samples.length / 2)];
                    }
                };
                video.requestVideoFrameCallback(cb);
            }

            // ---- cuts <-> scene_lengths string -------------------------------
            function boundaryLatents() {
                const ls = cuts.map((c) => boundaryLatent(snapBoundary(timeToFrame(c.t))));
                return [...new Set(ls)].sort((a, b) => a - b)
                    .filter((L) => L >= 1 && L <= effLatents() - 1);
            }
            function syncWidget() {
                if (!duration()) return;
                scheduleMismatch = false;   // an explicit (re-)emit supersedes the warning
                const Ls = boundaryLatents();
                const lengths = [];
                let prev = 0;
                for (const L of Ls) { lengths.push((L - prev) * 8); prev = L; }
                lengths.push((effLatents() - prev) * 8);   // final scene to end marker/video end
                scenesWidget.value = lengths.join("|");
                console.debug("[CutMarker] sync:", JSON.stringify(scenesWidget.value),
                              "endLat =", endLat,
                              "cuts[] =", cuts.map((c) => c.t.toFixed(3)).join(" "));
            }
            function restoreFromWidget() {
                console.debug("[CutMarker] RESTORE from:", JSON.stringify(scenesWidget.value),
                              "duration =", duration());
                if (!duration()) return;
                const toks = (scenesWidget.value || "").replace(/,/g, "|").split("|")
                    .map((s) => s.trim()).filter(Boolean);
                const latents = toks.map((s) => Math.round((parseInt(s, 10) || 0) / 8))
                    .filter((n) => n > 0);
                // A schedule summing short of the video's length implies an end marker.
                const total = latents.reduce((a, b) => a + b, 0);
                scheduleMismatch = total > totalLatents();
                endLat = (total >= 1 && total < totalLatents()) ? total : null;
                cuts = [];
                let cum = 0;
                // all entries except the last define a boundary after them
                for (let i = 0; i < latents.length - 1; i++) {
                    cum += latents[i];
                    if (cum >= 1 && cum <= effLatents() - 1) {
                        cuts.push({ t: frameToTime(1 + 8 * (cum - 1)) });
                    }
                }
                cuts.sort((a, b) => a.t - b.t);
                selected = -1;
                // Re-emit ONLY when the stored schedule fits the loaded media.
                // A schedule LONGER than the media means the wrong (or changed)
                // file is loaded — rewriting the widget then would destroy the
                // user's saved schedule. Keep the string untouched and warn.
                if (!scheduleMismatch) {
                    syncWidget();
                } else {
                    console.warn("[CutMarker] saved schedule (" + total + " latents) "
                        + "exceeds loaded media (" + totalLatents() + ") — string "
                        + "preserved, not re-emitted. Wrong video loaded?");
                }
                draw();
            }

            // ---- editing ----------------------------------------------------
            function addCutAtPlayhead() {
                if (!duration()) return;
                const p = snapBoundary(timeToFrame(video.currentTime));
                const L = boundaryLatent(p);
                console.debug("[CutMarker] ADD boundary px", p, "latent", L,
                              "| before:", cuts.length, "cuts");
                const existing = cuts.findIndex(
                    (c) => boundaryLatent(snapBoundary(timeToFrame(c.t))) === L
                );
                if (existing >= 0) { selected = existing; draw(); return; }
                cuts.push({ t: frameToTime(p) });
                cuts.sort((a, b) => a.t - b.t);
                selected = cuts.findIndex((c) => timeToFrame(c.t) === p);
                syncWidget();
                draw();
            }
            function setEndAtPlayhead() {
                if (!duration()) return;
                const f = timeToFrame(video.currentTime);
                let E = Math.round(f / 8) + 1;   // end sits on last px of latent E: 8(E-1)
                E = Math.max(lastBoundaryLat() + 1, Math.min(totalLatents(), E));
                endLat = E;
                selected = END_SEL;
                seekFrame(endFrame());
                syncWidget();
                draw();
            }
            function deleteSelected() {
                if (selected === END_SEL) {
                    endLat = null;
                    selected = -1;
                    syncWidget();
                    draw();
                    return;
                }
                if (selected < 0) return;
                cuts.splice(selected, 1);
                selected = -1;
                syncWidget();
                draw();
            }
            function autoPlaceCuts() {
                if (!duration()) return;
                const iv = parseFloat(autoInput.value);
                if (!(iv > 0)) return;
                const newCuts = [];
                let prevL = 0;
                for (let t = iv; t < endTime(); t += iv) {
                    const p = snapBoundary(timeToFrame(t));
                    const L = boundaryLatent(p);
                    if (L > prevL && L <= effLatents() - 1) {
                        newCuts.push({ t: frameToTime(p) });
                        prevL = L;
                    }
                }
                cuts = newCuts;
                selected = -1;
                console.debug("[CutMarker] auto-placed", cuts.length,
                              "cuts @", iv, "s");
                syncWidget();
                draw();
            }
            function nudgeSelected(dir) {
                if (selected === END_SEL) {
                    if (endLat === null) return;
                    endLat = Math.max(lastBoundaryLat() + 1,
                                      Math.min(totalLatents(), endLat + dir));
                    if (endLat >= totalLatents()) endLat = totalLatents();
                    seekFrame(endFrame());
                    syncWidget();
                    draw();
                    return;
                }
                if (selected < 0) return;
                let p = snapBoundary(timeToFrame(cuts[selected].t));
                p = snapBoundary(p + dir * 8);
                cuts[selected].t = frameToTime(p);
                seekFrame(p);
                syncWidget();
                draw();
            }
            function stepPlayhead(deltaFrames) {
                seekFrame(Math.max(0, Math.min(lastFrame(), timeToFrame(video.currentTime) + deltaFrames)));
            }
            function seekFrame(f) {
                video.currentTime = frameToTime(f);
            }
            function togglePlay() {
                if (video.paused) { video.play(); sampleFps(); }
                else video.pause();
            }

            // ---- timeline drawing -------------------------------------------
            function xForTime(t) {
                return (t / Math.max(duration(), 0.001)) * canvas.width;
            }
            function timeForX(x) {
                return (x / canvas.width) * duration();
            }
            function draw() {
                const ctx = canvas.getContext("2d");
                // Only reassign canvas.width on an actual size change — the
                // assignment itself forces a full canvas reset.
                const targetW = Math.max(1, Math.round(
                    canvas.clientWidth * (window.devicePixelRatio || 1)));
                if (canvas.width !== targetW) canvas.width = targetW;
                const W = canvas.width;
                const H = canvas.height;
                ctx.fillStyle = BAR_BG;
                ctx.fillRect(0, 0, W, H);
                if (!duration()) return;

                // waveform backdrop (audio files, or a video's audio track)
                if (peaks) {
                    ctx.fillStyle = WAVEFORM;
                    const mid = H * 0.5;
                    const amp = H * 0.46;
                    for (let x = 0; x < W; x++) {
                        const b = Math.min(peaks.n - 1, Math.floor((x / W) * peaks.n));
                        const y1 = mid - peaks.maxs[b] * amp;
                        const y2 = mid - peaks.mins[b] * amp;
                        ctx.fillRect(x, y1, 1, Math.max(1, y2 - y1));
                    }
                }

                // latent grid ticks
                const total = totalFrames();
                const pxPerFrame = W / total;
                const step = pxPerFrame * 8 < 4 ? 8 * Math.ceil(4 / (pxPerFrame * 8)) : 8;
                ctx.strokeStyle = GRID_COLOR;
                ctx.beginPath();
                for (let f = 1; f <= total; f += step) {
                    const x = xForTime(frameToTime(f) - 0.5 / fps());
                    ctx.moveTo(x, H * 0.45); ctx.lineTo(x, H);
                }
                ctx.stroke();

                // second ticks
                ctx.strokeStyle = TICK_COLOR;
                ctx.beginPath();
                for (let s = 0; s <= duration(); s += 1) {
                    const x = xForTime(s);
                    ctx.moveTo(x, 0); ctx.lineTo(x, H * 0.35);
                }
                ctx.stroke();

                // scene index labels at region midpoints (up to end marker)
                const sceneEnd = endLat !== null ? endTime() : duration();
                const bounds = [0, ...cuts.map((c) => c.t), sceneEnd];
                ctx.fillStyle = SCENE_LABEL;
                ctx.font = `${10 * (window.devicePixelRatio || 1)}px monospace`;
                for (let i = 0; i < bounds.length - 1; i++) {
                    const mid = (bounds[i] + bounds[i + 1]) / 2;
                    ctx.fillText(`#${i}`, xForTime(mid) - 6, H * 0.3);
                }

                // dead region beyond the end marker
                if (endLat !== null) {
                    const xe = xForTime(endTime());
                    ctx.fillStyle = DEAD_REGION;
                    ctx.fillRect(xe, 0, W - xe, H);
                    ctx.strokeStyle = selected === END_SEL ? END_SELECTED : END_COLOR;
                    ctx.lineWidth = selected === END_SEL ? 3 : 2;
                    ctx.beginPath();
                    ctx.moveTo(xe, 0); ctx.lineTo(xe, H);
                    ctx.stroke();
                    ctx.lineWidth = 1;
                }

                // cut markers
                cuts.forEach((c, i) => {
                    const x = xForTime(c.t);
                    ctx.strokeStyle = i === selected ? CUT_SELECTED : CUT_COLOR;
                    ctx.lineWidth = i === selected ? 3 : 2;
                    ctx.beginPath();
                    ctx.moveTo(x, 0); ctx.lineTo(x, H);
                    ctx.stroke();
                    ctx.lineWidth = 1;
                });

                // playhead
                ctx.strokeStyle = PLAYHEAD;
                ctx.beginPath();
                const px = xForTime(video.currentTime);
                ctx.moveTo(px, 0); ctx.lineTo(px, H);
                ctx.stroke();
            }

            // ---- pointer interaction ----------------------------------------
            let dragging = -1;
            function cutAtX(x) {
                const tol = 6 * (window.devicePixelRatio || 1);
                // end marker gets first claim so it stays grabbable next to a cut
                if (endLat !== null && Math.abs(xForTime(endTime()) - x) <= tol) return END_SEL;
                for (let i = 0; i < cuts.length; i++) {
                    if (Math.abs(xForTime(cuts[i].t) - x) <= tol) return i;
                }
                return -1;
            }
            function canvasX(e) {
                const rect = canvas.getBoundingClientRect();
                return (e.clientX - rect.left) * (canvas.width / rect.width);
            }
            canvas.addEventListener("pointerdown", (e) => {
                root.focus();
                if (e.button !== 0) return;
                const x = canvasX(e);
                const hit = cutAtX(x);
                if (hit === END_SEL) {
                    selected = END_SEL;
                    dragging = END_SEL;
                    seekFrame(endFrame());
                    canvas.setPointerCapture(e.pointerId);
                } else if (hit >= 0) {
                    selected = hit;
                    dragging = hit;
                    seekFrame(timeToFrame(cuts[hit].t));
                    canvas.setPointerCapture(e.pointerId);
                } else {
                    selected = -1;
                    video.currentTime = Math.max(0, Math.min(duration(), timeForX(x)));
                }
                draw();
            });
            canvas.addEventListener("pointermove", (e) => {
                if (dragging === -1) return;
                const t = Math.max(0, Math.min(duration(), timeForX(canvasX(e))));
                if (dragging === END_SEL) {
                    let E = Math.round(timeToFrame(t) / 8) + 1;
                    E = Math.max(lastBoundaryLat() + 1, Math.min(totalLatents(), E));
                    endLat = E;
                    seekFrame(endFrame());
                } else {
                    const p = snapBoundary(timeToFrame(t));
                    cuts[dragging].t = frameToTime(p);
                    seekFrame(p);
                }
                syncWidget();
                draw();
            });
            canvas.addEventListener("pointerup", () => {
                if (dragging === END_SEL) {
                    dragging = -1;
                    syncWidget();
                    draw();
                } else if (dragging >= 0) {
                    // Keep the clicked/dragged cut selected after release.
                    const t = cuts[dragging].t;
                    cuts.sort((a, b) => a.t - b.t);
                    selected = cuts.findIndex((c) => c.t === t);
                    dragging = -1;
                    syncWidget();
                    draw();
                }
            });
            canvas.addEventListener("contextmenu", (e) => {
                const hit = cutAtX(canvasX(e));
                if (hit === END_SEL) {
                    e.preventDefault();
                    e.stopPropagation();
                    endLat = null;
                    if (selected === END_SEL) selected = -1;
                    syncWidget();
                    draw();
                } else if (hit >= 0) {
                    e.preventDefault();
                    e.stopPropagation();
                    cuts.splice(hit, 1);
                    if (selected === hit) selected = -1;
                    else if (selected > hit) selected--;
                    syncWidget();
                    draw();
                }
            });
            canvas.addEventListener("dblclick", (e) => {
                const x = canvasX(e);
                if (cutAtX(x) >= 0) return;
                video.currentTime = Math.max(0, Math.min(duration(), timeForX(x)));
                addCutAtPlayhead();
            });

            // ---- keyboard -----------------------------------------------------
            root.addEventListener("keydown", (e) => {
                if (e.target.tagName === "INPUT") return;   // typing in the auto box
                const fine = e.altKey;
                const handled = () => { e.preventDefault(); e.stopPropagation(); };
                switch (e.key) {
                    case " ":
                        togglePlay(); handled(); break;
                    case "ArrowLeft":
                        if (selected >= 0) nudgeSelected(-1);
                        else stepPlayhead(fine ? -1 : -8);
                        handled(); break;
                    case "ArrowRight":
                        if (selected >= 0) nudgeSelected(1);
                        else stepPlayhead(fine ? 1 : 8);
                        handled(); break;
                    case "c": case "C":
                        addCutAtPlayhead(); handled(); break;
                    case "e": case "E":
                        setEndAtPlayhead(); handled(); break;
                    case "x": case "X":
                        deleteSelected(); handled(); break;
                    case "Escape":
                        selected = -1; draw(); handled(); break;
                }
            });

            // ---- readout ------------------------------------------------------
            function updateReadout() {
                const f = timeToFrame(video.currentTime);
                const Ls = boundaryLatents();
                let scene = 0;
                const myLat = frameToLatent(f);
                for (const L of Ls) { if (myLat >= L) scene++; }
                const lines = [];
                lines.push(
                    `t ${video.currentTime.toFixed(3)}s  |  frame ${f}  |  latent ${myLat}` +
                    `  |  scene #${scene}`
                );
                const fileInfo = video.videoWidth === 0
                    ? "audio file"
                    : (detectedFps ? `~${detectedFps.toFixed(2)}` : "play to detect");
                const capInfo = endLat !== null
                    ? `end @ frame ${endFrame()} -> cap ${8 * effLatents() - 7}`
                    : `no end marker (full video)`;
                lines.push(
                    `emit fps ${fps().toFixed(2)}  |  file fps ${fileInfo}  |  ` +
                    `${totalFrames()} frames -> ${effLatents()} latents  |  ${capInfo}`
                );
                if (selected === END_SEL && endLat !== null) {
                    lines.push(`selected: END marker at frame ${endFrame()} (latent ${effLatents()}) — ` +
                               `frames after this are excluded`);
                } else if (selected >= 0) {
                    const p = snapBoundary(timeToFrame(cuts[selected].t));
                    lines.push(`selected cut: scene #${cuts.indexOf(cuts[selected]) + 1} starts at ` +
                               `frame ${p} (latent ${boundaryLatent(p)})`);
                }
                lines.push(`scene_lengths: ${scenesWidget.value || "(none)"}  |  ` +
                           `${(scenesWidget.value || "").split("|").filter(Boolean).length} scenes`);
                if (scheduleMismatch) {
                    lines.push("⚠ saved schedule exceeds this media's length — string " +
                               "preserved, markers show what fits. Wrong file loaded?");
                }
                readout.textContent = lines.join("\n");
            }

            // re-derive when emit fps changes (time anchors preserved)
            const prevFpsCb = fpsWidget?.callback;
            if (fpsWidget) {
                fpsWidget.callback = function (...args) {
                    const rr = prevFpsCb?.apply(this, args);
                    cuts.forEach((c) => { c.t = frameToTime(snapBoundary(timeToFrame(c.t))); });
                    syncWidget();
                    draw();
                    return rr;
                };
            }

            video.addEventListener("loadedmetadata", () => {
                if (pendingReset) {
                    pendingReset = false;
                    cuts = [];
                    endLat = null;
                    selected = -1;
                    scheduleMismatch = false;
                    console.debug("[CutMarker] new media -> schedule reset");
                    syncWidget();   // emits a single full-length scene
                } else {
                    restoreFromWidget();
                }
                updateReadout();
                draw();
            });
            video.addEventListener("play",  () => { playBtn.textContent = "⏸"; });
            video.addEventListener("pause", () => { playBtn.textContent = "▶"; });

            // ---- lifecycle & render loop --------------------------------------
            // Dirty-flagged RAF: the loop ticks cheaply and only repaints when
            // state actually changed. It self-terminates when the widget leaves
            // the document (workflow reload paths that never fire onRemoved),
            // and onRemoved tears it down explicitly on node deletion —
            // otherwise every delete/re-add leaks a zombie loop + live <video>.
            let rafHandle = 0;
            let everConnected = false;
            let lastSig = "";

            function teardown() {
                if (rafHandle) { cancelAnimationFrame(rafHandle); rafHandle = 0; }
                try { video.pause(); } catch (e) { /* already dead */ }
                video.removeAttribute("src");
                video.load();   // makes the browser actually release the stream
                peaks = null;
                if (audioCtx) { audioCtx.close().catch(() => {}); audioCtx = null; }
            }

            function renderSig() {
                return [
                    video.currentTime.toFixed(3), duration().toFixed(3),
                    fps(), detectedFps ?? 0, selected, endLat ?? -1, dragging,
                    peaks ? 1 : 0, video.videoWidth, scheduleMismatch ? 1 : 0,
                    cuts.map((c) => c.t.toFixed(3)).join(","),
                    scenesWidget.value,
                    canvas.clientWidth, window.devicePixelRatio || 1,
                ].join("|");
            }

            function raf() {
                if (root.isConnected) {
                    everConnected = true;
                } else if (everConnected) {
                    teardown();   // widget left the document — stop for good
                    return;
                }
                const sig = renderSig();
                if (sig !== lastSig) {
                    lastSig = sig;
                    draw();
                    updateReadout();
                }
                rafHandle = requestAnimationFrame(raf);
            }
            rafHandle = requestAnimationFrame(raf);

            const prevOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                teardown();
                return prevOnRemoved?.apply(this, arguments);
            };

            // Exposed for the onConfigure hook: on workflow load/refresh,
            // onNodeCreated runs BEFORE serialized widget values are applied,
            // so the loadVideo() below fires with the combo's DEFAULT value.
            // onConfigure re-runs it with the restored filename (and metadata
            // then triggers restoreFromWidget against the right media).
            node._cutMarkerReload = loadVideo;

            loadVideo();
            return r;
        };
    },
});
