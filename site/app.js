/* Autonomous Perception System — static evidence page.
 *
 * This file used to be a client for a Flask backend at 127.0.0.1:5000, with
 * play/pause/stop controls, a "radar" obstacle spawner and a MiDaS depth
 * toggle. All of that belonged to the archived v1 system, and none of it
 * corresponds to anything in this repository: APS is an offline, open-loop
 * evaluation with no service, no radar and no actuation. Pointing a public page
 * at a localhost API nobody runs meant every control on it was permanently
 * broken, and the words on it contradicted the project's own scope rules.
 *
 * What remains is the one thing worth keeping: the cursor-proximity border glow
 * that matches the rest of aalind.org. The page is otherwise static, because
 * the evidence is static.
 */

function initBorderGlows() {
  const cards = document.querySelectorAll(".border-glow-card");
  if (!cards.length) return;

  // Pointer devices only. On touch there is no cursor to be near, and the
  // effect would either never fire or latch on at the last tap position.
  if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;

  let queued = false;
  let lastEvent = null;

  const paint = () => {
    queued = false;
    const e = lastEvent;
    if (!e) return;

    const maxDistance =
      Math.sqrt(window.innerWidth ** 2 + window.innerHeight ** 2) / 3;

    cards.forEach((card) => {
      const rect = card.getBoundingClientRect();

      // Skip cards scrolled out of view: their glow cannot be seen, and on a
      // long page this is most of them.
      if (rect.bottom < 0 || rect.top > window.innerHeight) {
        card.style.setProperty("--edge-proximity", 0);
        return;
      }

      const dx = e.clientX - (rect.left + rect.width / 2);
      const dy = e.clientY - (rect.top + rect.height / 2);
      const distance = Math.sqrt(dx * dx + dy * dy);

      const inside =
        e.clientX >= rect.left &&
        e.clientX <= rect.right &&
        e.clientY >= rect.top &&
        e.clientY <= rect.bottom;

      const proximity = inside
        ? 100
        : Math.max(0, Math.min(100, 100 * (1 - distance / maxDistance)));

      const angle = (Math.atan2(dy, dx) * (180 / Math.PI) + 360) % 360;

      card.style.setProperty("--edge-proximity", proximity);
      card.style.setProperty("--cursor-angle", `${angle}deg`);
    });
  };

  // The original recomputed every card's geometry on every mousemove, which on
  // this page is ~10 cards per event. Coalescing to one paint per animation
  // frame keeps it off the main thread's critical path.
  document.addEventListener(
    "mousemove",
    (e) => {
      lastEvent = e;
      if (!queued) {
        queued = true;
        requestAnimationFrame(paint);
      }
    },
    { passive: true }
  );
}

document.addEventListener("DOMContentLoaded", initBorderGlows);

/* ================== OFFLINE REPLAY PLAYER ==================
 *
 * Plays frames that app/replay.py already composited, in step with the
 * telemetry those same frames were drawn from. It deliberately does NOT redraw
 * boxes or recompute anything: the image is the authority, and the table beside
 * it is the exported record. A JavaScript reimplementation of the HUD's
 * abstention and envelope rules would be a second source of display truth, free
 * to drift from the Python one.
 *
 * `null` in the telemetry means the estimator declined. It renders as "--",
 * never as 0.
 */

const DRIVE = "2011_09_26_drive_0084";
const DATA = `data/${DRIVE}`;

function fmt(value, unit) {
  return value === null || value === undefined ? "--" : `${value.toFixed(1)}${unit}`;
}

async function initReplay() {
  const img = document.getElementById("replay-frame");
  const scrub = document.getElementById("replay-scrub");
  const playBtn = document.getElementById("replay-play");
  const counter = document.getElementById("replay-counter");
  const banner = document.getElementById("replay-banner");
  const tbody = document.querySelector("#replay-table tbody");
  const loading = document.getElementById("replay-loading");
  const loadingText = document.getElementById("loading-text");
  if (!img || !scrub) return;

  let meta, telemetry;
  try {
    [meta, telemetry] = await Promise.all([
      fetch(`${DATA}/meta.json`).then((r) => r.json()),
      fetch(`${DATA}/telemetry.json`).then((r) => r.json()),
    ]);
  } catch (err) {
    // Say what failed. A dead player that looks merely slow is worse than one
    // that admits it could not load.
    loadingText.textContent = "replay data unavailable";
    console.error("replay: could not load telemetry", err);
    return;
  }

  const n = Math.min(meta.frames, telemetry.length);
  scrub.max = String(n - 1);
  img.width = meta.width;
  img.height = meta.height;

  const setText = (id, value) => {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
  };
  setText("meta-drive", meta.drive.replace("2011_09_26_", ""));
  setText("meta-envelope", `${meta.envelope_m[0]}–${meta.envelope_m[1]} m`);
  const objectFrames = telemetry.reduce((a, f) => a + f.objects.length, 0);
  const declined = telemetry.reduce(
    (a, f) => a + f.objects.filter((o) => o.range_m === null).length,
    0
  );
  setText("meta-abstain", `${Math.round((100 * declined) / objectFrames)}%`);

  const src = (i) => `${DATA}/${String(i).padStart(4, "0")}.jpg`;

  let index = 0;
  let playing = false;
  let timer = null;

  function render(i) {
    index = Math.max(0, Math.min(n - 1, i));
    img.src = src(index);
    scrub.value = String(index);
    const f = telemetry[index];
    counter.textContent = `frame ${f.frame} · ${f.t_s.toFixed(2)} s · ${index + 1}/${n}`;

    if (f.aeb_request) {
      banner.hidden = false;
      banner.className = "replay-banner banner-aeb";
      banner.textContent = "AEB REQUEST — logged, not actuated";
    } else if (f.warning) {
      banner.hidden = false;
      banner.className = "replay-banner banner-warn";
      banner.textContent = "FORWARD COLLISION WARNING";
    } else {
      banner.hidden = true;
    }

    tbody.innerHTML = "";
    if (!f.objects.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="muted">no tracked objects</td></tr>';
      return;
    }
    for (const o of f.objects.slice().sort((a, b) => (a.range_m ?? 1e9) - (b.range_m ?? 1e9))) {
      const tr = document.createElement("tr");
      if (o.warning) tr.className = "row-warn";
      const range =
        o.range_m === null
          ? '<span class="muted" title="estimator declined">--</span>'
          : o.in_envelope
            ? `<b>${o.range_m.toFixed(1)} m</b>`
            : `<span class="muted" title="outside the credible envelope">${o.range_m.toFixed(1)} m</span>`;
      tr.innerHTML =
        `<td>#${o.id}</td><td>${o.group}</td><td>${range}</td>` +
        `<td>${fmt(o.ttc_s, " s")}</td><td>${fmt(o.lateral_m, " m")}</td>`;
      tbody.appendChild(tr);
    }
  }

  function stop() {
    playing = false;
    clearInterval(timer);
    playBtn.textContent = "▶ Play";
  }

  function play() {
    if (index >= n - 1) index = -1;
    playing = true;
    playBtn.textContent = "⏸ Pause";
    // Real sensor rate, from the measured timestamps -- not a round 10 Hz.
    timer = setInterval(() => {
      if (index >= n - 1) {
        stop();
        return;
      }
      render(index + 1);
    }, 1000 / meta.fps);
  }

  playBtn.addEventListener("click", () => (playing ? stop() : play()));
  scrub.addEventListener("input", () => {
    stop();
    render(Number(scrub.value));
  });
  document.addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
    if (document.activeElement !== scrub) return;
    stop();
  });

  // Warm the browser cache in the background so scrubbing is not a slideshow.
  // All requests are issued at once and the browser's own per-host connection
  // limit does the queueing -- at ~50 KB a frame there is nothing to gain from
  // scheduling it by hand.
  render(0);
  let loaded = 0;
  for (let i = 0; i < n; i++) {
    const pre = new Image();
    pre.onload = pre.onerror = () => {
      loaded += 1;
      if (loaded === n) {
        loading.classList.add("hidden");
      } else if (loaded % 10 === 0) {
        loadingText.textContent = `buffering ${Math.round((100 * loaded) / n)}%`;
      }
    };
    pre.src = src(i);
  }
}

document.addEventListener("DOMContentLoaded", initReplay);
