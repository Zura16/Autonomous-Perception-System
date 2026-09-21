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
