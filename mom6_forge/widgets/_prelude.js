// mom6_forge widget prelude, prepended to every widget's .js file (see _base.py).
//
// Rules for widget front ends (written and maintained by AI; nobody on the project
// debugs JavaScript by hand, so errors must surface on their own):
//  * Draw state from model traits and send one message per user action:
//      model.send({ op: "name", ...fields }, undefined, [buffers])
//    The Python class turns each op into mom6_forge calls. No mom6_forge logic here.
//  * Wrap every event handler, model.on callback and timer in guard(...), so an error
//    reaches the person (a visible box) and Python (widget.js_errors).
//  * Arrays arrive as bytes traits (DataView); never ask Python for JSON lists.
//  * Plain DOM and canvas only, no build step. Colours come from Jupyter's CSS
//    variables with fallbacks, so the widget follows light and dark themes.
//  * render() is synchronous and returns a cleanup function that removes its listeners.
//  * End the file with:  export default defineWidget(render);

const MF_ERROR_STYLE =
  "border:1px solid var(--jp-error-color1, #b42318);border-radius:6px;padding:8px 10px;" +
  "margin:6px 0;font:13px/1.4 var(--jp-ui-font-family, system-ui, sans-serif);" +
  "color:var(--jp-error-color1, #b42318);background:var(--jp-layout-color1, transparent)";

function reportError(model, el, err) {
  const message = String((err && err.message) || err);
  const stack = String((err && err.stack) || "");
  try {
    model.send({ op: "js_error", message, stack });
  } catch (_) {
    // The kernel may be gone; the box below still tells the person.
  }
  let box = el.querySelector(":scope > .mf-js-error");
  if (!box) {
    box = document.createElement("div");
    box.className = "mf-js-error";
    box.style.cssText = MF_ERROR_STYLE;
    el.prepend(box);
  }
  const head = document.createElement("div");
  head.textContent =
    "This widget hit an error. Copy the text below to your AI assistant; " +
    "Python also has it as widget.js_errors.";
  const pre = document.createElement("pre");
  pre.style.cssText = "white-space:pre-wrap;margin:6px 0 0;font:12px ui-monospace, monospace";
  pre.textContent = message + (stack ? "\n" + stack : "");
  box.replaceChildren(head, pre);
}

// A box that shows the `errors` trait: what went wrong in Python on the last action.
function errorBox(model, guard, cleanups) {
  const box = document.createElement("div");
  box.className = "mf-errors";
  box.style.cssText = MF_ERROR_STYLE;
  const show = guard(() => {
    const errors = model.get("errors") || [];
    box.hidden = errors.length === 0;
    box.textContent = errors.join("\n");
  });
  model.on("change:errors", show);
  cleanups.push(() => model.off("change:errors", show));
  show();
  return box;
}

function defineWidget(render) {
  return {
    render(ctx) {
      const { model, el } = ctx;
      const guard =
        (fn) =>
        (...args) => {
          try {
            return fn(...args);
          } catch (err) {
            reportError(model, el, err);
          }
        };
      const cleanups = []; // the prelude's own listeners, removed with the widget's
      const cleanAll = () => cleanups.forEach((fn) => fn());
      try {
        const cleanup = render({ ...ctx, guard, errorBox: () => errorBox(model, guard, cleanups) });
        return () => {
          if (typeof cleanup === "function") guard(cleanup)();
          cleanAll();
        };
      } catch (err) {
        reportError(model, el, err);
        return cleanAll;
      }
    },
  };
}
