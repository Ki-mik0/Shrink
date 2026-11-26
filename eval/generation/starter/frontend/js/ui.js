// Small DOM helpers shared by all pages.
//
// The rule that keeps pages safe: text that comes from users or from the API
// is always inserted as text (textContent), never as HTML. el() below does that
// for you; do not build markup with innerHTML from such values.

/**
 * Create an element with attributes and children.
 *
 *   el("li", { class: "todo", dataset: { id: 7 } }, el("span", {}, todo.title))
 *
 * Strings among the children become text nodes, so they are never parsed as
 * HTML. Attribute names starting with "on" are added as event listeners:
 *   el("button", { type: "button", onclick: () => save() }, "Save")
 *
 * @param {string} tag
 * @param {object} [attrs] Attributes; `class`, `dataset` and `on*` are handled specially.
 * @param {...(Node|string|number|null|undefined|false)} children
 * @returns {HTMLElement}
 */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (name === "class") node.className = value;
    else if (name === "dataset") Object.assign(node.dataset, value);
    else if (name.startsWith("on") && typeof value === "function") {
      node.addEventListener(name.slice(2).toLowerCase(), value);
    } else if (value === true) node.setAttribute(name, "");
    else node.setAttribute(name, String(value));
  }
  for (const child of children.flat()) {
    if (child === undefined || child === null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Remove every child of `node`. */
export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

let messageTimer = null;

/**
 * Show a short message in the shared message area (#message).
 *
 * @param {string} text Plain text; it is not interpreted as HTML.
 * @param {"info"|"success"|"error"} [kind="info"]
 * @param {number} [timeout=4000] Hide after this many milliseconds; 0 keeps it.
 */
export function showMessage(text, kind = "info", timeout = 4000) {
  const box = document.getElementById("message");
  if (!box) return;
  box.textContent = text;
  box.className = `message message-${kind}`;
  box.hidden = false;
  clearTimeout(messageTimer);
  if (timeout > 0) messageTimer = setTimeout(() => { box.hidden = true; }, timeout);
}

/** Read a form's fields into a plain object, trimming text values. */
export function formValues(form) {
  const values = {};
  for (const [name, value] of new FormData(form).entries()) {
    values[name] = typeof value === "string" ? value.trim() : value;
  }
  return values;
}

/**
 * Disable a form's controls while `work` runs, so it cannot be submitted twice.
 *
 * @param {HTMLFormElement} form
 * @param {() => Promise<any>} work
 */
export async function whileBusy(form, work) {
  const controls = [...form.elements];
  controls.forEach((control) => { control.disabled = true; });
  try {
    return await work();
  } finally {
    controls.forEach((control) => { control.disabled = false; });
  }
}

/** Format an ISO 8601 timestamp from the API for display in the user's locale. */
export function formatTime(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}
