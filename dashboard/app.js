// decides which screen to show and fills it in.
// nothing on this page has a number in it until the user connects their own account.

const money = (n) => "$" + Number(n).toLocaleString(undefined, { maximumFractionDigits: 0 });
const el = (id) => document.getElementById(id);
const show = (id, visible) => el(id).classList.toggle("hidden", !visible);

const VIEWS = ["view-connect", "view-data", "view-setup", "view-forecast"];

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const error = new Error(body.detail || `${path} returned ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function showOnly(id) {
  VIEWS.forEach((view) => show(view, view === id));
  show("account", id !== "view-connect");
  show("pipeline-console", id !== "view-connect");
}

// -- which screen -----------------------------------------------------------

async function route() {
  const [session, squareConfig] = await Promise.all([
    api("/api/session"),
    api("/api/square/config").catch((error) => ({ ok: false, error: error.message })),
  ]);
  renderSquareConfig(squareConfig);

  if (!session.connected) {
    showOnly("view-connect");
    el("page-title").textContent = "Connect Square";
    el("subtitle").textContent = "";
    return;
  }

  el("page-title").textContent = "Tomorrow's sales";
  el("subtitle").textContent = session.business_name
    ? `${session.business_name} · Square ${session.environment}`
    : `Square ${session.environment}`;
  el("account-line").textContent =
    `${session.merchant_id} · connected ${new Date(session.connected_at).toLocaleDateString()}`;
  loadAccounts();
  renderPipeline(session.setup);

  if (session.setup_running) {
    showOnly("view-setup");
    pollSetup();
    return;
  }

  if (session.has_model) {
    showOnly("view-forecast");
    await loadForecast();
    return;
  }

  showOnly("view-setup");
  if (session.setup.finished_at) {
    renderSteps(session.setup);
    if (session.setup.error) {
      el("setup-error").textContent = session.setup.error;
      show("setup-error", true);
      show("retry", true);
    }
    return;
  }

  // A connected account enters the real pipeline immediately. This avoids a
  // second full Square history scan just to calculate preview numbers.
  await startSetup();
}

function renderSquareConfig(config) {
  const box = el("square-config");
  if (!box) return;

  if (!config.ok && config.error) {
    show("square-config", true);
    box.className = "notice bad";
    box.textContent = config.error;
    return;
  }

  if (!config.ok) {
    show("square-config", true);
    box.className = "notice bad";
    box.innerHTML = config.warnings.map((warning) => `<p>${warning}</p>`).join("");
    return;
  }

  show("square-config", false);
}

// -- more than one restaurant on this machine -------------------------------

async function loadAccounts() {
  const { accounts } = await api("/api/accounts");
  // only worth showing the switcher once there is something to switch to
  el("account-list").innerHTML = accounts.length < 2 ? "" : accounts
    .map((a) => `<li class="${a.current ? "current" : ""}">
        <span>${a.business_name || a.merchant_id}</span>
        <span class="muted">${a.has_model ? "forecast ready" : "not set up yet"}</span>
        ${a.current
          ? '<span class="tag">viewing</span>'
          : `<button class="button ghost switch" data-id="${a.merchant_id}">Switch</button>`}
      </li>`)
    .join("");
}

// -- what is in their square ------------------------------------------------

async function loadPreview() {
  show("preview", false);
  show("ready-to-predict", false);
  show("not-enough", false);
  show("preview-loading", true);

  let data;
  try {
    data = await api("/api/data/preview");
  } catch (error) {
    el("preview-loading").textContent = error.message;
    return;
  }

  show("preview-loading", false);
  show("preview", true);

  el("stat-days").textContent = data.trading_days.toLocaleString();
  el("stat-orders").textContent = data.sales_orders.toLocaleString();
  el("stat-total").textContent = money(data.total_sales);
  el("stat-avg").textContent = money(data.average_day);
  el("preview-range").textContent = data.first_sale
    ? `from ${data.first_sale} to ${data.last_sale}`
    : "no completed sales found";

  if (data.enough_to_forecast) {
    show("ready-to-predict", true);
  } else {
    el("not-enough-text").textContent =
      `You have ${data.trading_days} days of sales. We need about ${data.minimum_days} ` +
      `before a forecast means anything, so about ${data.days_needed} more days of trading.`;
    show("not-enough", true);
  }
}

// -- building it ------------------------------------------------------------

const ICONS = { waiting: "○", running: "◐", done: "●", failed: "✕" };
let setupPoll = null;

async function pollSetup() {
  let status;
  try {
    status = await api("/api/setup/status");
  } catch (error) {
    el("terminal-state").textContent = "connection error";
    appendTerminalError(error.message);
    setupPoll = setTimeout(pollSetup, 2500);
    return;
  }
  renderSteps(status);
  renderPipeline(status);

  if (status.running) {
    setupPoll = setTimeout(pollSetup, 1000);
    return;
  }
  if (status.error) {
    el("setup-error").textContent = status.error;
    show("setup-error", true);
    show("retry", true);
    return;
  }
  if (status.complete) route();
}

function renderSteps(status) {
  el("setup-steps").innerHTML = status.steps
    .map((step) => `<li class="step ${step.status}">
        <span class="icon">${ICONS[step.status] || "○"}</span>
        <span>${step.label}</span>
        ${step.detail ? `<span class="detail">${step.detail}</span>` : ""}
      </li>`)
    .join("");
}

function renderPipeline(status = {}) {
  const logs = status.logs || [];
  const output = el("terminal-output");
  const wasAtBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 40;

  output.replaceChildren(...logs.map((entry) => {
    const line = document.createElement("div");
    line.className = `terminal-line kind-${entry.kind}`;

    const time = document.createElement("span");
    time.className = "time";
    time.textContent = `[${entry.time}] `;
    const kind = document.createElement("span");
    kind.className = "kind";
    kind.textContent = `${entry.kind.padEnd(10)} `;
    line.append(time, kind, document.createTextNode(entry.message));
    return line;
  }));

  if (status.running) {
    const cursorLine = document.createElement("div");
    cursorLine.className = "terminal-line";
    const cursor = document.createElement("span");
    cursor.className = "terminal-cursor";
    cursorLine.append(cursor);
    output.append(cursorLine);
  }
  if (wasAtBottom) output.scrollTop = output.scrollHeight;

  el("terminal-state").textContent = status.running
    ? "running"
    : status.error
      ? "failed"
      : status.complete
        ? "complete"
        : "idle";

  const tables = el("pipeline-tables");
  tables.replaceChildren(...(status.tables || []).map(buildTerminalTable));
}

function buildTerminalTable(data) {
  const table = document.createElement("table");
  table.className = "terminal-table";
  const caption = document.createElement("caption");
  caption.textContent = `> ${data.title}`;
  table.append(caption);

  const head = table.createTHead().insertRow();
  data.columns.forEach((column) => {
    const cell = document.createElement("th");
    cell.textContent = column;
    head.append(cell);
  });

  const body = table.createTBody();
  data.rows.forEach((row) => {
    const tr = body.insertRow();
    row.forEach((value) => {
      const cell = tr.insertCell();
      cell.textContent = value;
    });
  });
  return table;
}

function appendTerminalError(message) {
  const line = document.createElement("div");
  line.className = "terminal-line kind-error";
  line.textContent = message;
  el("terminal-output").append(line);
}

async function startSetup() {
  if (setupPoll) clearTimeout(setupPoll);
  showOnly("view-setup");
  show("setup-error", false);
  show("retry", false);
  const status = await api("/api/setup/start", { method: "POST" });
  renderSteps(status);
  renderPipeline(status);
  pollSetup();
}

// -- the forecast -----------------------------------------------------------

async function loadForecast() {
  const [prediction, history, metrics] = await Promise.all([
    api("/predict"),
    api("/history?days=30"),
    api("/metrics"),
  ]);
  showPrediction(prediction);
  drawChart(history.points);
  showMetrics(metrics);
}

function showPrediction(data) {
  el("predicted").textContent = money(data.predicted_sales);
  el("prediction-date").textContent = new Date(data.business_date + "T12:00:00")
    .toLocaleDateString(undefined, { weekday: "long", month: "short", day: "numeric" });
  el("interval").textContent =
    `${data.interval_label}: ${money(data.interval_low)} to ${money(data.interval_high)}`;

  el("confidence").textContent = Math.round(data.confidence * 100) + "%";
  el("confidence-bar").style.width = Math.round(data.confidence * 100) + "%";
  el("uncertainty").textContent = `model spread ±${money(data.model_uncertainty)}`;
  el("orders").textContent = data.estimated_orders || "—";

  const features = data.important_features || [];
  const top = features[0] ? features[0].contribution : 1;
  el("features").innerHTML = features
    .map((f) => `<li>
        <span class="name">${f.direction === "up" ? "▲" : "▼"} ${f.name.replace(/_/g, " ")}</span>
        <span class="value">${f.value.toLocaleString()}</span>
        <span class="bar" style="width:${(f.contribution / top) * 100}%"></span>
      </li>`)
    .join("");
}

// hand drawn svg so the page needs no chart library
function drawChart(points) {
  if (!points.length) return;
  const width = 900, height = 240;
  const pad = { top: 16, right: 12, bottom: 28, left: 56 };

  const values = points.flatMap((p) => [p.actual, p.predicted]);
  const min = Math.min(...values) * 0.95;
  const max = Math.max(...values) * 1.05;

  const x = (i) => pad.left + (i / Math.max(points.length - 1, 1)) * (width - pad.left - pad.right);
  const y = (v) => pad.top + (1 - (v - min) / (max - min || 1)) * (height - pad.top - pad.bottom);
  const line = (key) => points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p[key])}`).join(" ");

  const ticks = [min, (min + max) / 2, max].map((v) =>
    `<text x="${pad.left - 8}" y="${y(v) + 4}" text-anchor="end" fill="currentColor"
       opacity=".5" font-size="11">${money(v)}</text>
     <line x1="${pad.left}" y1="${y(v)}" x2="${width - pad.right}" y2="${y(v)}"
       stroke="currentColor" opacity=".12"/>`).join("");

  const labels = [0, points.length - 1].map((i) =>
    `<text x="${x(i)}" y="${height - 8}" text-anchor="${i ? "end" : "start"}"
       fill="currentColor" opacity=".5" font-size="11">${points[i].business_date}</text>`).join("");

  el("chart").innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img"
         aria-label="actual versus predicted sales">
      ${ticks}${labels}
      <path d="${line("actual")}" fill="none" stroke="#8b98a5" stroke-width="2"/>
      <path d="${line("predicted")}" fill="none" stroke="#3b82f6" stroke-width="2"
            stroke-dasharray="5 3"/>
    </svg>`;
}

function showMetrics(data) {
  const rows = [
    ["Typical miss", money(data.mae)],
    ["RMSE, punishes big misses", money(data.rmse)],
    ["Average percent off", data.mape === null ? "n/a" : data.mape.toFixed(1) + "%"],
    ["An average day", money(data.mean_actual)],
    ["Days tested", data.n_days],
  ];
  document.querySelector("#metrics tbody").innerHTML =
    rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  document.querySelector("#baselines tbody").innerHTML = data.baselines
    .map((b) => `<tr><td>${b.name}</td><td>${money(b.mae)}</td>
      <td>${b.improvement > 0 ? money(b.improvement) : "—"}</td>
      <td>${b.model_wins ? "✓ we win" : "✗ baseline wins"}</td></tr>`).join("");

  const verdict = el("verdict");
  verdict.className = "verdict " + (data.beats_all_baselines ? "win" : "lose");
  verdict.textContent = data.beats_all_baselines
    ? "Your model beats every simple guess on days it has never seen."
    : "A simple guess is as good or better. Treat these predictions with care.";
}

// -- buttons ----------------------------------------------------------------

document.addEventListener("click", async (event) => {
  const id = event.target.id;

  if (id === "predict-button" || id === "retry" || id === "rerun") {
    event.target.disabled = true;
    await startSetup();
    event.target.disabled = false;
  }
  if (id === "recheck") loadPreview();
  if (event.target.classList.contains("switch")) {
    await fetch(`/api/accounts/${event.target.dataset.id}/select`, { method: "POST" });
    location.href = "/";
  }
  if (id === "disconnect") {
    if (!confirm("This deletes your data and your model. Continue?")) return;
    await fetch("/api/account", { method: "DELETE" });
    location.href = "/";
  }
});

// -- start ------------------------------------------------------------------

const params = new URLSearchParams(location.search);
if (params.get("error")) {
  el("banner").textContent = params.get("error");
  show("banner", true);
}

route().catch((error) => {
  el("subtitle").textContent = error.message;
  el("subtitle").className = "error";
  showOnly("view-connect");
});
