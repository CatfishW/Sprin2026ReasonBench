function esc(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function formatInt(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '0';
  }
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(numeric);
}

function formatFloat(value, digits = 3) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return '--';
  }
  return numeric.toFixed(digits);
}

function formatTimestamp(raw) {
  if (!raw) {
    return '--';
  }
  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    return esc(raw);
  }
  return parsed.toLocaleString();
}

function secondsBetween(startRaw, endRaw) {
  if (!startRaw || !endRaw) {
    return null;
  }
  const start = new Date(startRaw);
  const end = new Date(endRaw);
  if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
    return null;
  }
  const seconds = (end.getTime() - start.getTime()) / 1000;
  return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
}

function estimateRemainingSeconds(completedValue, expectedValue, elapsedSeconds) {
  const completed = Number(completedValue || 0);
  const expected = Number(expectedValue || 0);
  const elapsed = Number(elapsedSeconds || 0);
  if (!Number.isFinite(completed) || !Number.isFinite(expected) || !Number.isFinite(elapsed)) {
    return null;
  }
  if (expected <= 0 || completed <= 0 || completed >= expected || elapsed <= 0) {
    return null;
  }
  const rate = completed / elapsed;
  if (!Number.isFinite(rate) || rate <= 0) {
    return null;
  }
  return (expected - completed) / rate;
}

function parseElapsedSecondsFromProgressLine(line) {
  const raw = String(line || '');
  const match = raw.match(/elapsed_s=([0-9]+(?:\.[0-9]+)?)/);
  if (!match) {
    return null;
  }
  const seconds = Number(match[1]);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
}

function formatDuration(secondsValue) {
  const seconds = Number(secondsValue);
  if (!Number.isFinite(seconds) || seconds < 0) {
    return '--';
  }

  const rounded = Math.max(0, Math.round(seconds));
  if (rounded < 60) {
    return `${rounded}s`;
  }

  const days = Math.floor(rounded / 86400);
  const hours = Math.floor((rounded % 86400) / 3600);
  const minutes = Math.floor((rounded % 3600) / 60);

  if (days > 0) {
    return `${days}d ${hours}h ${minutes}m`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
}

function detailOpenAttr(openState, key, fallbackOpen = false) {
  if (openState && openState.has(key)) {
    return 'open';
  }
  if (!openState && fallbackOpen) {
    return 'open';
  }
  return '';
}

let selectedStatusPath = 'status.json';

function normalizeStatusPath(pathValue) {
  const raw = String(pathValue || '').trim();
  if (!raw) {
    return 'status.json';
  }
  if (raw.startsWith('/')) {
    return raw.slice(1);
  }
  return raw;
}

function historyOptionLabel(item) {
  const title = String(item?.run_title || item?.run_tag || 'snapshot');
  const launched = formatTimestamp(item?.launched_at || '');
  const generated = formatTimestamp(item?.generated_at || '');
  const viewKind = String(item?.view_kind || 'latest').replaceAll('_', ' ');
  const active = item?.is_active ? ' [active]' : '';
  return `${title}${active} | ${viewKind} | launched: ${launched} | snapshot: ${generated}`;
}

function updateHistorySelector(indexPayload) {
  const select = document.getElementById('historySelect');
  const hint = document.getElementById('historyHint');
  if (!select || !hint) {
    return;
  }

  const runs = Array.isArray(indexPayload?.runs) ? indexPayload.runs : [];
  const options = [{
    value: 'status.json',
    label: 'Current Live | status.json',
  }];
  const seen = new Set(['status.json']);

  for (const item of runs) {
    // Skip active "latest" mirror because it is equivalent to Current Live.
    if (item?.is_active && String(item?.view_kind || '') === 'latest') {
      continue;
    }
    const statusPath = normalizeStatusPath(item?.status_path || '');
    if (!statusPath || statusPath === 'status.json') {
      continue;
    }
    if (seen.has(statusPath)) {
      continue;
    }
    seen.add(statusPath);
    options.push({
      value: statusPath,
      label: historyOptionLabel(item),
    });
  }

  const existingValue = normalizeStatusPath(selectedStatusPath);
  const validValues = new Set(options.map((option) => option.value));
  if (!validValues.has(existingValue)) {
    selectedStatusPath = normalizeStatusPath(indexPayload?.active_status_path || 'status.json');
  }

  select.innerHTML = options
    .map((option) => `<option value="${esc(option.value)}">${esc(option.label)}</option>`)
    .join('');
  select.value = normalizeStatusPath(selectedStatusPath);

  const selectedOption = options.find((option) => option.value === select.value);
  if (selectedOption && select.value !== 'status.json') {
    hint.textContent = `Showing archived snapshot: ${selectedOption.label}`;
  } else {
    hint.textContent = 'Showing current live status.';
  }
}

async function loadRunHistoryIndex() {
  const now = Date.now();
  const candidates = [
    `run_history_v2.json?t=${now}`,
    `run_history.json?t=${now}`,
  ];

  for (const url of candidates) {
    try {
      const response = await fetch(url, { cache: 'no-store' });
      if (!response.ok) {
        continue;
      }
      const payload = await response.json();
      return payload;
    } catch {
      // Try next candidate.
    }
  }
  return null;
}

async function fetchStatusPayload(statusPath) {
  const normalizedPath = normalizeStatusPath(statusPath);
  const response = await fetch(`${normalizedPath}?t=${Date.now()}`, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

function metricRollupMarkup(liveSummary) {
  const entries = Object.entries(liveSummary?.metric_rollup || {});
  if (entries.length === 0) {
    return '<p class="muted">No rollup metrics yet.</p>';
  }

  return `
    <div class="metric-pills">
      ${entries
        .map(
          ([name, value]) => `
            <span class="metric-pill">
              <label>${esc(name)}</label>
              <strong>${formatFloat(value, 4)}</strong>
            </span>
          `,
        )
        .join('')}
    </div>
  `;
}

function leaderboardMarkup(liveSummary) {
  const rows = liveSummary?.leaderboard || [];
  const primaryLabelRaw = String(liveSummary?.primary_metric || 'primary');
  const primaryLabel = primaryLabelRaw.replaceAll('_', ' ');
  if (rows.length === 0) {
    return '<p class="muted">No leaderboard rows yet.</p>';
  }

  return `
    <div class="table-wrap">
      <table class="leaderboard-table">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>${esc(primaryLabel)}</th>
            <th>Records</th>
            <th>API</th>
            <th>Wall(s)</th>
            <th>Cache</th>
          </tr>
        </thead>
        <tbody>
          ${rows
            .map((row) => {
              const strategyMetrics = Object.entries(row.metric_means || {})
                .map(([name, value]) => `${esc(name)}=${formatFloat(value, 4)}`)
                .join(' | ');

              return `
                <tr>
                  <td>
                    <div class="strategy-cell">
                      <strong>${esc(row.strategy || 'unknown')}</strong>
                      <span class="mono small muted">${esc(strategyMetrics || 'no metric means')}</span>
                    </div>
                  </td>
                  <td>${formatFloat(row.mean_primary_score, 4)}</td>
                  <td>${formatInt(row.records)}</td>
                  <td>${formatFloat(row.mean_api_calls, 3)}</td>
                  <td>${formatFloat(row.mean_wall_time_s, 3)}</td>
                  <td>${formatFloat(row.cache_hit_rate, 4)}</td>
                </tr>
              `;
            })
            .join('')}
        </tbody>
      </table>
    </div>
  `;
}

function sessionCard(session, openState, runElapsedSeconds) {
  const logText = (session.last_log_lines || []).join('\n') || 'No logs yet.';
  const displayName = session.display_name || session.name;
  const pct = Number(session.progress_pct || 0);
  const detailsKeyBase = esc(session.name || 'session');
  const liveSummary = session.live_summary || {};
  const isStuck = Boolean(session.stuck);
  const staleRetryStreak = Number(session.stale_retry_streak || 0);
  const stuckReason = session.stuck_reason || '';
  const evalMode = String(liveSummary.evaluation_mode || 'proxy').replaceAll('_', ' ');
  const primaryMetric = String(liveSummary.primary_metric || 'primary').replaceAll('_', ' ');
  const officialStatus = String(liveSummary.official_status || '');
  const officialMessage = String(liveSummary.official_message || '');
  const officialSummaryPath = String(liveSummary.official_summary_path || '');

  const detailsSummary = [
    `records scanned: ${formatInt(liveSummary.records_scanned || 0)}`,
    `best strategy: ${esc(liveSummary.best_strategy || 'N/A')}`,
    `eval mode: ${esc(evalMode)}`,
    `primary metric: ${esc(primaryMetric)}`,
  ].join(' | ');

  const completedRecords = formatInt(session.completed_records);
  const expectedRecords = formatInt(session.expected_records);
  const remainingRecords = formatInt(session.remaining_records);
  const strategyCount = formatInt(session.strategy_count);
  const exampleCount = formatInt(session.example_count);

  const sessionElapsedSeconds =
    parseElapsedSecondsFromProgressLine(session.last_progress_line) || runElapsedSeconds;
  const sessionEtaSeconds = estimateRemainingSeconds(
    session.completed_records,
    session.expected_records,
    sessionElapsedSeconds,
  );
  const sessionEtaLabel = formatDuration(sessionEtaSeconds);
  const sessionElapsedLabel = formatDuration(sessionElapsedSeconds);

  return `
    <article class="session-card card" data-session="${esc(session.name)}">
      <div class="session-head">
        <h2 class="session-name">${esc(displayName)}</h2>
        <div class="session-badges">
          <span class="badge ${esc(session.state)}">${esc(session.state.replaceAll('_', ' '))}</span>
          ${isStuck ? '<span class="badge stuck">stuck</span>' : ''}
        </div>
      </div>
      <div class="session-subhead">
        <span class="chip">${esc(session.dataset_kind || 'dataset')}</span>
        <span class="chip">${esc(session.model || 'model')}</span>
        <span class="chip">run: ${esc(session.run_tag || '--')}</span>
      </div>
      <div class="metrics">
        <div class="metric">
          <label>Progress</label>
          <strong>${pct.toFixed(2)}%</strong>
        </div>
        <div class="metric metric--wide">
          <label>Records</label>
          <strong>${completedRecords} / ${expectedRecords}</strong>
        </div>
        <div class="metric">
          <label>ETA</label>
          <strong>${esc(sessionEtaLabel)}</strong>
        </div>
        <div class="metric">
          <label>Stale Retry</label>
          <strong>${formatInt(staleRetryStreak)}</strong>
        </div>
      </div>
      <div class="progress-track"><div class="progress-bar" style="width:${pct}%;"></div></div>

      <details class="detail-block" data-section="live" ${detailOpenAttr(openState, `${detailsKeyBase}:live`, false)}>
        <summary>Live Metrics</summary>
        <p class="mono small muted">${detailsSummary}</p>
        ${officialStatus ? `<p class="mono small muted"><strong>official status:</strong> ${esc(officialStatus)}</p>` : ''}
        ${officialMessage ? `<p class="mono small muted"><strong>official note:</strong> ${esc(officialMessage)}</p>` : ''}
        ${officialSummaryPath ? `<p class="mono small muted"><strong>official summary:</strong> ${esc(officialSummaryPath)}</p>` : ''}
        ${metricRollupMarkup(liveSummary)}
        ${leaderboardMarkup(liveSummary)}
      </details>

      <details class="detail-block" data-section="signals" ${detailOpenAttr(openState, `${detailsKeyBase}:signals`, isStuck)}>
        <summary>Signals</summary>
        <p class="signal-line mono"><strong>Completed / Expected:</strong> ${completedRecords} / ${expectedRecords}</p>
        <p class="signal-line mono"><strong>Remaining:</strong> ${remainingRecords}</p>
        <p class="signal-line mono"><strong>Elapsed:</strong> ${esc(sessionElapsedLabel)}</p>
        <p class="signal-line mono"><strong>Estimated remaining:</strong> ${esc(sessionEtaLabel)}</p>
        <p class="signal-line mono"><strong>Examples:</strong> ${exampleCount}</p>
        <p class="signal-line mono"><strong>Strategies:</strong> ${strategyCount}</p>
        <p class="signal-line mono"><strong>Stale retry streak:</strong> ${formatInt(staleRetryStreak)}${stuckReason ? ` (${esc(stuckReason)})` : ''}</p>
        <p class="signal-line mono"><strong>Last exit code:</strong> ${esc(session.last_exit_code || 'N/A')}</p>
        <p class="signal-line mono"><strong>Last progress:</strong> ${esc(session.last_progress_line || 'N/A')}</p>
        <p class="signal-line mono"><strong>Last error:</strong> ${esc(session.last_error_line || 'N/A')}</p>
      </details>

      <details class="detail-block" data-section="logs" ${detailOpenAttr(openState, `${detailsKeyBase}:logs`, false)}>
        <summary>Paths & Logs</summary>
        <p class="path mono">config: ${esc(session.config_path)}</p>
        <p class="path mono">output: ${esc(session.output_dir || 'N/A')}</p>
        <p class="path mono">checkpoint: ${esc(session.checkpoint_path)}</p>
        <p class="path mono">latest log: ${esc(session.latest_log || 'N/A')}</p>
        <pre class="log-snippet mono">${esc(logText)}</pre>
      </details>
    </article>
  `;
}

function captureOpenDetailState() {
  const state = new Set();
  document.querySelectorAll('.session-card details[open]').forEach((details) => {
    const card = details.closest('.session-card');
    const sessionName = card?.dataset.session;
    const section = details.dataset.section;
    if (sessionName && section) {
      state.add(`${sessionName}:${section}`);
    }
  });
  return state;
}

function setAllDetails(open) {
  document.querySelectorAll('.session-card details').forEach((details) => {
    details.open = open;
  });
}

async function refresh() {
  try {
    const runHistory = await loadRunHistoryIndex();
    if (runHistory) {
      updateHistorySelector(runHistory);
    }

    const data = await fetchStatusPayload(selectedStatusPath);

    const overallPct = Number(data.overall?.progress_pct || 0);
    const overallBar = document.getElementById('overallBar');
    const overallPctText = document.getElementById('overallPct');
    const overallCounts = document.getElementById('overallCounts');
    const overallEta = document.getElementById('overallEta');
    const generatedAt = document.getElementById('generatedAt');
    const runTitle = document.getElementById('runTitle');
    const runTag = document.getElementById('runTag');
    const launchedAt = document.getElementById('launchedAt');
    const grid = document.getElementById('sessionGrid');
    const hasPreviousCards = grid.children.length > 0;
    const openState = hasPreviousCards ? captureOpenDetailState() : null;

    const runElapsedSeconds = secondsBetween(data.launched_at, data.generated_at);
    const overallEtaSeconds = estimateRemainingSeconds(
      data.overall?.completed_records || 0,
      data.overall?.expected_records || 0,
      runElapsedSeconds,
    );

    overallBar.style.width = `${overallPct}%`;
    overallPctText.textContent = `${overallPct.toFixed(2)}%`;
    overallCounts.textContent = `${formatInt(data.overall?.completed_records || 0)} / ${formatInt(data.overall?.expected_records || 0)} records`;
    if (overallEta) {
      overallEta.textContent = `Estimated remaining: ${formatDuration(overallEtaSeconds)}`;
    }
    generatedAt.textContent = `Last update: ${formatTimestamp(data.generated_at)}`;
    runTitle.textContent = data.run_title || 'ReasonBench Live Sessions';
    runTag.textContent = `Run Tag: ${data.run_tag || '--'}`;
    launchedAt.textContent = `Launched: ${formatTimestamp(data.launched_at)}`;

    const sessions = data.sessions || [];
    grid.innerHTML = sessions.map((session) => sessionCard(session, openState, runElapsedSeconds)).join('');
  } catch (error) {
    const historyHint = document.getElementById('historyHint');
    if (historyHint && normalizeStatusPath(selectedStatusPath) !== 'status.json') {
      historyHint.textContent = `Selected archived snapshot failed to load: ${String(error.message || error)}`;
    }
    document.getElementById('sessionGrid').innerHTML = `
      <article class="session-card card">
        <h2 class="session-name">Status source unavailable</h2>
        <p class="mono">${esc(error.message || error)}</p>
      </article>
    `;
  }
}

document.getElementById('expandAll')?.addEventListener('click', () => setAllDetails(true));
document.getElementById('collapseAll')?.addEventListener('click', () => setAllDetails(false));
document.getElementById('historySelect')?.addEventListener('change', (event) => {
  const target = event.target;
  if (!(target instanceof HTMLSelectElement)) {
    return;
  }
  selectedStatusPath = normalizeStatusPath(target.value || 'status.json');
  refresh();
});

refresh();
setInterval(refresh, 10000);
