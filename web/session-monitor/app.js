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

function detailOpenAttr(openState, key, fallbackOpen = false) {
  if (openState && openState.has(key)) {
    return 'open';
  }
  if (!openState && fallbackOpen) {
    return 'open';
  }
  return '';
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
  if (rows.length === 0) {
    return '<p class="muted">No leaderboard rows yet.</p>';
  }

  return `
    <div class="table-wrap">
      <table class="leaderboard-table">
        <thead>
          <tr>
            <th>Strategy</th>
            <th>Primary</th>
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

function sessionCard(session, openState) {
  const logText = (session.last_log_lines || []).join('\n') || 'No logs yet.';
  const displayName = session.display_name || session.name;
  const pct = Number(session.progress_pct || 0);
  const detailsKeyBase = esc(session.name || 'session');
  const liveSummary = session.live_summary || {};
  const isStuck = Boolean(session.stuck);
  const staleRetryStreak = Number(session.stale_retry_streak || 0);
  const stuckReason = session.stuck_reason || '';

  const detailsSummary = [
    `records scanned: ${formatInt(liveSummary.records_scanned || 0)}`,
    `best strategy: ${esc(liveSummary.best_strategy || 'N/A')}`,
  ].join(' | ');

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
        <div class="metric">
          <label>Completed</label>
          <strong>${esc(session.completed_records)}</strong>
        </div>
        <div class="metric">
          <label>Expected</label>
          <strong>${formatInt(session.expected_records)}</strong>
        </div>
        <div class="metric">
          <label>Remaining</label>
          <strong>${formatInt(session.remaining_records)}</strong>
        </div>
        <div class="metric">
          <label>Strategies</label>
          <strong>${formatInt(session.strategy_count)}</strong>
        </div>
        <div class="metric">
          <label>Examples</label>
          <strong>${formatInt(session.example_count)}</strong>
        </div>
      </div>
      <div class="progress-track"><div class="progress-bar" style="width:${pct}%;"></div></div>

      <details class="detail-block" data-section="live" ${detailOpenAttr(openState, `${detailsKeyBase}:live`, session.state === 'running')}>
        <summary>Live Metrics</summary>
        <p class="mono small muted">${detailsSummary}</p>
        ${metricRollupMarkup(liveSummary)}
        ${leaderboardMarkup(liveSummary)}
      </details>

      <details class="detail-block" data-section="signals" ${detailOpenAttr(openState, `${detailsKeyBase}:signals`, isStuck)}>
        <summary>Signals</summary>
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
    const response = await fetch(`status.json?t=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const overallPct = Number(data.overall?.progress_pct || 0);
    const overallBar = document.getElementById('overallBar');
    const overallPctText = document.getElementById('overallPct');
    const overallCounts = document.getElementById('overallCounts');
    const generatedAt = document.getElementById('generatedAt');
    const runTitle = document.getElementById('runTitle');
    const runTag = document.getElementById('runTag');
    const launchedAt = document.getElementById('launchedAt');
    const grid = document.getElementById('sessionGrid');
    const hasPreviousCards = grid.children.length > 0;
    const openState = hasPreviousCards ? captureOpenDetailState() : null;

    overallBar.style.width = `${overallPct}%`;
    overallPctText.textContent = `${overallPct.toFixed(2)}%`;
    overallCounts.textContent = `${formatInt(data.overall?.completed_records || 0)} / ${formatInt(data.overall?.expected_records || 0)} records`;
    generatedAt.textContent = `Last update: ${formatTimestamp(data.generated_at)}`;
    runTitle.textContent = data.run_title || 'ReasonBench Live Sessions';
    runTag.textContent = `Run Tag: ${data.run_tag || '--'}`;
    launchedAt.textContent = `Launched: ${formatTimestamp(data.launched_at)}`;

    const sessions = data.sessions || [];
    grid.innerHTML = sessions.map((session) => sessionCard(session, openState)).join('');
  } catch (error) {
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

refresh();
setInterval(refresh, 10000);
