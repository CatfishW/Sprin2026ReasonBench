function esc(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function sessionCard(session) {
  const logText = (session.last_log_lines || []).join('\n') || 'No logs yet.';
  const pct = Number(session.progress_pct || 0);

  return `
    <article class="session-card card">
      <div class="session-head">
        <h2 class="session-name">${esc(session.name)}</h2>
        <span class="badge ${esc(session.state)}">${esc(session.state.replaceAll('_', ' '))}</span>
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
          <strong>${esc(session.expected_records)}</strong>
        </div>
      </div>
      <div class="progress-track"><div class="progress-bar" style="width:${pct}%;"></div></div>
      <p class="path mono">checkpoint: ${esc(session.checkpoint_path)}</p>
      <p class="path mono">log: ${esc(session.latest_log || 'N/A')}</p>
      <pre class="log-snippet mono">${esc(logText)}</pre>
    </article>
  `;
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
    const grid = document.getElementById('sessionGrid');

    overallBar.style.width = `${overallPct}%`;
    overallPctText.textContent = `${overallPct.toFixed(2)}%`;
    overallCounts.textContent = `${data.overall?.completed_records || 0} / ${data.overall?.expected_records || 0} records`;
    generatedAt.textContent = `Last update: ${data.generated_at || '--'}`;

    const sessions = data.sessions || [];
    grid.innerHTML = sessions.map(sessionCard).join('');
  } catch (error) {
    document.getElementById('sessionGrid').innerHTML = `
      <article class="session-card card">
        <h2 class="session-name">Status source unavailable</h2>
        <p class="mono">${esc(error.message || error)}</p>
      </article>
    `;
  }
}

refresh();
setInterval(refresh, 10000);
