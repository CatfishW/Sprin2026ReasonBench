let catalog = [];

function esc(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function parseYear(raw) {
  const m = String(raw || '').match(/(19|20)\d{2}/);
  return m ? Number(m[0]) : null;
}

function populateSelect(select, values, allLabel) {
  select.innerHTML = `<option value="all">${esc(allLabel)}</option>` +
    values.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
}

function rowMarkup(row) {
  const status = row.download_status || 'unknown';
  return `
    <tr>
      <td>${esc(row.dataset || '')}</td>
      <td>${esc(row.title || '')}</td>
      <td>${esc(row.year || '')}</td>
      <td>${esc(row.venue_or_source || '')}</td>
      <td>${esc(row.model_or_system || '')}</td>
      <td>${esc(row.reported_results || '')}</td>
      <td><span class="status-pill ${esc(status)}">${esc(status)}</span></td>
      <td>
        <a href="${esc(row.source_url || '#')}" target="_blank" rel="noopener">source</a>
        ${row.normalized_pdf_url ? ` | <a href="${esc(row.normalized_pdf_url)}" target="_blank" rel="noopener">pdf</a>` : ''}
      </td>
    </tr>
  `;
}

function applyFilters() {
  const datasetVal = document.getElementById('datasetFilter').value;
  const yearVal = document.getElementById('yearFilter').value;
  const statusVal = document.getElementById('statusFilter').value;
  const query = document.getElementById('searchInput').value.trim().toLowerCase();
  const downloadedOnly = document.getElementById('downloadedOnly').checked;

  let rows = catalog.slice();

  if (datasetVal !== 'all') {
    rows = rows.filter((r) => (r.dataset || '') === datasetVal);
  }
  if (yearVal !== 'all') {
    rows = rows.filter((r) => String(parseYear(r.year) || '') === yearVal);
  }
  if (statusVal !== 'all') {
    rows = rows.filter((r) => (r.download_status || '') === statusVal);
  }
  if (downloadedOnly) {
    rows = rows.filter((r) => ['downloaded', 'downloaded_cached'].includes(r.download_status || ''));
  }
  if (query) {
    rows = rows.filter((r) => {
      const blob = [r.title, r.model_or_system, r.reported_results, r.venue_or_source].join(' ').toLowerCase();
      return blob.includes(query);
    });
  }

  rows.sort((a, b) => (parseYear(b.year) || 0) - (parseYear(a.year) || 0));

  const body = document.getElementById('tableBody');
  body.innerHTML = rows.length
    ? rows.map(rowMarkup).join('')
    : '<tr><td colspan="8">No rows match current filters.</td></tr>';

  document.getElementById('metaShown').textContent = `Shown: ${rows.length}`;
}

function bindEvents() {
  ['datasetFilter', 'yearFilter', 'statusFilter', 'searchInput', 'downloadedOnly'].forEach((id) => {
    document.getElementById(id).addEventListener('input', applyFilters);
    document.getElementById(id).addEventListener('change', applyFilters);
  });

  document.getElementById('resetBtn').addEventListener('click', () => {
    document.getElementById('datasetFilter').value = 'all';
    document.getElementById('yearFilter').value = 'all';
    document.getElementById('statusFilter').value = 'all';
    document.getElementById('searchInput').value = '';
    document.getElementById('downloadedOnly').checked = false;
    applyFilters();
  });
}

async function init() {
  try {
    const res = await fetch('../papers_catalog.json');
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    catalog = await res.json();

    const datasets = Array.from(new Set(catalog.map((r) => r.dataset).filter(Boolean))).sort();
    const years = Array.from(new Set(catalog.map((r) => parseYear(r.year)).filter(Boolean))).sort((a, b) => b - a);
    const statuses = Array.from(new Set(catalog.map((r) => r.download_status).filter(Boolean))).sort();

    populateSelect(document.getElementById('datasetFilter'), datasets, 'All datasets');
    populateSelect(document.getElementById('yearFilter'), years.map(String), 'All years');
    populateSelect(document.getElementById('statusFilter'), statuses, 'All statuses');

    document.getElementById('metaTotal').textContent = `Total: ${catalog.length}`;
    bindEvents();
    applyFilters();
  } catch (err) {
    document.getElementById('tableBody').innerHTML =
      `<tr><td colspan="8">Failed to load catalog: ${esc(err.message || err)}</td></tr>`;
  }
}

init();
