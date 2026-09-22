/**
 * 42 Prague PeerFinder - 100% Native Client JavaScript
 * No other programming languages used.
 */
(function () {
  'use strict';

  const CAMPUS_PRAGUE_ID = 51;
  const API_BASE = 'https://api.intra.42.fr/v2';
  const CORS_RELAY = 'https://corsproxy.io/?url=';

  // --- 1. IndexedDB Native Storage ---
  const DB_NAME = 'PeerFinderDB';
  const openDB = () => {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains('peers')) {
          db.createObjectStore('peers', { keyPath: 'login' });
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  };

  const savePeersToCache = async (peers) => {
    const db = await openDB();
    const tx = db.transaction('peers', 'readwrite');
    const store = tx.objectStore('peers');
    store.clear();
    peers.forEach((p) => store.put(p));
  };

  const getPeersFromCache = async () => {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction('peers', 'readonly');
      const req = tx.objectStore('peers').getAll();
      req.onsuccess = () => resolve(req.result || []);
      req.onerror = () => resolve([]);
    });
  };

  // --- 2. Dynamic Pure JS UI Construction ---
  const buildInterface = () => {
    // Inject styles dynamically
    const style = document.createElement('style');
    style.textContent = `
      :root {
        --bg: #0d1117; --card: #161b22; --border: #30363d;
        --text: #c9d1d9; --accent: #00babc; --white: #ffffff;
      }
      * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, sans-serif; }
      body { background: var(--bg); color: var(--text); padding: 1.5rem; max-width: 1200px; margin: auto; }
      header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 1rem; }
      h1 { font-size: 1.3rem; color: var(--white); }
      .badge { background: rgba(0, 186, 188, 0.15); color: var(--accent); border: 1px solid var(--accent); padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
      .inputs { display: flex; gap: 0.5rem; }
      input, select, button { background: var(--card); border: 1px solid var(--border); color: #fff; padding: 0.5rem 0.8rem; border-radius: 6px; }
      button { background: var(--accent); color: #000; font-weight: bold; border: none; cursor: pointer; }
      .filters { display: flex; gap: 0.8rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
      .filters input, .filters select { flex: 1; min-width: 180px; }
      .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 1rem; }
      .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; display: flex; gap: 0.8rem; align-items: center; }
      .card img { width: 48px; height: 48px; border-radius: 50%; border: 2px solid var(--border); object-fit: cover; }
      .info { display: flex; flex-direction: column; gap: 0.2rem; overflow: hidden; }
      .info .u-name { font-weight: bold; color: var(--white); font-size: 0.95rem; }
      .info .u-host { font-family: monospace; font-size: 0.8rem; color: var(--accent); }
      .info .u-proj { font-size: 0.75rem; color: #58a6ff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .info .u-time { font-size: 0.75rem; color: #8b949e; }
      .msg { text-align: center; padding: 3rem 0; color: #8b949e; width: 100%; grid-column: 1 / -1; }
    `;
    document.head.appendChild(style);

    // Create App DOM
    document.body.innerHTML = `
      <header>
        <div>
          <h1>42 Prague <span class="badge">PeerFinder</span></h1>
          <span id="txt-count" style="font-size: 0.85rem; color: #8b949e;">Ready</span>
        </div>
        <div class="inputs">
          <input type="password" id="input-token" placeholder="Bearer Token">
          <button id="btn-sync">Fetch Cluster</button>
        </div>
      </header>

      <div class="filters">
        <input type="text" id="filter-user" placeholder="Filter by user...">
        <select id="filter-project">
          <option value="ALL">All Projects</option>
          <option value="libft">Libft</option>
          <option value="ft_printf">ft_printf</option>
          <option value="get_next_line">get_next_line</option>
          <option value="push_swap">push_swap</option>
          <option value="minishell">minishell</option>
          <option value="philosophers">philosophers</option>
        </select>
        <select id="filter-cluster">
          <option value="ALL">All Workstations</option>
          <option value="c1">Cluster 1 (c1)</option>
          <option value="c2">Cluster 2 (c2)</option>
        </select>
      </div>

      <div class="grid" id="grid">
        <div class="msg">Provide an access token and click "Fetch Cluster".</div>
      </div>
    `;
  };

  // --- 3. Rate-Limited API Dispatcher ---
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const intraFetch = async (endpoint, token) => {
    const rawTarget = `${API_BASE}${endpoint}`;
    const url = `${CORS_RELAY}${encodeURIComponent(rawTarget)}`;

    const res = await fetch(url, {
      method: 'GET',
      headers: {
        Authorization: `Bearer ${token}`,
        Accept: 'application/json',
      },
    });

    if (res.status === 429) {
      const wait = (parseInt(res.headers.get('Retry-After'), 10) || 2) * 1000;
      await sleep(wait);
      return intraFetch(endpoint, token);
    }

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  };

  // --- 4. Logic & State Orchestrator ---
  let peers = [];

  const render = () => {
    const grid = document.getElementById('grid');
    const qUser = document.getElementById('filter-user').value.toLowerCase().trim();
    const qProj = document.getElementById('filter-project').value.toLowerCase();
    const qClust = document.getElementById('filter-cluster').value.toLowerCase();

    const matches = peers.filter((p) => {
      const mUser = p.login.toLowerCase().includes(qUser);
      const mProj = qProj === 'all' || p.project.toLowerCase().includes(qProj);
      const mClust = qClust === 'all' || p.host.toLowerCase().startsWith(qClust);
      return mUser && mProj && mClust;
    });

    document.getElementById('txt-count').textContent = `${matches.length} peers shown`;

    if (matches.length === 0) {
      grid.innerHTML = '<div class="msg">No active peers match filters.</div>';
      return;
    }

    grid.innerHTML = matches
      .map(
        (p) => `
        <div class="card">
          <img src="${p.avatar}" alt="${p.login}" onerror="this.src='https://profile.intra.42.fr/assets/42_logo-7dfc9110a5319a308863b96bda33cea2b2a03eed6a553d37807739867b43d1c6.png'">
          <div class="info">
            <span class="u-name">${p.login}</span>
            <span class="u-host">${p.host}</span>
            <span class="u-proj">🔨 ${p.project}</span>
            <span class="u-time">⏱ ${p.duration}</span>
          </div>
        </div>
      `
      )
      .join('');
  };

  const runSync = async () => {
    const tokenInput = document.getElementById('input-token');
    const token = tokenInput.value.trim();
    if (!token) return alert('Token required');

    localStorage.setItem('intra_token', token);
    const grid = document.getElementById('grid');
    grid.innerHTML = '<div class="msg">Connecting to 42 API...</div>';

    try {
      const locations = await intraFetch(
        `/campus/${CAMPUS_PRAGUE_ID}/locations?filter[active]=true&page[size]=50`,
        token
      );

      peers = [];
      const now = Date.now();

      for (const loc of locations.slice(0, 20)) {
        if (!loc.user) continue;

        let projName = 'Common Core';
        try {
          await sleep(520); // 2 requests/sec limit
          const projs = await intraFetch(
            `/users/${loc.user.login}/projects_users?filter[status]=in_progress`,
            token
          );
          if (projs.length > 0) projName = projs[0].project.name;
        } catch (_) {}

        const hours = (now - new Date(loc.begin_at).getTime()) / 36e5;
        const duration = hours < 1 ? `${Math.round(hours * 60)}m` : `${hours.toFixed(1)}h`;

        peers.push({
          login: loc.user.login,
          host: loc.host,
          project: projName,
          duration: duration,
          avatar: loc.user.image?.versions?.small || loc.user.image?.link || '',
        });

        render();
      }

      await savePeersToCache(peers);
    } catch (err) {
      grid.innerHTML = `<div class="msg" style="color:#da3633;">Sync error: ${err.message}. Loading cache...</div>`;
      peers = await getPeersFromCache();
      render();
    }
  };

  // --- 5. Init ---
  document.addEventListener('DOMContentLoaded', async () => {
    buildInterface();

    const tokenInput = document.getElementById('input-token');
    tokenInput.value = localStorage.getItem('intra_token') || '';

    document.getElementById('btn-sync').addEventListener('click', runSync);
    document.getElementById('filter-user').addEventListener('input', render);
    document.getElementById('filter-project').addEventListener('change', render);
    document.getElementById('filter-cluster').addEventListener('change', render);

    // Initial load from IndexedDB cache if available
    peers = await getPeersFromCache();
    if (peers.length > 0) render();

    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('./sw.js').catch(() => {});
    }
  });
})();
