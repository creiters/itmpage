/**
 * app.js - Application Controller
 */
window.addEventListener('DOMContentLoaded', async () => {
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('./sw.js').catch(console.error);
  }

  const currentUri = `${window.location.origin}${window.location.pathname}`;
  document.getElementById('display-uri').textContent = currentUri;

  const clientIdInput = document.getElementById('client-id-input');
  const btnLogin = document.getElementById('btn-login');
  const btnLogout = document.getElementById('btn-logout');
  const setupView = document.getElementById('setup-view');
  const dashboardView = document.getElementById('dashboard-view');

  clientIdInput.value = localStorage.getItem('intra_saved_uid') || '';

  // 1. Check if returning from OAuth redirection
  try {
    if (window.location.search.includes('code=')) {
      document.body.style.opacity = '0.5';
      await IntraAuth.handleCallback();
      document.body.style.opacity = '1';
    }
  } catch (err) {
    alert(`Authentication failed: ${err.message}`);
    document.body.style.opacity = '1';
  }

  // 2. Start Login Flow
  btnLogin.addEventListener('click', () => {
    const uid = clientIdInput.value.trim();
    if (!uid) {
      alert('Please enter your Application UID from 42 Intra Settings.');
      return;
    }
    localStorage.setItem('intra_saved_uid', uid);
    IntraAuth.startLogin(uid, currentUri);
  });

  // 3. Logout
  btnLogout.addEventListener('click', () => {
    IntraAuth.clearToken();
    window.location.reload();
  });

  // 4. Render Profile if Authenticated
  const token = IntraAuth.getToken();
  if (token) {
    setupView.classList.add('hidden');
    dashboardView.classList.remove('hidden');
    btnLogout.classList.remove('hidden');

    const client = new FortyTwoAPI(token);

    try {
      // Execute calls in parallel using ES8
      const [profile, projects, locations] = await Promise.all([
        client.getMyProfile(),
        client.getMyProjects(),
        client.getMyLocations(8),
      ]);

      renderProfile(profile);
      renderProjects(projects);
      renderLocations(locations);
    } catch (err) {
      console.error(err);
      if (err.message.includes('401')) {
        IntraAuth.clearToken();
        alert('Session expired. Please log in again.');
        window.location.reload();
      }
    }
  }

  function renderProfile(user) {
    document.getElementById('u-avatar').src = user.image?.versions?.medium || user.image?.link || '';
    document.getElementById('u-name').textContent = user.usual_full_name || user.displayname;
    document.getElementById('u-login').textContent = `@${user.login} • ${user.email} • Campus: ${user.campus[0]?.name || 'N/A'}`;

    const badgeBar = document.getElementById('u-badges');
    badgeBar.innerHTML = '';

    const cursus42 = user.cursus_users.find((c) => c.cursus.name.includes('42'));
    if (cursus42) {
      badgeBar.innerHTML += `<span class="badge">Level ${cursus42.level.toFixed(2)}</span>`;
    }
    badgeBar.innerHTML += `<span class="badge">Wallet: ${user.wallet} ₳</span>`;
    badgeBar.innerHTML += `<span class="badge">Correction Points: ${user.correction_point}</span>`;
  }

  function renderProjects(projects) {
    const list = document.getElementById('project-list');
    list.innerHTML = '';

    if (!projects.length) {
      list.innerHTML = '<li class="stat-item">No project records found.</li>';
      return;
    }

    projects.slice(0, 15).forEach((p) => {
      const li = document.createElement('li');
      li.className = 'stat-item';
      
      let statusClass = 'status-progress';
      let statusLabel = `${p.status}`;

      if (p['validated?'] === true) {
        statusClass = 'status-ok';
        statusLabel = `✓ ${p.final_mark}`;
      } else if (p['validated?'] === false) {
        statusClass = 'status-fail';
        statusLabel = `✗ ${p.final_mark}`;
      }

      li.innerHTML = `
        <span>${p.project.name}</span>
        <span class="${statusClass}">${statusLabel}</span>
      `;
      list.appendChild(li);
    });
  }

  function renderLocations(locations) {
    const list = document.getElementById('location-list');
    list.innerHTML = '';

    if (!locations.length) {
      list.innerHTML = '<li class="stat-item">No recorded workstation logins.</li>';
      return;
    }

    locations.forEach((loc) => {
      const start = new Date(loc.begin_at).toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
      const duration = loc.end_at
        ? `${((new Date(loc.end_at) - new Date(loc.begin_at)) / 36e5).toFixed(1)} hrs`
        : 'Active now';

      const li = document.createElement('li');
      li.className = 'stat-item';
      li.innerHTML = `
        <span><strong>${loc.host}</strong> (${start})</span>
        <span style="color: #8b949e;">${duration}</span>
      `;
      list.appendChild(li);
    });
  }
});
