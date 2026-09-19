/**
 * 42pulse.js - Native API Fetcher and Local Data Sync
 */
class IntraApp {
  constructor(token) {
    this.token = token;
    this.baseUrl = 'https://api.intra.42.fr/v2';
  }

  async fetchWithRateLimit(endpoint) {
    const res = await fetch(`${this.baseUrl}${endpoint}`, {
      headers: { Authorization: `Bearer ${this.token}`, Accept: 'application/json' }
    });

    if (res.status === 429) {
      const wait = (parseInt(res.headers.get('Retry-After'), 10) || 2) * 1000;
      await new Promise(r => setTimeout(r, wait));
      return this.fetchWithRateLimit(endpoint);
    }
    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
    return res.json();
  }

  // Fetch online data and store it into IndexedDB
  async syncData(campusId = 51) {
    try {
      // 1. Fetch personal profile
      const profile = await this.fetchWithRateLimit('/me');
      await idbPut('campus_stats', { key: 'profile', data: profile, updatedAt: Date.now() });

      // 2. Fetch cluster presence
      const locations = await this.fetchWithRateLimit(`/campus/${campusId}/locations?filter[active]=true&page[size]=50`);
      
      const timeBuckets = { '< 2h': 0, '2h - 5h': 0, '> 5h': 0 };
      const now = Date.now();
      locations.forEach(loc => {
        const diffHours = (now - new Date(loc.begin_at).getTime()) / 36e5;
        if (diffHours < 2) timeBuckets['< 2h']++;
        else if (diffHours < 5) timeBuckets['2h - 5h']++;
        else timeBuckets['> 5h']++;
      });

      await idbPut('campus_stats', { key: 'visiting_time', data: timeBuckets, updatedAt: Date.now() });

      // 3. Fetch user projects
      const projects = await this.fetchWithRateLimit('/me/projects_users?page[size]=15&sort=-updated_at');
      await idbPut('campus_stats', { key: 'projects', data: projects, updatedAt: Date.now() });

      return { profile, timeBuckets, projects, source: 'network' };
    } catch (err) {
      // Network failure: Fall back to IndexedDB offline store
      console.warn('Network request failed, retrieving cached data from IndexedDB...', err);
      const cached = await this.getOfflineData();
      if (!cached.profile) throw new Error('Offline and no local cache found.');
      cached.source = 'indexeddb';
      return cached;
    }
  }

  async getOfflineData() {
    const profile = await idbGet('campus_stats', 'profile');
    const visitingTime = await idbGet('campus_stats', 'visiting_time');
    const projects = await idbGet('campus_stats', 'projects');

    return {
      profile: profile ? profile.data : null,
      timeBuckets: visitingTime ? visitingTime.data : null,
      projects: projects ? projects.data : null
    };
  }
}
