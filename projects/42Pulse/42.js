/**
 * 42.js - ES8 Framework for 42 Intranet API Personal Endpoints
 */
class FortyTwoAPI {
  constructor(token) {
    this.baseUrl = 'https://api.intra.42.fr/v2';
    this.token = token;
  }

  async request(endpoint) {
    const res = await fetch(`${this.baseUrl}${endpoint}`, {
      headers: {
        Authorization: `Bearer ${this.token}`,
        Accept: 'application/json',
      },
    });

    if (res.status === 429) {
      const wait = (parseInt(res.headers.get('Retry-After'), 10) || 2) * 1000;
      await new Promise((r) => setTimeout(r, wait));
      return this.request(endpoint);
    }

    if (!res.ok) {
      throw new Error(`API ${res.status}: ${res.statusText}`);
    }
    return res.json();
  }

  // 1. Get Logged-in Personal Profile
  async getMyProfile() {
    return this.request('/me');
  }

  // 2. Get All User Projects
  async getMyProjects() {
    return this.request('/me/projects_users?page[size]=100&sort=-updated_at');
  }

  // 3. Get Recent Cluster Presence & Logtime
  async getMyLocations(limit = 10) {
    return this.request(`/me/locations?page[size]=${limit}&sort=-begin_at`);
  }
}
