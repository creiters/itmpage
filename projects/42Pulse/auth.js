/**
 * auth.js - Pure ES8 OAuth 2.0 PKCE client for 42 Intranet API
 */
const IntraAuth = (() => {
  const OAUTH_AUTH_URL = 'https://api.intra.42.fr/oauth/authorize';
  const OAUTH_TOKEN_URL = 'https://api.intra.42.fr/oauth/token';

  const base64UrlEncode = (buffer) => {
    return btoa(String.fromCharCode(...new Uint8Array(buffer)))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  };

  const generateRandomString = (length = 64) => {
    const charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~';
    const randomVals = new Uint8Array(length);
    window.crypto.getRandomValues(randomVals);
    return Array.from(randomVals, (val) => charset[val % charset.length]).join('');
  };

  const generateCodeChallenge = async (verifier) => {
    const encoder = new TextEncoder();
    const data = encoder.encode(verifier);
    const hash = await window.crypto.subtle.digest('SHA-256', data);
    return base64UrlEncode(hash);
  };

  return {
    getToken: () => localStorage.getItem('intra_user_token'),

    saveToken: (tokenData) => {
      localStorage.setItem('intra_user_token', tokenData.access_token);
      if (tokenData.refresh_token) {
        localStorage.setItem('intra_refresh_token', tokenData.refresh_token);
      }
    },

    clearToken: () => {
      localStorage.removeItem('intra_user_token');
      localStorage.removeItem('intra_refresh_token');
      localStorage.removeItem('intra_code_verifier');
      localStorage.removeItem('intra_client_id');
      localStorage.removeItem('intra_redirect_uri');
    },

    // 1. Persist PKCE credentials in localStorage (survives origin/tab resets)
    async startLogin(clientId, redirectUri) {
      const verifier = generateRandomString(96);
      
      // Use localStorage instead of sessionStorage
      localStorage.setItem('intra_code_verifier', verifier);
      localStorage.setItem('intra_client_id', clientId);
      localStorage.setItem('intra_redirect_uri', redirectUri);

      const challenge = await generateCodeChallenge(verifier);
      const params = new URLSearchParams({
        client_id: clientId,
        redirect_uri: redirectUri,
        response_type: 'code',
        scope: 'public',
        code_challenge: challenge,
        code_challenge_method: 'S256',
      });

      window.location.href = `${OAUTH_AUTH_URL}?${params.toString()}`;
    },

    // 2. Intercept callback code using localStorage
    async handleCallback() {
      const urlParams = new URLSearchParams(window.location.search);
      const code = urlParams.get('code');
      if (!code) return null;

      // Retrieve state from localStorage
      const verifier = localStorage.getItem('intra_code_verifier');
      const clientId = localStorage.getItem('intra_client_id');
      const redirectUri = localStorage.getItem('intra_redirect_uri');

      if (!verifier || !clientId || !redirectUri) {
        throw new Error(
          `PKCE state lost: verifier=${!!verifier}, client_id=${!!clientId}, redirect_uri=${!!redirectUri}`
        );
      }

      const body = new URLSearchParams({
        grant_type: 'authorization_code',
        client_id: clientId,
        code: code,
        redirect_uri: redirectUri,
        code_verifier: verifier,
      });

      const response = await fetch(OAUTH_TOKEN_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: body.toString(),
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Token exchange failed (${response.status}): ${err}`);
      }

      const tokenData = await response.json();
      this.saveToken(tokenData);

      // Clean up verifier state after successful exchange
      localStorage.removeItem('intra_code_verifier');
      localStorage.removeItem('intra_redirect_uri');

      // Strip ?code= parameter from URL without reloading
      window.history.replaceState({}, document.title, window.location.pathname);
      return tokenData.access_token;
    },
  };
})();
