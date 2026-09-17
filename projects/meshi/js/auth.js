/**
 * MeshiAuth: Zero-cloud cryptographic peer identity and challenge-response.
 */
export class MeshiAuth {
  constructor(keyPair, authorizedPublicKeys = new Set()) {
    this.keyPair = keyPair;
    this.authorizedPublicKeys = authorizedPublicKeys; // Set of base64 public keys
  }

  static async generateIdentity() {
    const keyPair = await window.crypto.subtle.generateKey(
      { name: "ECDSA", namedCurve: "P-256" },
      true,
      ["sign", "verify"]
    );
    const rawPub = await window.crypto.subtle.exportKey("spki", keyPair.publicKey);
    const publicKeyBase64 = btoa(String.fromCharCode(...new Uint8Array(rawPub)));
    return { keyPair, publicKeyBase64 };
  }

  async signChallenge(challengeNonce) {
    const encoder = new TextEncoder();
    const data = encoder.encode(challengeNonce);
    const signature = await window.crypto.subtle.sign(
      { name: "ECDSA", hash: { name: "SHA-256" } },
      this.keyPair.privateKey,
      data
    );
    return btoa(String.fromCharCode(...new Uint8Array(signature)));
  }

  async verifyPeer(publicKeyBase64, challengeNonce, signatureBase64) {
    // Check local authorization whitelist
    if (!this.authorizedPublicKeys.has(publicKeyBase64)) {
      return false;
    }

    try {
      const pubBinary = Uint8Array.from(atob(publicKeyBase64), c => c.charCodeAt(0));
      const sigBinary = Uint8Array.from(atob(signatureBase64), c => c.charCodeAt(0));
      const peerKey = await window.crypto.subtle.importKey(
        "spki",
        pubBinary,
        { name: "ECDSA", namedCurve: "P-256" },
        true,
        ["verify"]
      );

      return await window.crypto.subtle.verify(
        { name: "ECDSA", hash: { name: "SHA-256" } },
        peerKey,
        sigBinary,
        new TextEncoder().encode(challengeNonce)
      );
    } catch {
      return false;
    }
  }
}
