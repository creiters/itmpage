/**
 * MeshiCrypto: Persistent Web Crypto API Engine
 * Handles ECDSA P-256 signing, verification, and session token encryption.
 */
export class MeshiCrypto {
  constructor(keyPair, publicKeyBase64) {
    this.keyPair = keyPair;
    this.publicKeyBase64 = publicKeyBase64;
  }

  static async init(db) {
    let savedKeys = await db.getSystemKey('mesh_identity');
    let keyPair;
    let publicKeyBase64;

    if (!savedKeys) {
      keyPair = await window.crypto.subtle.generateKey(
        { name: "ECDSA", namedCurve: "P-256" },
        true,
        ["sign", "verify"]
      );

      const exportedPub = await window.crypto.subtle.exportKey("spki", keyPair.publicKey);
      const exportedPriv = await window.crypto.subtle.exportKey("pkcs8", keyPair.privateKey);

      publicKeyBase64 = btoa(String.fromCharCode(...new Uint8Array(exportedPub)));
      const privateKeyBase64 = btoa(String.fromCharCode(...new Uint8Array(exportedPriv)));

      await db.setSystemKey('mesh_identity', { publicKeyBase64, privateKeyBase64 });
    } else {
      publicKeyBase64 = savedKeys.publicKeyBase64;
      const pubBuf = Uint8Array.from(atob(savedKeys.publicKeyBase64), c => c.charCodeAt(0));
      const privBuf = Uint8Array.from(atob(savedKeys.privateKeyBase64), c => c.charCodeAt(0));

      const publicKey = await window.crypto.subtle.importKey(
        "spki",
        pubBuf,
        { name: "ECDSA", namedCurve: "P-256" },
        true,
        ["verify"]
      );
      const privateKey = await window.crypto.subtle.importKey(
        "pkcs8",
        privBuf,
        { name: "ECDSA", namedCurve: "P-256" },
        true,
        ["sign"]
      );
      keyPair = { publicKey, privateKey };
    }

    return new MeshiCrypto(keyPair, publicKeyBase64);
  }

  async sign(dataString) {
    const encoder = new TextEncoder();
    const signature = await window.crypto.subtle.sign(
      { name: "ECDSA", hash: { name: "SHA-256" } },
      this.keyPair.privateKey,
      encoder.encode(dataString)
    );
    return btoa(String.fromCharCode(...new Uint8Array(signature)));
  }

  static async verify(publicKeyBase64, dataString, signatureBase64) {
    try {
      const pubBuf = Uint8Array.from(atob(publicKeyBase64), c => c.charCodeAt(0));
      const sigBuf = Uint8Array.from(atob(signatureBase64), c => c.charCodeAt(0));

      const key = await window.crypto.subtle.importKey(
        "spki",
        pubBuf,
        { name: "ECDSA", namedCurve: "P-256" },
        true,
        ["verify"]
      );

      return await window.crypto.subtle.verify(
        { name: "ECDSA", hash: { name: "SHA-256" } },
        key,
        sigBuf,
        new TextEncoder().encode(dataString)
      );
    } catch {
      return false;
    }
  }

  async generateSignedHandshake(payload) {
    const payloadStr = JSON.stringify(payload);
    const signature = await this.sign(payloadStr);
    return btoa(JSON.stringify({
      payload: payloadStr,
      signature,
      publicKey: this.publicKeyBase64,
      timestamp: Date.now()
    }));
  }

  static async verifyHandshakeToken(token) {
    try {
      const decoded = JSON.parse(atob(token));
      const { payload, signature, publicKey } = decoded;
      const valid = await MeshiCrypto.verify(publicKey, payload, signature);
      if (!valid) throw new Error("Invalid cryptographic signature");
      return { payload: JSON.parse(payload), publicKey };
    } catch (err) {
      throw new Error(`Token verification failed: ${err.message}`);
    }
  }
}
