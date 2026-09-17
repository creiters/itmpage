export class MeshiCrypto {
  constructor(keyPair, publicKeyBase64) {
    this.keyPair = keyPair;
    this.publicKeyBase64 = publicKeyBase64;
  }

  static async generateIdentity() {
    const keyPair = await window.crypto.subtle.generateKey(
      { name: "ECDSA", namedCurve: "P-256" },
      false, // non-extractable private key for memory safety
      ["sign", "verify"]
    );
    const rawPub = await window.crypto.subtle.exportKey("spki", keyPair.publicKey);
    const publicKeyBase64 = btoa(String.fromCharCode(...new Uint8Array(rawPub)));
    return new MeshiCrypto(keyPair, publicKeyBase64);
  }

  async signChallenge(challengeNonce) {
    const encoder = new TextEncoder();
    const sigBuffer = await window.crypto.subtle.sign(
      { name: "ECDSA", hash: { name: "SHA-256" } },
      this.keyPair.privateKey,
      encoder.encode(challengeNonce)
    );
    return btoa(String.fromCharCode(...new Uint8Array(sigBuffer)));
  }

  static async verifyPeer(publicKeyBase64, challengeNonce, signatureBase64) {
    try {
      const pubBinary = Uint8Array.from(atob(publicKeyBase64), (c) => c.charCodeAt(0));
      const sigBinary = Uint8Array.from(atob(signatureBase64), (c) => c.charCodeAt(0));

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
