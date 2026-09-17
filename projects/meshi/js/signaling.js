export class MeshiSignaler {
  static async createOffer(cryptoInstance) {
    const pc = new RTCPeerConnection({ iceServers: [] });
    const dc = pc.createDataChannel('meshi-p2p-channel', { ordered: true });
    const candidates = [];

    pc.onicecandidate = (e) => {
      if (e.candidate) candidates.push(e.candidate);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await new Promise(r => setTimeout(r, 600));

    const token = await cryptoInstance.generateSignedHandshake({
      sdp: pc.localDescription,
      candidates
    });

    return { pc, dc, token };
  }

  static async acceptOffer(encodedOffer, cryptoInstance) {
    const { payload, publicKey } = await cryptoInstance.constructor.verifyHandshakeToken(encodedOffer);
    const { sdp, candidates } = payload;

    const pc = new RTCPeerConnection({ iceServers: [] });
    const candidatesOut = [];

    pc.onicecandidate = (e) => {
      if (e.candidate) candidatesOut.push(e.candidate);
    };

    const dcPromise = new Promise(resolve => {
      pc.ondatachannel = (e) => resolve(e.channel);
    });

    await pc.setRemoteDescription(new RTCSessionDescription(sdp));
    for (const cand of candidates) {
      await pc.addIceCandidate(new RTCIceCandidate(cand));
    }

    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    await new Promise(r => setTimeout(r, 600));

    const token = await cryptoInstance.generateSignedHandshake({
      sdp: pc.localDescription,
      candidates: candidatesOut
    });

    return { pc, dcPromise, token, remotePublicKey: publicKey };
  }

  static async finalizeHandshake(pc, encodedAnswer, cryptoInstance) {
    const { payload, publicKey } = await cryptoInstance.constructor.verifyHandshakeToken(encodedAnswer);
    const { sdp, candidates } = payload;

    await pc.setRemoteDescription(new RTCSessionDescription(sdp));
    for (const cand of candidates) {
      await pc.addIceCandidate(new RTCIceCandidate(cand));
    }
    return publicKey;
  }
}
