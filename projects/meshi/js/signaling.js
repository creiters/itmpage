export class MeshiSignaler {
  static async createOffer() {
    const pc = new RTCPeerConnection({ iceServers: [] });
    const dc = pc.createDataChannel('meshi-p2p-channel', { ordered: true });
    const candidates = [];

    pc.onicecandidate = (e) => {
      if (e.candidate) candidates.push(e.candidate);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await new Promise((resolve) => setTimeout(resolve, 600));

    return {
      pc,
      dc,
      token: btoa(JSON.stringify({ sdp: pc.localDescription, candidates }))
    };
  }

  static async acceptOffer(encodedOffer) {
    const { sdp, candidates } = JSON.parse(atob(encodedOffer));
    const pc = new RTCPeerConnection({ iceServers: [] });
    const candidatesOut = [];

    pc.onicecandidate = (e) => {
      if (e.candidate) candidatesOut.push(e.candidate);
    };

    const dcPromise = new Promise((resolve) => {
      pc.ondatachannel = (e) => resolve(e.channel);
    });

    await pc.setRemoteDescription(new RTCSessionDescription(sdp));
    for (const cand of candidates) {
      await pc.addIceCandidate(new RTCIceCandidate(cand));
    }

    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    await new Promise((resolve) => setTimeout(resolve, 600));

    return {
      pc,
      dcPromise,
      token: btoa(JSON.stringify({ sdp: pc.localDescription, candidates: candidatesOut }))
    };
  }

  static async finalizeHandshake(pc, encodedAnswer) {
    const { sdp, candidates } = JSON.parse(atob(encodedAnswer));
    await pc.setRemoteDescription(new RTCSessionDescription(sdp));
    for (const cand of candidates) {
      await pc.addIceCandidate(new RTCIceCandidate(cand));
    }
  }
}
