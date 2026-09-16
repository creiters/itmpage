export class MeshiSignaler {
  /**
   * Generates a local SDP Offer containing all subnet host ICE candidates.
   */
  static async createOffer() {
    const pc = new RTCPeerConnection({ iceServers: [] });
    const dc = pc.createDataChannel('meshi-sync-channel', { ordered: true });
    const candidates = [];

    pc.onicecandidate = (event) => {
      if (event.candidate) candidates.push(event.candidate);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // Wait 600ms to gather local host candidates on subnet/hotspot
    await new Promise((resolve) => setTimeout(resolve, 600));

    const tokenPayload = {
      sdp: pc.localDescription,
      candidates
    };

    return {
      pc,
      dc,
      token: btoa(JSON.stringify(tokenPayload))
    };
  }

  /**
   * Accepts an incoming Offer and generates an Answer token.
   */
  static async acceptOffer(encodedOffer) {
    const { sdp, candidates } = JSON.parse(atob(encodedOffer));
    const pc = new RTCPeerConnection({ iceServers: [] });
    const candidatesOut = [];

    pc.onicecandidate = (event) => {
      if (event.candidate) candidatesOut.push(event.candidate);
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

    const tokenPayload = {
      sdp: pc.localDescription,
      candidates: candidatesOut
    };

    return {
      pc,
      dcPromise,
      token: btoa(JSON.stringify(tokenPayload))
    };
  }

  /**
   * Finalizes the local RTCPeerConnection using the returned Answer token.
   */
  static async finalizeHandshake(pc, encodedAnswer) {
    const { sdp, candidates } = JSON.parse(atob(encodedAnswer));
    await pc.setRemoteDescription(new RTCSessionDescription(sdp));
    for (const cand of candidates) {
      await pc.addIceCandidate(new RTCIceCandidate(cand));
    }
  }
}
