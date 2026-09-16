/**
 * MeshiSignaler: Air-gapped / Local-subnet SDP and Candidate Exchanger.
 */
export class MeshiSignaler {
  static async createOffer() {
    const pc = new RTCPeerConnection({ iceServers: [] });
    const dc = pc.createDataChannel('meshi-datachannel', { ordered: true });
    const candidates = [];

    pc.onicecandidate = (e) => {
      if (e.candidate) candidates.push(e.candidate);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    // Allow 600ms to accumulate local host subnet candidates
    await new Promise((r) => setTimeout(r, 600));

    const token = btoa(JSON.stringify({ sdp: pc.localDescription, candidates }));
    return { pc, dc, token };
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

    await new Promise((r) => setTimeout(r, 600));

    const answerToken = btoa(JSON.stringify({ sdp: pc.localDescription, candidates: candidatesOut }));
    return { pc, answerToken, dcPromise };
  }

  static async finalizeConnection(pc, encodedAnswer) {
    const { sdp, candidates } = JSON.parse(atob(encodedAnswer));
    await pc.setRemoteDescription(new RTCSessionDescription(sdp));
    for (const cand of candidates) {
      await pc.addIceCandidate(new RTCIceCandidate(cand));
    }
  }
}

/**
 * MeshiMeshSocket: Drop-in WebSocket interface backed by RTCDataChannel.
 */
export class MeshiMeshSocket extends EventTarget {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(peerId, dataChannel) {
    super();
    this.peerId = peerId;
    this.dc = dataChannel;
    this.readyState = MeshiMeshSocket.CONNECTING;

    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;

    this._bindChannel();
  }

  _bindChannel() {
    if (this.dc.readyState === 'open') {
      this.readyState = MeshiMeshSocket.OPEN;
    }

    this.dc.onopen = () => {
      this.readyState = MeshiMeshSocket.OPEN;
      const ev = new Event('open');
      if (this.onopen) this.onopen(ev);
      this.dispatchEvent(ev);
    };

    this.dc.onmessage = (event) => {
      const msgEvent = new MessageEvent('message', { data: event.data });
      if (this.onmessage) this.onmessage(msgEvent);
      this.dispatchEvent(msgEvent);
    };

    this.dc.onclose = () => {
      this.readyState = MeshiMeshSocket.CLOSED;
      const ev = new CloseEvent('close', { wasClean: true });
      if (this.onclose) this.onclose(ev);
      this.dispatchEvent(ev);
    };

    this.dc.onerror = (error) => {
      const ev = new ErrorEvent('error', { error });
      if (this.onerror) this.onerror(ev);
      this.dispatchEvent(ev);
    };
  }

  send(data) {
    if (this.readyState !== MeshiMeshSocket.OPEN) {
      throw new DOMException('DataChannel is not open', 'InvalidStateError');
    }
    const payload = typeof data === 'object' && !(data instanceof ArrayBuffer)
      ? JSON.stringify(data)
      : data;
    this.dc.send(payload);
  }

  close() {
    this.readyState = MeshiMeshSocket.CLOSING;
    this.dc.close();
  }
}

