/**
 * mesh.js
 * Serverless WebRTC DataChannel Mesh with 6-Digit PIN Exchange
 */

export class MeshCoordinator {
  constructor(nodeId, onMessage, onPeerChange) {
    this.nodeId = nodeId;
    this.onMessage = onMessage;
    this.onPeerChange = onPeerChange;
    this.peers = new Map();
    this.localPin = null;
  }

  createPin() {
    this.localPin = Math.floor(100000 + Math.random() * 900000).toString();
    return this.localPin;
  }

  static encodePayload(obj) {
    return btoa(unescape(encodeURIComponent(JSON.stringify(obj))));
  }

  static decodePayload(base64) {
    return JSON.parse(decodeURIComponent(escape(atob(base64))));
  }

  async generateHostToken(pin) {
    this.localPin = pin;
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    const channel = pc.createDataChannel('clean-jsonllm-mesh');
    this._bindEvents(channel, `peer_${pin}`, pc);

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    await this._waitForIce(pc);
    return MeshCoordinator.encodePayload({ pin, sdp: pc.localDescription });
  }

  async acceptTokenAndCreateAnswer(hostToken) {
    const payload = MeshCoordinator.decodePayload(hostToken);
    const pc = new RTCPeerConnection({
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    });

    pc.ondatachannel = (e) => {
      this._bindEvents(e.channel, `peer_${payload.pin}`, pc);
    };

    await pc.setRemoteDescription(new RTCSessionDescription(payload.sdp));
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    await this._waitForIce(pc);
    return MeshCoordinator.encodePayload({ pin: payload.pin, sdp: pc.localDescription });
  }

  async finalizeHostConnection(answerToken) {
    const payload = MeshCoordinator.decodePayload(answerToken);
    const peerRec = Array.from(this.peers.values())[0];
    if (peerRec && peerRec.pc) {
      await peerRec.pc.setRemoteDescription(new RTCSessionDescription(payload.sdp));
    }
  }

  _bindEvents(channel, key, pc) {
    channel.onopen = () => {
      this.peers.set(key, { channel, pc, key });
      if (this.onPeerChange) this.onPeerChange(this.peers.size);
    };
    channel.onclose = () => {
      this.peers.delete(key);
      if (this.onPeerChange) this.onPeerChange(this.peers.size);
    };
    channel.onmessage = (e) => {
      if (this.onMessage) this.onMessage(JSON.parse(e.data), key, channel);
    };
  }

  _waitForIce(pc) {
    return new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') resolve();
      else {
        const check = () => {
          if (pc.iceGatheringState === 'complete') {
            pc.removeEventListener('icegatheringstatechange', check);
            resolve();
          }
        };
        pc.addEventListener('icegatheringstatechange', check);
        setTimeout(resolve, 1500);
      }
    });
  }

  dispatchCompute(prompt) {
    const list = Array.from(this.peers.values());
    if (!list.length) return null;
    const target = list[Math.floor(Math.random() * list.length)];
    const taskId = 'task_' + Date.now();
    target.channel.send(JSON.stringify({ type: 'MESH_INFER', prompt, taskId }));
    return { peerKey: target.key, taskId };
  }
}
