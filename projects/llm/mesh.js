export class MeshCoordinator {
  constructor(localNodeId, onMessage, onPeersChanged) {
    this.localNodeId = localNodeId;
    this.onMessage = onMessage;
    this.onPeersChanged = onPeersChanged;
    this.peers = new Map(); // pin -> { pc, channel, role }
    this.pendingHosts = new Map();
  }

  static encodePayload(obj) {
    return btoa(unescape(encodeURIComponent(JSON.stringify(obj))));
  }

  static decodePayload(str) {
    return JSON.parse(decodeURIComponent(escape(atob(str))));
  }

  createPin() {
    return Math.floor(100000 + Math.random() * 900000).toString();
  }

  async generateHostToken(pin) {
    const pc = new RTCPeerConnection({ iceServers: [] });
    const channel = pc.createDataChannel('shared-llm-channel');
    channel.binaryType = 'arraybuffer';

    this.bindChannel(channel, `peer-${pin}`);

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    await new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') resolve();
      else {
        pc.onicecandidate = (e) => {
          if (!e.candidate) resolve();
        };
      }
    });

    const envelope = {
      pin,
      sender: this.localNodeId,
      sdp: pc.localDescription
    };

    this.pendingHosts.set(pin, { pc, channel });
    return MeshCoordinator.encodePayload(envelope);
  }

  async acceptTokenAndCreateAnswer(hostToken) {
    const envelope = MeshCoordinator.decodePayload(hostToken);
    const pc = new RTCPeerConnection({ iceServers: [] });

    pc.ondatachannel = (e) => {
      e.channel.binaryType = 'arraybuffer';
      this.bindChannel(e.channel, `host-${envelope.pin}`);
    };

    await pc.setRemoteDescription(new RTCSessionDescription(envelope.sdp));
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);

    await new Promise((resolve) => {
      if (pc.iceGatheringState === 'complete') resolve();
      else {
        pc.onicecandidate = (e) => {
          if (!e.candidate) resolve();
        };
      }
    });

    return MeshCoordinator.encodePayload({
      pin: envelope.pin,
      sender: this.localNodeId,
      sdp: pc.localDescription
    });
  }

  async finalizeHostConnection(answerToken) {
    const envelope = MeshCoordinator.decodePayload(answerToken);
    const pending = this.pendingHosts.get(envelope.pin);
    if (!pending) throw new Error('No pending offer found matching this PIN.');

    await pending.pc.setRemoteDescription(new RTCSessionDescription(envelope.sdp));
    this.peers.set(envelope.pin, pending);
    this.onPeersChanged(this.peers.size);
  }

  bindChannel(channel, peerKey) {
    channel.onopen = () => this.onPeersChanged(this.peers.size + 1);
    channel.onclose = () => {
      this.peers.delete(peerKey);
      this.onPeersChanged(this.peers.size);
    };
    channel.onmessage = (e) => {
      if (typeof e.data === 'string') {
        const parsed = JSON.parse(e.data);
        if (this.onMessage) this.onMessage(parsed, peerKey, channel);
      } else {
        if (this.onMessage) {
          this.onMessage({ type: 'BINARY_PAYLOAD', buffer: e.data }, peerKey, channel);
        }
      }
    };
  }

  distributeChunk(channel, metadata, arrayBuffer) {
    if (channel.readyState !== 'open') return;
    channel.send(JSON.stringify({ type: 'CHUNK_HEADER', metadata }));
    channel.send(arrayBuffer);
  }

  dispatchCompute(prompt) {
    const active = Array.from(this.peers.entries()).filter(
      ([, item]) => item.channel.readyState === 'open'
    );
    if (active.length === 0) return null;

    const taskId = 'task_' + Math.random().toString(36).substring(2, 9);
    const [targetKey, target] = active[0];

    target.channel.send(JSON.stringify({
      type: 'INFERENCE_REQUEST',
      prompt,
      taskId,
      from: this.localNodeId
    }));

    return { taskId, peerKey: targetKey };
  }

  returnComputeResult(channel, taskId, text, isDone) {
    if (channel.readyState !== 'open') return;
    channel.send(JSON.stringify({
      type: 'INFERENCE_RESPONSE',
      taskId,
      text,
      isDone
    }));
  }
}
