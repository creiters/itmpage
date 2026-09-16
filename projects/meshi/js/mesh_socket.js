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

    this._bindEvents();
  }

  _bindEvents() {
    if (this.dc.readyState === 'open') {
      this.readyState = MeshiMeshSocket.OPEN;
    }

    this.dc.onopen = () => {
      this.readyState = MeshiMeshSocket.OPEN;
      const ev = new Event('open');
      if (typeof this.onopen === 'function') this.onopen(ev);
      this.dispatchEvent(ev);
    };

    this.dc.onmessage = (event) => {
      const msgEv = new MessageEvent('message', { data: event.data });
      if (typeof this.onmessage === 'function') this.onmessage(msgEv);
      this.dispatchEvent(msgEv);
    };

    this.dc.onclose = () => {
      this.readyState = MeshiMeshSocket.CLOSED;
      const ev = new CloseEvent('close', { wasClean: true });
      if (typeof this.onclose === 'function') this.onclose(ev);
      this.dispatchEvent(ev);
    };

    this.dc.onerror = (error) => {
      const ev = new ErrorEvent('error', { error });
      if (typeof this.onerror === 'function') this.onerror(ev);
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
