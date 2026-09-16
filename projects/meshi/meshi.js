/**
 * meshi.js v1.0.0 - Pure Modular JavaScript Framework Core
 * 0% External Libraries // 100% Native Architecture
 */
export const meshi = {
    crypto: {
        async createKey() {
            return await crypto.subtle.generateKey({ name: "AES-GCM", length: 256 }, true, ["encrypt", "decrypt"]);
        },
        async serializeKey(key) {
            const raw = await crypto.subtle.exportKey("raw", key);
            return btoa(String.fromCharCode(...new Uint8Array(raw)));
        },
        async unserializeKey(b64) {
            const buf = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
            return await crypto.subtle.importKey("raw", buf, "AES-GCM", true, ["encrypt", "decrypt"]);
        },
        async encrypt(text, key) {
            const iv = crypto.getRandomValues(new Uint8Array(12));
            const enc = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, key, new TextEncoder().encode(text));
            const res = new Uint8Array(iv.length + enc.byteLength);
            res.set(iv, 0);
            res.set(new Uint8Array(enc), iv.length);
            return btoa(String.fromCharCode(...res));
        },
        async decrypt(b64, key) {
            const buf = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
            const iv = buf.slice(0, 12);
            const data = buf.slice(12);
            const dec = await crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, data);
            return new TextDecoder().decode(dec);
        }
    },

    db: {
        instance: null,
        async connect() {
            return new Promise((res, rej) => {
                const req = indexedDB.open('MeshiLocalStore', 1);
                req.onupgradeneeded = e => e.target.result.createObjectStore('logs', { keyPath: 'id', autoIncrement: true });
                req.onsuccess = e => { this.instance = e.target.result; res(); };
                req.onerror = e => rej(e.target.error);
            });
        },
        async save(record) {
            if (!this.instance) return;
            return new Promise(res => {
                this.instance.transaction(['logs'], 'readwrite').objectStore('logs').add({ time: Date.now(), record });
                res();
            });
        }
    },

    p2p: {
        peer: null,
        chan: null,
        init(onMsg, onStatus) {
            this.peer = new RTCPeerConnection({ iceServers: [{ urls: 'stun:://google.com' }] });
            this.peer.onicecandidate = e => {
                if (!e.candidate) {
                    const token = btoa(JSON.stringify(this.peer.localDescription));
                    onStatus({ type: 'TOKEN_READY', token });
                }
            };
            this.peer.ondatachannel = e => this._bind(e.channel, onMsg, onStatus);
        },
        async start(onMsg, onStatus) {
            this.init(onMsg, onStatus);
            this.chan = this.peer.createDataChannel("meshi-pipe");
            this._bind(this.chan, onMsg, onStatus);
            await this.peer.setLocalDescription(await this.peer.createOffer());
        },
        async receive(b64Token, onMsg, onStatus) {
            this.init(onMsg, onStatus);
            await this.peer.setRemoteDescription(new RTCSessionDescription(JSON.parse(atob(b64Token))));
            await this.peer.setLocalDescription(await this.peer.createAnswer());
        },
        async finalize(b64Token) {
            await this.peer.setRemoteDescription(new RTCSessionDescription(JSON.parse(atob(b64Token))));
        },
        send(data) {
            if (this.chan && this.chan.readyState === "open") this.chan.send(data);
        },
        _bind(ch, onMsg, onStatus) {
            this.chan = ch;
            ch.onopen = () => onStatus({ type: 'CONNECTED' });
            ch.onclose = () => onStatus({ type: 'DISCONNECTED' });
            ch.onmessage = e => onMsg(e.data);
        }
    },

    pwa: {
        register() {
            if ('serviceWorker' in navigator) {
                const sw = `self.addEventListener('fetch', e => e.respondWith(fetch(e.request)));`;
                const blob = new Blob([sw], { type: 'application/javascript' });
                navigator.serviceWorker.register(URL.createObjectURL(blob), { scope: './' });
            }
        }
    }
};

