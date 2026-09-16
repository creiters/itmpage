export class P2PNode {
    constructor(callbacks) {
        this.onMessage = callbacks.onMessage;
        this.onStatus = callbacks.onStatus;
        this.onFileProgress = callbacks.onFileProgress;
        this.onFileReceived = callbacks.onFileReceived;
        
        this.peerConnection = null;
        this.dataChannel = null;
        this.config = { iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] };
        
        // Chunk configuration for metadata processing and files
        this.CHUNK_SIZE = 16384; // 16KB standard WebRTC buffer allocation
        this.fileReceiver = {
            currentMeta: null,
            receivedChunks: [],
            bytesReceived: 0
        };
    }

    init() {
        this.peerConnection = new RTCPeerConnection(this.config);

        this.peerConnection.onicecandidate = (event) => {
            if (!event.candidate) {
                this.onStatus("ICE assembly complete. QR and internal matrices loaded.");
                const sdpString = btoa(JSON.stringify(this.peerConnection.localDescription));
                document.getElementById('p2p-local-sdp').value = sdpString;
                if (this.onLocalSdpReady) this.onLocalSdpReady(sdpString);
            }
        };

        this.peerConnection.ondatachannel = (event) => {
            this.setupDataChannel(event.channel);
        };
    }

    async createOffer() {
        this.init();
        this.dataChannel = this.peerConnection.createDataChannel("sshanet-mesh-channel");
        this.setupDataChannel(this.dataChannel);

        const offer = await this.peerConnection.createOffer();
        await this.peerConnection.setLocalDescription(offer);
        this.onStatus("Assembling network crypt-tokens...");
    }

    async acceptOffer(base64Offer) {
        this.init();
        const offer = JSON.parse(atob(base64Offer));
        await this.peerConnection.setRemoteDescription(new RTCSessionDescription(offer));
        
        const answer = await this.peerConnection.createAnswer();
        await this.peerConnection.setLocalDescription(answer);
        this.onStatus("Local target signature resolved. Scan the generated verification token.");
    }

    async connect(base64Answer) {
        const answer = JSON.parse(atob(base64Answer));
        await this.peerConnection.setRemoteDescription(new RTCSessionDescription(answer));
    }

    setupDataChannel(channel) {
        this.dataChannel = channel;
        this.dataChannel.binaryType = "arraybuffer";
        
        this.dataChannel.onopen = () => this.onStatus("🚀 P2P LAYER ACTIVE: Cryptographic link verified.");
        this.dataChannel.onclose = () => this.onStatus("❌ P2P CONNECTION SEVERED.");
        
        this.dataChannel.onmessage = (event) => this.handleIncomingMessage(event.data);
    }

    handleIncomingMessage(data) {
        if (typeof data === "string") {
            try {
                const msgObj = JSON.parse(data);
                if (msgObj.type === "FILE_META") {
                    this.fileReceiver.currentMeta = msgObj;
                    this.fileReceiver.receivedChunks = [];
                    this.fileReceiver.bytesReceived = 0;
                    this.onStatus(`Incoming file transmission sequence: ${msgObj.name} (${msgObj.size} bytes)`);
                    return;
                }
            } catch (e) {
                // Not a structural JSON object - handle as standard network packet string
            }
            this.onMessage(data);
        } else {
            // Process incoming binary file pipeline stream segment
            this.fileReceiver.receivedChunks.push(data);
            this.fileReceiver.bytesReceived += data.byteLength;
            
            const pct = Math.floor((this.fileReceiver.bytesReceived / this.fileReceiver.currentMeta.size) * 100);
            this.onFileProgress(pct);

            if (this.fileReceiver.bytesReceived >= this.fileReceiver.currentMeta.size) {
                const completeBlob = new Blob(this.fileReceiver.receivedChunks, { type: this.fileReceiver.currentMeta.mime });
                this.onFileReceived(this.fileReceiver.currentMeta.name, completeBlob);
                this.fileReceiver.currentMeta = null;
            }
        }
    }

    sendPacket(payload) {
        if (this.dataChannel && this.dataChannel.readyState === "open") {
            this.dataChannel.send(payload);
        }
    }

    async sendFile(file, progressCallback) {
        if (!this.dataChannel || this.dataChannel.readyState !== "open") return;

        // Step A: Broadcast Meta Configuration Array to Receiver Node
        const meta = { type: "FILE_META", name: file.name, size: file.size, mime: file.type };
        this.dataChannel.send(JSON.stringify(meta));

        // Step B: Slice file array binary stack buffer stream
        const reader = new FileReader();
        let offset = 0;

        const sliceAndSend = () => {
            if (offset >= file.size) {
                progressCallback(100);
                return;
            }
            const slice = file.slice(offset, offset + this.CHUNK_SIZE);
            reader.readAsArrayBuffer(slice);
        };

        reader.onload = (e) => {
            this.dataChannel.send(e.target.result);
            offset += e.target.result.byteLength;
            progressCallback(Math.floor((offset / file.size) * 100));
            
            // Handle high speed execution buffering dynamics
            if (this.dataChannel.bufferedAmount > this.CHUNK_SIZE * 4) {
                setTimeout(sliceAndSend, 10);
            } else {
                sliceAndSend();
            }
        };

        sliceAndSend();
    }
}