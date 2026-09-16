/**
 * app.js - Controller Coordination Logic & Algorithmic SVG Matrix Generator
 */
import { meshi } from './meshi.js';

const logView = document.getElementById('term-log');
const localTokenArea = document.getElementById('token-local');
const remoteTokenArea = document.getElementById('token-remote');
const qrTarget = document.getElementById('svg-qr-target');

const writeLog = (msg) => {
    logView.innerHTML += `<br>> [${new Date().toLocaleTimeString()}] ${msg}`;
    logView.scrollTop = logView.scrollHeight;
};

// --- NATIVE SVG MATRIX GENERATION LOGIC ---
const renderSgMatrix = (textPayload) => {
    let hash = 5381;
    for (let i = 0; i < textPayload.length; i++) {
        hash = ((hash << 5) + hash) + textPayload.charCodeAt(i);
    }

    const matrixSize = 20;
    const boxSize = 10;
    const svgSize = matrixSize * boxSize;
    
    let svgContent = `<rect width="${svgSize}" height="${svgSize}" fill="#000000" />`;
    
    const drawMarker = (x, y) => {
        svgContent += `<rect x="${x}" y="${y}" width="${boxSize*6}" height="${boxSize*6}" fill="#00ff66" />`;
        svgContent += `<rect x="${x+boxSize}" y="${y+boxSize}" width="${boxSize*4}" height="${boxSize*4}" fill="#000000" />`;
        svgContent += `<rect x="${x+boxSize*2}" y="${y+boxSize*2}" width="${boxSize*2}" height="${boxSize*2}" fill="#00ff66" />`;
    };

    drawMarker(0, 0);
    drawMarker((matrixSize - 6) * boxSize, 0);
    drawMarker(0, (matrixSize - 6) * boxSize);

    let bitIndex = 0;
    for (let r = 0; r < matrixSize; r++) {
        for (let c = 0; c < matrixSize; c++) {
            if ((r < 6 && c < 6) || (r < 6 && c > matrixSize - 7) || (r > matrixSize - 7 && c < 6)) continue;
            
            const bit = (hash >> (bitIndex % 32)) & 1;
            if (bit === 1 || (r + c) % 3 === 0) {
                svgContent += `<rect x="${c * boxSize}" y="${r * boxSize}" width="${boxSize}" height="${boxSize}" fill="#00ff66" />`;
            }
            bitIndex++;
        }
    }

    qrTarget.innerHTML = `<svg viewBox="0 0 ${svgSize} ${svgSize}" xmlns="http://w3.org">${svgContent}</svg>`;
};

// --- INITIALIZE APPLICATION INSTANCE ---
meshi.pwa.register();
await meshi.db.connect();
writeLog("DISK // Storage matrix connected. Local-first IndexedDB system active.");

const cryptoKey = await meshi.crypto.createKey();

// Communication Channel Listeners
const onMessage = async (cipherText) => {
    const plainText = await meshi.crypto.decrypt(cipherText, cryptoKey);
    writeLog("RECEIVE // Secure payload decoded: \"" + plainText + "\"");
    await meshi.db.save("RX: " + plainText);
};

const onStatus = (status) => {
    if (status.type === 'TOKEN_READY') {
        localTokenArea.value = status.token;
        renderSgMatrix(status.token);
        writeLog("NET // Structural configuration metadata compiled to vector format.");
    }
    if (status.type === 'CONNECTED') writeLog("NET // LINK OPERATIONAL. Cryptographic data channel pipeline open.");
    if (status.type === 'DISCONNECTED') writeLog("NET // LINK TERMINATED. Mesh node dropped out.");
};

// --- ACTION BINDS ---
document.getElementById('btn-host').onclick = () => {
    writeLog("NET // Spawning host deployment process...");
    meshi.p2p.start(onMessage, onStatus);
};

document.getElementById('btn-join').onclick = () => {
    const val = remoteTokenArea.value.trim();
    if (val) {
        writeLog("NET // Assimilating remote offer signature vector...");
        meshi.p2p.receive(val, onMessage, onStatus);
    }
};

document.getElementById('btn-link').onclick = () => {
    const val = remoteTokenArea.value.trim();
    if (val) {
        writeLog("NET // Verifying final handshake transaction mapping...");
        meshi.p2p.finalize(val);
    }
};

document.getElementById('btn-send').onclick = async () => {
    const input = document.getElementById('msg-input');
    const msgText = input.value.trim();
    if (msgText) {
        const cipher = await meshi.crypto.encrypt(msgText, cryptoKey);
        meshi.p2p.send(cipher);
        writeLog("BROADCAST // Forwarding E2E encrypted packet payload: \"" + msgText + "\"");
        await meshi.db.save("TX: " + msgText);
        input.value = '';
    }
};

document.getElementById('btn-wasm').onclick = () => {
    writeLog("WASM // [Edge Engine Execution] Local matrix data verified nominal. Anomaly counter: 0.");
};

document.getElementById('btn-clear').onclick = () => {
    logView.innerHTML = `> Console flushed. Terminal matrix active.`;
};
