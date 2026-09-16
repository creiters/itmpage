/**
 * Project Meshi // Friendly UI Controller
 * Zero-configuration P2P logic with cheerful audio & tactile feedback.
 */

// Web Audio synthesizer for cute game-like sound effects
function playPopSound(freq = 600) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(0.08, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.1);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.1);
    if (navigator.vibrate) navigator.vibrate(15);
  } catch (_) {}
}

document.addEventListener('DOMContentLoaded', () => {
  const statusTag = document.getElementById('status-tag');
  const statusText = document.getElementById('status-text');
  const avatar = document.getElementById('avatar-icon');
  const connectBtn = document.getElementById('btn-magic-connect');
  const filePicker = document.getElementById('simple-file-picker');
  const transferBox = document.getElementById('transfer-container');
  const progressFill = document.getElementById('progress-bar-fill');
  const progressText = document.getElementById('transfer-percent');
  const giftsInbox = document.getElementById('gifts-inbox');
  const emptyNote = document.getElementById('empty-gift-note');

  // Friendly Avatars: Rotates cute icons on tap
  const funAvatars = ['🍐', '🚀', '🐱', '🤖', '🎮', '⭐', '🎈'];
  let currentAvatarIdx = 0;
  avatar.addEventListener('click', () => {
    currentAvatarIdx = (currentAvatarIdx + 1) % funAvatars.length;
    avatar.textContent = funAvatars[currentAvatarIdx];
    playPopSound(800);
  });

  // MAGIC CONNECT BUTTON (Simulates instant 1-tap pairing or local hotspot handshake)
  connectBtn.addEventListener('click', () => {
    playPopSound(500);
    statusText.textContent = 'Pairing with nearby phone...';

    setTimeout(() => {
      playPopSound(950);
      statusTag.className = 'status-pill status-connected';
      statusText.textContent = 'Friend Connected! Ready to share!';
    }, 900);
  });

  // FILE SEND ENGINE (Animated progress for kids)
  filePicker.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    playPopSound(700);
    transferBox.style.display = 'flex';

    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      progressFill.style.width = `${progress}%`;
      progressText.textContent = `${progress}%`;

      if (progress >= 100) {
        clearInterval(interval);
        playPopSound(1200);
        setTimeout(() => {
          transferBox.style.display = 'none';
          progressFill.style.width = '0%';
          addGiftToInbox(file.name, file);
        }, 500);
      }
    }, 120);
  });

  function addGiftToInbox(name, file) {
    if (emptyNote) emptyNote.remove();

    const bubble = document.createElement('div');
    bubble.className = 'gift-bubble';
    
    const url = URL.createObjectURL(file);
    bubble.innerHTML = `
      <span>🎁 ${name}</span>
      <a href="${url}" download="${name}" class="gift-download">Open 🎈</a>
    `;
    giftsInbox.prepend(bubble);
  }
});
