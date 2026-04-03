// ── DOM refs ──────────────────────────────────────────────────────────────
const menu             = document.getElementById('menu')!;
const game             = document.getElementById('game')!;
const songList         = document.getElementById('song-list')!;
const uploadForm       = document.getElementById('upload-form')!;
const playerScores     = document.getElementById('player-scores')!;
const videoFrame       = document.getElementById('video-frame') as HTMLImageElement;
const scoreText        = document.getElementById('score-text')!;
const silhouetteOverlay= document.getElementById('silhouette-overlay') as HTMLImageElement;
const beatCounter      = document.getElementById('beat-counter')!;
const fpsDisplay       = document.getElementById('fps-display')!;
const btnStart         = document.getElementById('btn-start') as HTMLButtonElement;
const uploadStatus     = document.getElementById('upload-status')!;
const gameAudio        = document.getElementById('game-audio') as HTMLAudioElement;

// ── State ─────────────────────────────────────────────────────────────────
let selectedSong: string | null = null;
let ws: WebSocket | null = null;
let lastFrameMs = 0;
let scorePopTimeout: ReturnType<typeof setTimeout> | null = null;

interface PlayerState { rating: string; updatedAt: number; }
const playerStates = new Map<string, PlayerState>();

const EFFECT_DURATION_MS = 500;

// ── Song list ─────────────────────────────────────────────────────────────
async function loadSongs() {
  try {
    const res  = await fetch('/songs');
    const data = await res.json() as { songs: string[] };
    renderSongList(data.songs);
  } catch {
    console.error('Could not reach server');
  }
}

function renderSongList(songs: string[]) {
  songList.innerHTML = '';
  if (songs.length === 0) {
    songList.textContent = 'No songs yet — upload one!';
    return;
  }
  for (const song of songs) {
    const card = document.createElement('div');
    card.className = 'song-card' + (song === selectedSong ? ' selected' : '');
    card.textContent = song;
    card.onclick = () => selectSong(song);
    songList.appendChild(card);
  }
}

function selectSong(name: string) {
  selectedSong = name;
  btnStart.disabled = false;
  document.querySelectorAll('.song-card').forEach(c => {
    c.classList.toggle('selected', c.textContent === name);
  });
}

// ── Game flow ─────────────────────────────────────────────────────────────
async function startGame() {
  if (!selectedSong) return;

  const res  = await fetch(`/start/${selectedSong}`, { method: 'POST' });
  const data = await res.json() as { status?: string; error?: string };
  if (data.error) { alert(data.error); return; }

  menu.style.display = 'none';
  game.style.display = 'block';

  gameAudio.src = `/song/${selectedSong}/audio`;
  gameAudio.play().catch(() => console.warn('Audio autoplay blocked'));

  connectWebSocket();
}

async function stopGame() {
  await fetch('/stop', { method: 'POST' });
  ws?.close();
  showMenu();
}

function showMenu() {
  game.style.display  = 'none';
  menu.style.display  = 'block';
  gameAudio.pause();
  ws = null;
  playerStates.clear();
  playerScores.innerHTML = '';
  lastFrameMs = 0;
  silhouetteOverlay.classList.add('hidden');
  loadSongs();
}

// ── WebSocket ─────────────────────────────────────────────────────────────
function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onmessage = (ev) => {
    try { handleMessage(JSON.parse(ev.data as string)); }
    catch { /* ignore parse errors */ }
  };

  ws.onclose = () => showMenu();
}

interface FrameMsg {
  type:       'frame';
  frame:      string;
  scores:     Record<string, string>;
  cumulative: Record<string, number>;
  beat:       number;
  effect_ms:  number | null;
  silhouette: string | null;
}

function handleMessage(msg: { type: string } & Partial<FrameMsg>) {
  if (msg.type === 'stopped') { showMenu(); return; }
  if (msg.type !== 'frame')   return;

  const { frame, scores = {}, cumulative = {}, beat = 0, effect_ms, silhouette } = msg;

  // Video frame
  if (frame) videoFrame.src = `data:image/jpeg;base64,${frame}`;

  // FPS
  const now = Date.now();
  if (lastFrameMs) fpsDisplay.textContent = `FPS: ${Math.round(1000 / (now - lastFrameMs))}`;
  lastFrameMs = now;

  beatCounter.textContent = `Beat: ${beat}`;

  // Silhouette overlay
  if (silhouette) {
    silhouetteOverlay.src = `data:image/jpeg;base64,${silhouette}`;
    silhouetteOverlay.classList.remove('hidden');
  }

  updateScoreText(scores, effect_ms);
  updatePlayerScores(scores, cumulative, effect_ms);
}

// ── Score text pop ────────────────────────────────────────────────────────
const LABEL: Record<string, string> = {
  GREAT: 'GREAT!',
  OK:    'GOOD!',
  BAD:   'BAD!',
};

function updateScoreText(scores: Record<string, string>, effectMs: number | null) {
  if (effectMs === null || effectMs >= EFFECT_DURATION_MS) return;
  // Only trigger on the very first message of a new effect (effectMs is small)
  if (effectMs > 80) return;

  // Best rating across all players
  const ratings = Object.values(scores);
  let best = 'BAD';
  if (ratings.includes('GREAT')) best = 'GREAT';
  else if (ratings.includes('OK')) best = 'OK';

  // Remove old animation class, force reflow, re-add
  scoreText.classList.remove('pop', 'GREAT', 'OK', 'BAD');
  scoreText.textContent = LABEL[best];
  // eslint-disable-next-line @typescript-eslint/no-unused-expressions
  scoreText.offsetWidth; // reflow trick to restart animation
  scoreText.classList.add(best, 'pop');

  if (scorePopTimeout) clearTimeout(scorePopTimeout);
  scorePopTimeout = setTimeout(() => {
    scoreText.classList.remove('pop');
  }, 600);
}

// ── Per-player score cards ────────────────────────────────────────────────
function updatePlayerScores(
  scores: Record<string, string>,
  cumulative: Record<string, number>,
  effectMs: number | null,
) {
  const now = Date.now();

  if (effectMs !== null && effectMs < EFFECT_DURATION_MS) {
    for (const [id, rating] of Object.entries(scores)) {
      playerStates.set(id, { rating, updatedAt: now });
    }
  }

  for (const id of Object.keys(cumulative)) {
    if (!playerStates.has(id)) playerStates.set(id, { rating: 'idle', updatedAt: 0 });
  }

  for (const [id, state] of playerStates.entries()) {
    const isActive   = now - state.updatedAt < EFFECT_DURATION_MS;
    const totalScore = cumulative[id] ?? 0;

    let card = document.getElementById(`player-${id}`);
    if (!card) {
      card = document.createElement('div');
      card.id = `player-${id}`;
      playerScores.appendChild(card);
    }

    card.className = `player-score ${isActive ? state.rating : 'idle'}`;
    card.innerHTML  = `<div>P${id}</div>`
                    + `<div>${isActive ? LABEL[state.rating] ?? state.rating : '...'}</div>`
                    + `<div style="font-size:0.9rem;opacity:0.75">${totalScore} pts</div>`;
  }

  // Remove players gone for > 5 s
  for (const [id, state] of playerStates.entries()) {
    if (state.updatedAt > 0 && now - state.updatedAt > 5000) {
      playerStates.delete(id);
      document.getElementById(`player-${id}`)?.remove();
    }
  }
}

// ── Upload ────────────────────────────────────────────────────────────────
async function uploadSong() {
  const name  = (document.getElementById('upload-name')  as HTMLInputElement).value.trim();
  const url   = (document.getElementById('upload-url')   as HTMLInputElement).value.trim();
  const bpm   = (document.getElementById('upload-bpm')   as HTMLInputElement).value;
  const start = (document.getElementById('upload-start') as HTMLInputElement).value;

  if (!name || !url || !bpm) { alert('Fill in Song Name, URL, and BPM'); return; }

  const body = new FormData();
  body.append('song_name', name);
  body.append('url', url);
  body.append('bpm', bpm);
  body.append('start_beat', start || '0');

  uploadStatus.textContent = 'Uploading & processing… (this can take a few minutes)';
  try {
    const res  = await fetch('/upload', { method: 'POST', body });
    const data = await res.json() as { status?: string; song?: string; error?: string };
    uploadStatus.textContent = data.status === 'uploading'
      ? `Processing "${data.song}"… refresh the song list when done.`
      : `Error: ${data.error ?? 'unknown'}`;
  } catch {
    uploadStatus.textContent = 'Upload failed — is the server running?';
  }
}

// ── Wire up buttons ───────────────────────────────────────────────────────
document.getElementById('btn-refresh')!.onclick      = loadSongs;
document.getElementById('btn-show-upload')!.onclick  = () => uploadForm.classList.toggle('hidden');
document.getElementById('btn-start')!.onclick        = startGame;
document.getElementById('btn-stop')!.onclick         = stopGame;
document.getElementById('btn-upload')!.onclick       = uploadSong;

loadSongs();
