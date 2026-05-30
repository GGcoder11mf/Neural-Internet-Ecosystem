/* ===== FakeNet Ecosystem — Shared JS v3 ===== */
/* SSE real-time, auto-tick, neural net status */

// Simulation API helpers
const API = {
  get: url => fetch(url).then(r => r.json()),
  post: (url, data) => fetch(url, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data || {}) }).then(r => r.json()),
};

// HTML escape
function esc(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

// ===== SIM CONTROLS =====

function getParams() {
  return {
    n_agents: parseInt(document.getElementById('rAgents').value),
    n_topics: parseInt(document.getElementById('rTopics').value),
    bot_fraction: parseInt(document.getElementById('rBots').value) / 100,
    echo_chamber_strength: parseInt(document.getElementById('rEcho').value) / 100,
    virality_sensitivity: parseInt(document.getElementById('rViral').value) / 100,
    novelty_decay: parseInt(document.getElementById('rDecay').value) / 100,
    mutation_rate: parseInt(document.getElementById('rMut').value) / 100,
  };
}

async function createSim() {
  const data = await API.post('/api/create', getParams());
  if (data.error) { alert(data.error); return; }
  document.getElementById('tickNum').textContent = '0';
  refreshAll();
  checkNNStatus();
}

async function resetSim() {
  await API.post('/api/reset', getParams());
  document.getElementById('tickNum').textContent = '0';
  refreshAll();
}

async function doTick(n) {
  if (n === 1) {
    const data = await API.post('/api/tick');
    if (data.error) { alert('Create a simulation first!'); return; }
    document.getElementById('tickNum').textContent = data.tick || '?';
  } else {
    const data = await API.post('/api/ticks', { n });
    if (data.error) { alert('Create a simulation first!'); return; }
    document.getElementById('tickNum').textContent = data.tick || '?';
  }
  refreshAll();
}

// ===== AUTO-TICK CONTROLS =====

let autoTickInterval = null;

async function startAutoTick() {
  const speed = parseFloat(document.getElementById('rSpeed').value) || 1;
  const data = await API.post('/api/auto-tick/start', { speed });
  if (data.status === 'running') {
    document.getElementById('btnAutoStart').disabled = true;
    document.getElementById('btnAutoStop').disabled = false;
    document.getElementById('btnAutoStart').classList.add('pulse');
  }
}

async function stopAutoTick() {
  const data = await API.post('/api/auto-tick/stop');
  if (data.status === 'stopped') {
    document.getElementById('btnAutoStart').disabled = false;
    document.getElementById('btnAutoStop').disabled = true;
    document.getElementById('btnAutoStart').classList.remove('pulse');
  }
}

function updateSpeed(val) {
  const speed = val / 1;
  document.getElementById('speedVal').textContent = speed.toFixed(1) + 's';
  // Update auto-tick speed if running
  if (!document.getElementById('btnAutoStart').disabled) {
    // Not running, will use this speed on next start
  }
}

async function checkAutoTickStatus() {
  try {
    const data = await API.get('/api/auto-tick/status');
    if (data.running) {
      document.getElementById('btnAutoStart').disabled = true;
      document.getElementById('btnAutoStop').disabled = false;
      document.getElementById('btnAutoStart').classList.add('pulse');
    } else {
      document.getElementById('btnAutoStart').disabled = false;
      document.getElementById('btnAutoStop').disabled = true;
      document.getElementById('btnAutoStart').classList.remove('pulse');
    }
  } catch(e) {}
}

// ===== NEURAL NET STATUS =====

async function checkNNStatus() {
  try {
    const data = await API.get('/api/neural-net');
    const el = document.getElementById('nnStatus');
    const indicator = document.getElementById('nnIndicator');
    const text = document.getElementById('nnText');
    
    if (!el) return;
    
    if (data.torch_available) {
      el.style.display = '';
      if (data.cuda_available) {
        indicator.className = 'nn-indicator nn-cuda';
        text.textContent = `NN: CUDA (${data.train_steps} steps)`;
      } else {
        indicator.className = 'nn-indicator nn-cpu';
        text.textContent = `NN: CPU (${data.train_steps} steps)`;
      }
      if (data.latest_loss) {
        text.textContent += ` loss=${data.latest_loss}`;
      }
    } else {
      el.style.display = '';
      indicator.className = 'nn-indicator nn-off';
      text.textContent = 'NN: No PyTorch';
    }
  } catch(e) {}
}

// ===== REFRESH CURRENT PAGE DATA =====

function refreshAll() {
  const path = window.location.pathname;
  if (path === '/' || path === '/index.html') {
    if (typeof loadFeed === 'function') loadFeed(currentSort || 'attention');
  } else if (path === '/explore') {
    if (typeof loadTrending === 'function') loadTrending();
    if (typeof loadAgents === 'function') loadAgents();
    if (typeof loadViral === 'function') loadViral();
  } else if (path === '/communities') {
    if (typeof loadCommunities === 'function') loadCommunities();
  } else if (path === '/memes') {
    if (typeof loadMemes === 'function') loadMemes();
  } else if (path === '/analytics') {
    if (typeof loadAnalytics === 'function') loadAnalytics();
  }
  loadTrending();
  loadPlatform();
  loadSentimentWave();
  checkNNStatus();
}

// ===== SIDEBAR: TRENDING =====

async function loadTrending() {
  const data = await API.get('/api/trending');
  const body = document.getElementById('trendingBody');
  if (!body) return;

  if (!data.hashtags || data.hashtags.length === 0) {
    body.innerHTML = '<div style="color:var(--text-secondary);font-size:14px;padding:8px 0">Create a simulation to see trends</div>';
    return;
  }

  let allHashtags = data.hashtags || [];
  let popularHashtags = data.popular_hashtags || [];
  
  let seen = new Set();
  let combined = [];
  for (const h of [...popularHashtags, ...allHashtags]) {
    if (!seen.has(h.tag)) {
      seen.add(h.tag);
      combined.push(h);
    }
  }

  body.innerHTML = combined.slice(0, 10).map((h, i) => `
    <div class="trend-item">
      <div class="trend-category">${i + 1} · Trending</div>
      <div class="trend-name">${esc(h.tag)}</div>
      <div class="trend-posts">${h.volume.toFixed(1)} volume</div>
    </div>
  `).join('');
}

// ===== SIDEBAR: PLATFORM =====

async function loadPlatform() {
  const data = await API.get('/api/platform');
  const card = document.getElementById('platformCard');
  const body = document.getElementById('platformBody');
  if (!card || !body) return;

  if (!data.echo_chamber_strength && data.echo_chamber_strength !== 0) {
    card.style.display = 'none';
    return;
  }
  card.style.display = '';

  body.innerHTML = `
    <div class="community-stat"><span>Echo Chamber</span><span class="val">${data.echo_chamber_strength.toFixed(2)}</span></div>
    <div class="polarization-bar"><div class="polarization-fill" style="width:${data.echo_chamber_strength * 100}%;background:var(--accent)"></div></div>
    <div class="community-stat"><span>Virality</span><span class="val">${data.virality_sensitivity.toFixed(2)}</span></div>
    <div class="community-stat"><span>Controversy Boost</span><span class="val" style="color:var(--orange)">${(data.controversy_boost || 0).toFixed(2)}</span></div>
    <div class="community-stat"><span>Decay</span><span class="val">${data.novelty_decay.toFixed(2)}</span></div>
    <div class="community-stat"><span>Trending</span><span class="val">${data.trending_count || 0}</span></div>
    <div class="community-stat"><span>Tick</span><span class="val">${data.tick_count || 0}</span></div>
  `;
}

// ===== SIDEBAR: SENTIMENT WAVE =====

async function loadSentimentWave() {
  try {
    const data = await API.get('/api/analytics');
    const card = document.getElementById('waveCard');
    const body = document.getElementById('waveBody');
    if (!card || !body) return;

    const wave = data.sentiment_wave;
    if (!wave) {
      card.style.display = 'none';
      return;
    }
    card.style.display = '';
    card.className = `card wave-${wave.sentiment}`;

    const badgeClass = `wave-badge wave-badge-${wave.sentiment}`;
    body.innerHTML = `
      <span class="${badgeClass}">${wave.sentiment.toUpperCase()}</span>
      <div class="community-stat" style="margin-top:8px"><span>Intensity</span><span class="val">${wave.intensity.toFixed(2)}</span></div>
      <div class="community-stat"><span>Duration left</span><span class="val">${wave.duration} ticks</span></div>
    `;
  } catch(e) {}
}

// Auto-init trending + platform on page load
document.addEventListener('DOMContentLoaded', () => {
  loadTrending();
  loadPlatform();
  loadSentimentWave();
  checkAutoTickStatus();
  checkNNStatus();
});
