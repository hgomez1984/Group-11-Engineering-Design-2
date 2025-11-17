// JS_ED2.js — full file

document.addEventListener('DOMContentLoaded', () => {
  // ====== Config / Refs ======
  const API_BASE = "http://127.0.0.1:8000";
  const d = document;

  // IDs in your HTML
  const fileInput   = d.getElementById('file-input');
  const dropzone    = d.getElementById('dropzone');
  const gallery     = d.getElementById('gallery');
  const statusEl    = d.getElementById('status');
  const clearBtn    = d.getElementById('btn-clear') || d.getElementById('clearBtn');
  const analyzeBtn  = d.getElementById('analyzeBtn');
  const resultsEl   = d.getElementById('results');   // big 2-panel area

  const formLogin   = d.getElementById('form-login');
  const loginBtn    = d.getElementById('loginBtn');
  const logoutBtn   = d.getElementById('logoutBtn');
  const loginStatus = d.getElementById('loginStatus');
  const loginUserEl = d.getElementById('login-username');
  const loginPassEl = d.getElementById('login-password');

  const patientsList= d.getElementById('patientsList');

  // Upload state
  const filesState = [];
  const MAX_MB = 10;

  // ====== Helpers ======
  function humanSize(bytes){
    const u=['B','KB','MB','GB']; let i=0,n=bytes;
    while(n>=1024 && i<u.length-1){ n/=1024; i++; }
    return `${n.toFixed(i?1:0)} ${u[i]}`;
  }
  function setStatus(msg,type){
    if (!statusEl) return;
    statusEl.textContent = msg;
    statusEl.className = 'status ' + (type||'');
  }

  // Token helpers
  function setToken(t){ sessionStorage.setItem('token', t); }
  function getToken(){ return sessionStorage.getItem('token'); }
  function clearToken(){ sessionStorage.removeItem('token'); }

  async function authFetch(url, options={}){
    const headers = new Headers(options.headers || {});
    const t = getToken();
    if (t) headers.set('Authorization','Bearer '+t);
    return fetch(url, { ...options, headers });
  }

  function setLoggedInUI(isIn){
    if (logoutBtn) logoutBtn.style.display = isIn ? '' : 'none';
    if (loginBtn)  loginBtn.disabled = isIn;
    if (loginStatus){
      loginStatus.textContent = isIn ? 'Logged in.' : 'Please log in to analyze.';
      loginStatus.className = 'status ' + (isIn ? 'ok' : 'warn');
    }
  }

  // ------- Shared helper: get current user from token -------
  async function fetchCurrentUser() {
    const res = await authFetch(`${API_BASE}/api/me`, {
      method: 'GET'
    });

    if (!res.ok) {
      throw new Error('Failed to fetch current user');
    }

    return await res.json(); // expect { username, role, ... }
  }

  // ====== Login flow (PATIENT portal) ======
  async function doLogin(username, password){
    const body = new URLSearchParams({ username, password });
    const res = await fetch(`${API_BASE}/auth/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body
    });

    if (!res.ok){
      let msg = 'Login failed.';
      try { const j = await res.json(); if (j.detail) msg = j.detail; } catch {}
      throw new Error(msg);
    }

    const data = await res.json(); // {access_token, token_type}
    const token = data.access_token;
    setToken(token);

    // ✅ Check that this is a PATIENT account
    let info;
    try {
      info = await fetchCurrentUser();
    } catch (err) {
      clearToken();
      setLoggedInUI(false);
      throw new Error('Could not verify account. Please try again.');
    }

    if (info.role !== 'patient') {
      // Not allowed on this portal
      clearToken();
      setLoggedInUI(false);
      throw new Error('Please use the Staff Portal to log in with this account.');
    }

    setLoggedInUI(true);

    // Clear login fields
    if (loginUserEl) loginUserEl.value = '';
    if (loginPassEl) loginPassEl.value = '';

    // Save username for welcome page
    sessionStorage.setItem('username', info.username || username);

    if (typeof listPatients === 'function') listPatients();

    // Redirect to patient welcome page
    window.location.href = `${API_BASE}/welcome`;
  }

  async function onLoginSubmit(e){
    e.preventDefault();
    const u = (loginUserEl?.value || '').trim();
    const p = (loginPassEl?.value || '');
    if (!u || !p){
      if (loginStatus) {
        loginStatus.textContent = 'Enter username & password.';
        loginStatus.className = 'status warn';
      }
      return;
    }
    try{
      if (loginStatus) {
        loginStatus.textContent = 'Signing in…';
        loginStatus.className = 'status';
      }
      await doLogin(u, p);
    }catch(err){
      if (loginStatus) {
        loginStatus.textContent = err.message;
        loginStatus.className = 'status err';
      }
    }
  }

  formLogin?.addEventListener('submit', onLoginSubmit);
  loginBtn?.addEventListener('click', () => formLogin?.requestSubmit());
  logoutBtn?.addEventListener('click', ()=>{
    clearToken();
    setLoggedInUI(false);
    if (patientsList) patientsList.textContent='Log in to view patients.';
  });

  // ====== Register modal (optional) ======
  const regBtns     = d.querySelectorAll('.js-register');
  const regModal    = d.getElementById('registerModal');
  const regSave     = d.getElementById('registerSaveBtn');
  const regCancel   = d.getElementById('registerCancelBtn');
  const regStatus   = d.getElementById('registerStatus');
  const reg_username= d.getElementById('reg_username');
  const reg_password= d.getElementById('reg_password');
  const reg_name    = d.getElementById('reg_name');
  const reg_dob     = d.getElementById('reg_dob');
  const reg_sex     = d.getElementById('reg_sex');
  const reg_note    = d.getElementById('reg_note');

  function openReg(){
    if (regStatus){ regStatus.textContent = ''; regStatus.className='status'; }
    [reg_username, reg_password, reg_name, reg_dob, reg_sex, reg_note].forEach(el => el && (el.value=''));
    regModal?.classList.remove('hidden');
    regModal?.setAttribute('aria-hidden','false');
  }
  function closeReg(){
    regModal?.classList.add('hidden');
    regModal?.setAttribute('aria-hidden','true');
  }
  regBtns.forEach(b => b.addEventListener('click', openReg));
  regCancel?.addEventListener('click', closeReg);
  regModal?.querySelector('.modal-backdrop')?.addEventListener('click', closeReg);

  async function registerAccount(){
    const username = (reg_username?.value || '').trim();
    const password = (reg_password?.value || '').trim();
    const name     = (reg_name?.value || '').trim();
    const dob      = (reg_dob?.value || '') || null;
    const sex      = (reg_sex?.value || '') || null;
    const note     = (reg_note?.value || '') || null;

    if(!username || !password || !name){
      if (regStatus) {
        regStatus.textContent='Username, password, and full name are required.';
        regStatus.className='status warn';
      }
      return;
    }

    try{
      if (regStatus) { regStatus.textContent='Creating account…'; regStatus.className='status'; }
      const r = await fetch(`${API_BASE}/auth/register`, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ username, password, role:"patient", name, dob, sex, note })
      });

      let detail = '';
      if (!r.ok && r.status !== 400) {
        try { const j = await r.json(); detail = j.detail || JSON.stringify(j); }
        catch { detail = await r.text(); }
        throw new Error(detail || `HTTP ${r.status}`);
      }

      const loginRes = await fetch(`${API_BASE}/auth/token`, {
        method:'POST',
        headers:{'Content-Type':'application/x-www-form-urlencoded'},
        body: new URLSearchParams({ username, password })
      });
      if (!loginRes.ok){
        let msg = '';
        try { const j = await loginRes.json(); msg = j.detail || JSON.stringify(j); }
        catch { msg = await loginRes.text(); }
        throw new Error(msg || `HTTP ${loginRes.status}`);
      }

      const tokenData = await loginRes.json();
      setToken(tokenData.access_token);
      sessionStorage.setItem('username', username);
      setLoggedInUI(true);

      [reg_username, reg_password, reg_name, reg_dob, reg_sex, reg_note].forEach(el => el && (el.value=''));
      if (regStatus) regStatus.textContent = '✅ Account ready. You’re logged in.';

      setTimeout(() => {
        closeReg();
        if (regStatus) regStatus.textContent = '';
        window.location.href = `${API_BASE}/welcome`;
      }, 700);

    } catch (err){
      if (regStatus) {
        regStatus.textContent = `Registration failed: ${err.message || err}`;
        regStatus.className='status err';
      }
    }
  }
  regSave?.addEventListener('click', (e)=>{
    e.preventDefault();
    e.stopPropagation();
    registerAccount();
  });

  // ====== Upload / Gallery ======
  function ensureIds() {
    const missing = [];
    if (!dropzone) missing.push('#dropzone');
    if (!gallery)  missing.push('#gallery');
    if (missing.length) setStatus(`Missing required element(s): ${missing.join(', ')}`, 'err');
  }
  ensureIds();

  function getOrCreateFileInput() {
    if (fileInput) return fileInput;
    if (!dropzone) return null;
    const input = document.createElement('input');
    input.type = 'file';
    input.id = 'file-input';
    input.accept = 'image/*,.dcm,.dicom,.tif,.tiff,.heic,.heif,.webp';
    input.multiple = true;
    input.style.display = 'none';
    dropzone.appendChild(input);
    return input;
  }
  const fileInputEl = getOrCreateFileInput();

  function humanExt(name) {
    const ext = (name.split('.').pop() || '').toLowerCase();
    return ext ? ext.toUpperCase() : '';
  }

  function addThumb(file){
    const url = URL.createObjectURL(file);
    const fig = document.createElement('figure');
    fig.className = 'thumb';
    fig.style.position = 'relative';
    fig.style.margin = '8px';
    fig.style.border = '1px solid var(--border, #2a2a2a)';
    fig.style.borderRadius = '10px';
    fig.style.overflow = 'hidden';
    fig.style.width = '180px';
    fig.style.background = 'var(--card, #111)';

    const previewNote = (/dcm|dicom/i.test(file.name) ? '(no preview)' : '');
    fig.innerHTML = `
      <img alt="${file.name}" src="${url}" style="display:block;width:100%;height:130px;object-fit:cover"/>
      <figcaption style="padding:6px 8px;font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${file.name}">
        ${file.name}
      </figcaption>
      <footer style="display:flex;justify-content:space-between;align-items:center;padding:6px 8px;border-top:1px solid var(--border, #2a2a2a)">
        <span class="chip" style="font-size:11px;opacity:.75">
          ${humanExt(file.name) || (file.type||'').replace('image/','').toUpperCase() || 'FILE'} · ${humanSize(file.size)} ${previewNote}
        </span>
        <button type="button" class="del" style="font-size:11px;line-height:1;border:none;background:transparent;color:inherit;cursor:pointer">✕</button>
      </footer>
    `;

    const imgEl = fig.querySelector('img');
    imgEl.addEventListener('error', () => {
      imgEl.style.display = 'none';
      fig.style.width = 'auto';
    });

    fig.querySelector('.del').addEventListener('click', () => {
      const idx = filesState.indexOf(file);
      if (idx > -1) filesState.splice(idx, 1);
      fig.remove();
      setStatus(`${filesState.length} file(s) ready.`);
    });

    gallery?.appendChild(fig);
  }

  function handleFiles(list){
    const allowed = [
      'image/png','image/jpeg','image/jpg','image/webp','image/heic','image/heif',
      'image/tiff','image/x-tiff','image/dicom','image/x-dicom','application/dicom'
    ];
    const okExts = ['png','jpg','jpeg','webp','heic','heif','tif','tiff','dcm','dicom'];

    let added = 0;
    for (const f of list){
      const type = (f.type||'').toLowerCase();
      const ext  = (f.name.split('.').pop()||'').toLowerCase();
      if (!(okExts.includes(ext) || allowed.includes(type))) continue;
      if (MAX_MB && f.size > MAX_MB*1024*1024) {
        setStatus(`"${f.name}" is larger than ${MAX_MB} MB. Skipped.`, 'warn');
        continue;
      }
      filesState.push(f);
      addThumb(f);
      added++;
    }
    if (added) setStatus(`${filesState.length} file(s) ready.`, 'ok');
    else       setStatus(`No acceptable files detected.`, 'warn');
  }

  // Picker & DnD
  dropzone?.addEventListener('click', (e) => {
    const target = e.target;
    if (target.closest && target.closest('.del')) return;
    fileInputEl?.click();
  });
  fileInputEl?.addEventListener('change', (e) => {
    const input = e.target;
    if (!input.files || !input.files.length) return;
    handleFiles(input.files);
    input.value = '';
  });
  ['dragenter','dragover','dragleave','drop'].forEach(ev => {
    document.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); });
  });
  ['dragenter','dragover'].forEach(ev => {
    dropzone?.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropzone.classList.add('drag'); });
  });
  ['dragleave','drop'].forEach(ev => {
    dropzone?.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('drag'); });
  });
  dropzone?.addEventListener('drop', (e) => {
    const dt = e.dataTransfer;
    if (!dt) return;
    const files = dt.files;
    if (files && files.length) handleFiles(files);
  });
  window.addEventListener('paste', (e) => {
    const items = Array.from(e.clipboardData?.items||[]).map(i => i.getAsFile()).filter(Boolean);
    if (items.length) handleFiles(items);
  });

  clearBtn?.addEventListener('click', ()=>{
    filesState.splice(0,filesState.length);
    if (gallery)  gallery.innerHTML='';
    if (fileInputEl) fileInputEl.value='';
    if (resultsEl) resultsEl.innerHTML='';
    setStatus('Cleared.','');
  });

  // ====== Big results (side-by-side) with masked click fallback ======
  function renderResults(maskUrl, overlayUrl){
    if (!resultsEl) return;
    resultsEl.innerHTML = `
      <div class="result-card">
        <div class="result-head">512×512 Masked Input</div>
        <div class="result-body">
          <a id="bigMaskedLink" href="${maskUrl || '#'}" target="_blank" style="display:block">
            <img id="bigMaskedImg" class="result-img" src="${maskUrl || ''}" alt="Masked">
          </a>
        </div>
      </div>
      <div class="result-card">
        <div class="result-head">Highlighted Overlay + Legend</div>
        <div class="result-body">
          <a href="${overlayUrl || '#'}" target="_blank" style="display:block">
            <img class="result-img" src="${overlayUrl || ''}" alt="Overlay">
          </a>
        </div>
      </div>
    `;
    const a  = d.getElementById('bigMaskedLink');
    const im = d.getElementById('bigMaskedImg');
    if (a){
      const url = maskUrl || im?.src || '#';
      a.href = url;
      a.addEventListener('click', (ev) => {
        if (!url || url === '#') {
          ev.preventDefault();
          const s = im?.src;
          if (s) window.open(s, '_blank');
        }
      }, { once: true });
    }
  }

  // ====== Analyze ======
  function normalizePercent(x) {
    if (typeof x !== 'number' || Number.isNaN(x)) return 0;
    return x <= 1 ? x * 100 : x;
  }

  function upsertResultPanel(figEl) {
    let panel = figEl.querySelector('.result-panel');
    if (!panel) {
      panel = d.createElement('div');
      panel.className = 'result-panel';
      panel.style.cssText = 'padding:8px;border-top:1px dashed var(--border,#2a2a2a);display:grid;gap:8px';
      panel.innerHTML = `
        <div class="result-images" style="display:flex;gap:8px;flex-wrap:wrap">
          <figure style="margin:0">
            <figcaption style="font-size:11px;opacity:.75">Masked (512×512)</figcaption>
            <a class="masked-link" href="#" target="_blank" style="display:block">
              <img class="masked-out" alt="Masked"
                   style="display:block;width:160px;height:120px;object-fit:cover;border-radius:6px;border:1px solid var(--border,#2a2a2a)"/>
            </a>
          </figure>
          <figure style="margin:0">
            <figcaption style="font-size:11px;opacity:.75">Overlay + Legend</figcaption>
            <a class="overlay-link" href="#" target="_blank" style="display:block">
              <img class="overlay-out" alt="Overlay"
                   style="display:block;width:160px;height:120px;object-fit:cover;border-radius:6px;border:1px solid var(--border,#2a2a2a)"/>
            </a>
          </figure>
        </div>
        <div class="legend-wrap">
          <div style="font-size:12px;font-weight:600;margin-bottom:4px">Top Predictions</div>
          <ul class="legend-list" style="margin:0;padding-left:16px;font-size:12px"></ul>
        </div>
      `;
      figEl.appendChild(panel);
    }
    return panel;
  }

  analyzeBtn?.addEventListener('click', async () => {
    if (!filesState.length) { setStatus('Please add at least one image.', 'warn'); return; }
    if (!getToken())        { setStatus('Please log in first.', 'warn'); return; }

    try {
      setStatus('Uploading to local API…', '');

      const t0 = performance.now();
      const fd = new FormData();
      const f = filesState[0];
      if (!f) { setStatus('Please add an image.', 'warn'); return; }
      fd.append('file', f, f.name);

      const res = await authFetch(`${API_BASE}/api/analyze`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const data = await res.json(); // { results: [ {prediction, confidence, masked_url, overlay_url, top3:[{label,p}]} ] }

      // ----- BIG TOP PANELS -----
      if (data.results && data.results[0]) {
        const r = data.results[0];
        renderResults(r.masked_input_url || r.masked_url, r.overlay_url);
      }

      // ----- Per-file thumbnails (small cards) -----
      data.results?.forEach((item, i) => {
        const fig = gallery?.children[i];
        if (!fig) return;

        // Badge (headline prediction)
        let badge = fig.querySelector('.badge');
        if (!badge) {
          badge = d.createElement('div');
          badge.className = 'badge';
          badge.style.cssText = 'position:absolute;top:8px;left:8px;background:rgba(0,0,0,.6);border:1px solid var(--border);padding:6px 8px;border-radius:999px;font-size:12px';
          fig.appendChild(badge);
        }
        const pct = normalizePercent(item.confidence);
        badge.textContent = `${item.prediction || 'Result'} (${pct.toFixed(1)}%)`;

        const panel = upsertResultPanel(fig);
        const maskedEl   = panel.querySelector('.masked-out');
        const overlayEl  = panel.querySelector('.overlay-out');
        const maskedA    = panel.querySelector('.masked-link');
        const overlayA   = panel.querySelector('.overlay-link');
        const legendList = panel.querySelector('.legend-list');

        if (maskedEl && item.masked_url)   maskedEl.src = item.masked_url;
        if (overlayEl && item.overlay_url) overlayEl.src = item.overlay_url;
        if (maskedA  && item.masked_url)   maskedA.href = item.masked_url;
        if (overlayA && item.overlay_url)  overlayA.href = item.overlay_url;

        if (legendList) {
          legendList.innerHTML = '';
          (item.top3 || []).forEach(row => {
            const li = d.createElement('li');
            const p = normalizePercent(row.p);
            li.textContent = `${row.label}: ${p.toFixed(1)}%`;
            legendList.appendChild(li);
          });
        }
      });

      setStatus('Analysis complete.', 'ok');

      const elapsed = (performance.now() - t0) / 1000;
      if (window.ED2Metrics && typeof window.ED2Metrics.recordUpload === 'function') {
        window.ED2Metrics.recordUpload({ seconds: elapsed });
      } else {
        window.dispatchEvent(new CustomEvent('upload:success', { detail: { seconds: elapsed } }));
      }
    } catch (e) {
      console.error(e);
      setStatus('Analysis failed. Is the API running and are you logged in?', 'err');
    }
  });

  // ====== Patients (optional panel elsewhere) ======
  async function listPatients(){
    if(!patientsList) return;
    if(!getToken()){ patientsList.textContent='Log in to view patients.'; return; }
    try{
      const res = await authFetch(`${API_BASE}/api/patients`, { method:'GET' });
      if(!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if(!data.length){ patientsList.textContent='No patients yet.'; return; }
      patientsList.innerHTML = data.map(p => `
        <div class="card pad" style="padding:12px">
          <div><strong>${p.name}</strong></div>
          <div class="small muted">DOB: ${p.dob ?? '—'} · Sex: ${p.sex ?? '—'}</div>
          <div class="small">${p.note || ''}</div>
          <div class="small muted">Created: ${new Date(p.created_at).toLocaleString()}</div>
        </div>
      `).join('');
    }catch(err){
      console.error(err);
      patientsList.textContent='Could not load patients.';
    }
  }

  // ====== Init ======
  setLoggedInUI(!!getToken());
  listPatients();
  console.log('ED2 JS loaded.');
});
