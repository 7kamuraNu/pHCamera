/* pH Camera - browser side.
 *
 * Grabs frames from the rear camera, crops the region inside the green box,
 * posts it to /api/predict and shows the pH value in real time.  Pressing
 * MEASURE stores the current reading in "Previous Results" (localStorage).
 * When no camera is available (or the page is not served over HTTPS) a photo
 * can be uploaded instead; the center of the photo is used as the ROI.
 */
(function () {
  'use strict';

  var I18N = {
    en: {
      title: 'pH Camera', subtitle: 'Real-time pH scanning', measure: 'MEASURE',
      upload: 'Use a photo', camera: 'Back to camera', details: 'Details', previous: 'Previous Results',
      clear: 'Clear', starting: 'Starting camera…',
      insecure: 'Camera access requires HTTPS (or localhost). Open this page over https://, or use "Use a photo".',
      noCamera: 'Camera not available on this device/browser. You can still measure from a photo.',
      denied: 'Camera permission was denied. Allow camera access in the browser settings or use a photo.',
      aim: 'Put the colored part of the test strip inside the green box.',
      server: 'Server error', classified: 'SVM classification', regression: 'regression',
      outOfRange: 'outside pH 1–11'
    },
    ja: {
      title: 'pH Camera', subtitle: 'リアルタイムpH測定', measure: '測定',
      upload: '写真を使う', camera: 'カメラに戻る', details: '詳細', previous: '測定履歴',
      clear: 'クリア', starting: 'カメラを起動中…',
      insecure: 'カメラの利用にはHTTPS（またはlocalhost）が必要です。https:// で開くか「写真を使う」を選んでください。',
      noCamera: 'この端末/ブラウザではカメラを使えません。写真から測定できます。',
      denied: 'カメラの許可が拒否されました。ブラウザ設定で許可するか写真を使ってください。',
      aim: 'pH試験紙の変色部分を緑の枠に入れてください。',
      server: 'サーバーエラー', classified: 'SVM分類', regression: '回帰',
      outOfRange: 'pH 1–11の範囲外'
    }
  };

  var POLL_MS = 500;        // live update interval
  var MAX_RESULTS = 6;      // swatches kept in "Previous Results"
  var ROI_SEND_PX = 96;     // ROI is resampled to this size before upload

  var $ = function (id) { return document.getElementById(id); };
  var video = $('video'), still = $('still'), roiEl = $('roi'), readout = $('readout'), badge = $('badge');
  var notice = $('notice'), measureBtn = $('measureBtn'), fileInput = $('fileInput'), cameraBtn = $('cameraBtn');
  var detailBtn = $('detailBtn'), detail = $('detail'), resultsEl = $('results'), clearBtn = $('clearBtn');
  var langBtn = $('langBtn'), stage = $('stage'), work = $('work');

  var lang = (new URLSearchParams(location.search).get('lang') ||
              localStorage.getItem('phcam.lang') ||
              (navigator.language || 'en').slice(0, 2));
  if (!I18N[lang]) lang = 'en';

  var state = { source: null, last: null, inflight: false, timer: null, stream: null, results: [] };

  function t(key) { return (I18N[lang] && I18N[lang][key]) || I18N.en[key] || key; }

  function applyLang() {
    document.documentElement.lang = lang;
    document.querySelectorAll('[data-i18n]').forEach(function (el) { el.textContent = t(el.getAttribute('data-i18n')); });
    langBtn.textContent = lang === 'ja' ? 'EN' : 'JA';
    localStorage.setItem('phcam.lang', lang);
    renderResults();
  }

  function showNotice(msg) { notice.textContent = msg; notice.hidden = !msg; }

  /* ---------- geometry: map the on-screen ROI box to source pixels ---------- */
  function sourceSize() {
    if (state.source === 'video') return { w: video.videoWidth, h: video.videoHeight };
    if (state.source === 'still') return { w: still.naturalWidth, h: still.naturalHeight };
    return { w: 0, h: 0 };
  }

  function roiInSource() {
    var src = sourceSize();
    if (!src.w || !src.h) return null;
    var el = stage.getBoundingClientRect();
    var box = roiEl.getBoundingClientRect();
    // object-fit: cover -> uniform scale, centered crop
    var scale = Math.max(el.width / src.w, el.height / src.h);
    var dispW = src.w * scale, dispH = src.h * scale;
    var offX = (el.width - dispW) / 2, offY = (el.height - dispH) / 2;
    var x = (box.left - el.left - offX) / scale;
    var y = (box.top - el.top - offY) / scale;
    var w = box.width / scale, h = box.height / scale;
    // shrink a little so the green border itself is never sampled
    var inset = Math.min(w, h) * 0.08;
    return { x: x + inset, y: y + inset, w: w - 2 * inset, h: h - 2 * inset };
  }

  function grabRoiDataUrl() {
    var r = roiInSource();
    if (!r || r.w < 2 || r.h < 2) return null;
    var el = state.source === 'video' ? video : still;
    work.width = ROI_SEND_PX; work.height = ROI_SEND_PX;
    var ctx = work.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(el, r.x, r.y, r.w, r.h, 0, 0, ROI_SEND_PX, ROI_SEND_PX);
    return work.toDataURL('image/png');   // lossless: no JPEG color shift
  }

  /* ---------- prediction loop ---------- */
  function predictOnce() {
    if (state.inflight || !state.source) return;
    var dataUrl;
    try { dataUrl = grabRoiDataUrl(); } catch (e) { dataUrl = null; }
    if (!dataUrl) return;
    state.inflight = true;
    fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    }).then(function (res) {
      return res.json().then(function (j) { if (!res.ok) throw new Error(j.error || res.statusText); return j; });
    }).then(function (j) {
      state.last = j;
      readout.textContent = j.ph.toFixed(2);
      readout.classList.toggle('out-of-range', !j.in_range);
      var parts = [];
      if (j.source === 'classification') parts.push(t('classified'));
      if (!j.in_range) parts.push(t('outOfRange'));
      badge.textContent = parts.join(' · ');
      badge.hidden = parts.length === 0;
      measureBtn.disabled = false;
      if (!detail.hidden) renderDetail();
      if (notice.textContent === t('server')) showNotice('');
    }).catch(function (err) {
      readout.textContent = '--';
      showNotice(t('server') + ': ' + err.message);
    }).finally(function () { state.inflight = false; });
  }

  function startLoop() {
    stopLoop();
    predictOnce();
    state.timer = setInterval(predictOnce, POLL_MS);
  }
  function stopLoop() { if (state.timer) { clearInterval(state.timer); state.timer = null; } }

  function renderDetail() {
    var j = state.last;
    if (!j) { detail.textContent = ''; return; }
    detail.textContent =
      'pH (shown)       : ' + j.ph.toFixed(2) + '  [' + j.source + ']\n' +
      'regression RHb*  : ' + j.ph_regression.toFixed(2) + '\n' +
      'SVM class        : ' + j.ph_classification.toFixed(1) + '\n' +
      'mean RGB         : ' + j.mean_rgb.join(', ') + '\n' +
      'R / H / b*       : ' + j.raw_features.R + ' / ' + j.raw_features.H + ' / ' + j.raw_features['b*'] + '\n' +
      'ROI (px)         : ' + j.roi_size.join(' x ');
  }

  /* ---------- camera ---------- */
  function startCamera() {
    still.hidden = true; video.hidden = false; cameraBtn.hidden = true;
    state.source = null; measureBtn.disabled = true; readout.textContent = '--'; badge.hidden = true;
    if (!window.isSecureContext) { showNotice(t('insecure')); return Promise.resolve(false); }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { showNotice(t('noCamera')); return Promise.resolve(false); }
    showNotice(t('starting'));
    var constraints = { audio: false, video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } } };
    return navigator.mediaDevices.getUserMedia(constraints).catch(function () {
      return navigator.mediaDevices.getUserMedia({ audio: false, video: true });
    }).then(function (stream) {
      state.stream = stream;
      video.srcObject = stream;
      return video.play().catch(function () { /* autoplay may need a gesture; playsinline+muted normally OK */ });
    }).then(function () {
      return new Promise(function (resolve) {
        if (video.videoWidth) return resolve();
        video.addEventListener('loadedmetadata', function () { resolve(); }, { once: true });
      });
    }).then(function () {
      state.source = 'video';
      showNotice(t('aim'));
      setTimeout(function () { if (notice.textContent === t('aim')) showNotice(''); }, 4000);
      startLoop();
      return true;
    }).catch(function (err) {
      var denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
      showNotice(denied ? t('denied') : (t('noCamera') + ' (' + (err && err.name || err) + ')'));
      return false;
    });
  }

  function stopCamera() {
    stopLoop();
    if (state.stream) { state.stream.getTracks().forEach(function (tr) { tr.stop(); }); state.stream = null; }
    video.srcObject = null;
  }

  /* ---------- still photo fallback ---------- */
  fileInput.addEventListener('change', function () {
    var f = fileInput.files && fileInput.files[0];
    if (!f) return;
    stopCamera();
    var url = URL.createObjectURL(f);
    still.onload = function () {
      URL.revokeObjectURL(url);
      video.hidden = true; still.hidden = false; cameraBtn.hidden = false;
      state.source = 'still';
      showNotice(t('aim'));
      startLoop();
    };
    still.src = url;
    fileInput.value = '';
  });
  cameraBtn.addEventListener('click', function () { startCamera(); });

  /* ---------- results ---------- */
  function loadResults() {
    try { state.results = JSON.parse(localStorage.getItem('phcam.results') || '[]'); } catch (e) { state.results = []; }
  }
  function saveResults() { localStorage.setItem('phcam.results', JSON.stringify(state.results.slice(0, MAX_RESULTS))); }

  function renderResults() {
    resultsEl.innerHTML = '';
    state.results.slice(0, MAX_RESULTS).forEach(function (r) {
      var li = document.createElement('li');
      var chip = document.createElement('span');
      chip.className = 'chip';
      chip.style.background = 'rgb(' + r.rgb.map(Math.round).join(',') + ')';
      var val = document.createElement('span');
      val.className = 'val';
      val.textContent = Number(r.ph).toFixed(1);
      if (r.source === 'classification') {
        var s = document.createElement('small'); s.textContent = ' SVM'; val.appendChild(s);
      }
      li.appendChild(chip); li.appendChild(val);
      li.title = new Date(r.time).toLocaleString() + ' / pH ' + Number(r.ph).toFixed(2);
      resultsEl.appendChild(li);
    });
    clearBtn.hidden = state.results.length === 0;
  }

  measureBtn.addEventListener('click', function () {
    var j = state.last;
    if (!j) return;
    state.results.unshift({ ph: j.ph, rgb: j.mean_rgb, source: j.source, time: Date.now() });
    state.results = state.results.slice(0, MAX_RESULTS);
    saveResults();
    renderResults();
    if (navigator.vibrate) navigator.vibrate(30);
  });
  clearBtn.addEventListener('click', function () { state.results = []; saveResults(); renderResults(); });
  detailBtn.addEventListener('click', function () { detail.hidden = !detail.hidden; renderDetail(); });
  langBtn.addEventListener('click', function () { lang = lang === 'ja' ? 'en' : 'ja'; applyLang(); });

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) { stopLoop(); } else if (state.source) { startLoop(); }
  });

  /* ---------- boot ---------- */
  loadResults();
  applyLang();
  startCamera();

  // exposed for automated tests
  window.__phcam = { state: state, predictOnce: predictOnce, roiInSource: roiInSource };
})();
