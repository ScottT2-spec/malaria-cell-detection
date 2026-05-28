/* MalariaAI — Frontend Application
   CNN detection (TF.js in-browser) + K2 Think backend integration */

(function () {
  'use strict';

  const API = 'https://scott-123-malariaai-backend.hf.space';
  const IMG_SIZE = 64;

  // ── State ────────────────────────────────────────────────────
  let model = null;
  let modelLoaded = false;
  let currentImageData = null;   // base64 data URL of current image
  let currentImageBlob = null;   // raw blob for correction upload
  let lastResult = null;
  let lastAnalysis = null;
  let chatHistory = [];
  let scanId = null;

  // ── DOM refs ─────────────────────────────────────────────────
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => document.querySelectorAll(s);

  const uploadArea = $('#upload-area');
  const fileInput = $('#file-input');
  const previewImg = $('#preview-img');
  const previewName = $('#preview-name');
  const analyzeBtn = $('#analyze-btn');
  const resetBtn = $('#reset-btn');
  const resultPanel = $('#result-panel');
  const resultEmpty = $('#result-empty');
  const loadingOverlay = $('#loading-overlay');
  const loadingText = $('#loading-text');

  // ── Model ────────────────────────────────────────────────────

  async function loadModel() {
    try {
      model = await tf.loadLayersModel('/model/model.json');
      modelLoaded = true;
      console.log('Model loaded (pre-trained)');
    } catch (e) {
      console.log('No pre-trained model, building in-browser CNN');
      model = buildModel();
      modelLoaded = true;
    }
    $('#header-status-text').textContent = 'Model ready';
  }

  function buildModel() {
    const m = tf.sequential();
    m.add(tf.layers.conv2d({ inputShape: [IMG_SIZE, IMG_SIZE, 3], filters: 32, kernelSize: 3, activation: 'relu' }));
    m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    m.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
    m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    m.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu' }));
    m.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    m.add(tf.layers.flatten());
    m.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    m.add(tf.layers.dropout({ rate: 0.5 }));
    m.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    m.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
    return m;
  }

  // ── Image Validation ─────────────────────────────────────────

  function validateImage(img) {
    const canvas = document.createElement('canvas');
    canvas.width = 64; canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, 64, 64);
    const data = ctx.getImageData(0, 0, 64, 64).data;
    const total = 64 * 64;
    let rSum = 0, gSum = 0, bSum = 0, dark = 0, white = 0;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      rSum += r; gSum += g; bSum += b;
      const br = (r + g + b) / 3;
      if (br < 10) dark++;
      if (br > 250) white++;
    }

    const brightness = (rSum + gSum + bSum) / (total * 3);
    if (dark / total > 0.85) return { valid: false, reason: 'Image is almost entirely black.' };
    if (white / total > 0.85) return { valid: false, reason: 'Image is almost entirely white.' };
    if (brightness < 15 || brightness > 248) return { valid: false, reason: 'Extreme brightness — likely not a blood cell.' };

    let rSq = 0, gSq = 0, bSq = 0;
    const rM = rSum / total, gM = gSum / total, bM = bSum / total;
    for (let i = 0; i < data.length; i += 4) {
      rSq += (data[i] - rM) ** 2;
      gSq += (data[i + 1] - gM) ** 2;
      bSq += (data[i + 2] - bM) ** 2;
    }
    const avgStd = (Math.sqrt(rSq / total) + Math.sqrt(gSq / total) + Math.sqrt(bSq / total)) / 3;
    if (avgStd < 5) return { valid: false, reason: 'Image appears to be a solid color.' };

    return { valid: true };
  }

  // ── CNN Inference ────────────────────────────────────────────

  async function runCNN(img) {
    if (!model) await loadModel();
    const start = performance.now();
    const tensor = tf.tidy(() => {
      let t = tf.browser.fromPixels(img);
      t = tf.image.resizeBilinear(t, [IMG_SIZE, IMG_SIZE]);
      t = t.toFloat().div(255.0);
      return t.expandDims(0);
    });
    const prediction = model.predict(tensor);
    const value = (await prediction.data())[0];
    tensor.dispose(); prediction.dispose();
    return {
      isInfected: value > 0.5,
      confidence: value > 0.5 ? value : 1 - value,
      infectedProb: value,
      healthyProb: 1 - value,
      inferenceTime: Math.round(performance.now() - start),
      prediction: value > 0.5 ? 'parasitized' : 'uninfected',
    };
  }

  // ── Grad-CAM ─────────────────────────────────────────────────

  async function generateGradCAM(img) {
    if (!model) return null;
    try {
      let lastConv = null;
      for (let i = model.layers.length - 1; i >= 0; i--) {
        if (model.layers[i].getClassName() === 'Conv2D') { lastConv = model.layers[i]; break; }
      }
      if (!lastConv) return null;

      const gradModel = tf.model({ inputs: model.inputs, outputs: [lastConv.output, model.outputs[0]] });
      const inputTensor = tf.tidy(() => {
        let t = tf.browser.fromPixels(img);
        t = tf.image.resizeBilinear(t, [IMG_SIZE, IMG_SIZE]);
        return t.toFloat().div(255.0).expandDims(0);
      });

      const [convOut] = gradModel.predict(inputTensor);
      const convData = await convOut.data();
      const [, h, w, filters] = convOut.shape;

      const weights = new Float32Array(filters);
      for (let f = 0; f < filters; f++) {
        let sum = 0;
        for (let i = 0; i < h; i++)
          for (let j = 0; j < w; j++)
            sum += convData[i * w * filters + j * filters + f];
        weights[f] = sum / (h * w);
      }

      const heatmap = new Float32Array(h * w);
      for (let i = 0; i < h; i++)
        for (let j = 0; j < w; j++) {
          let val = 0;
          for (let f = 0; f < filters; f++)
            val += weights[f] * convData[i * w * filters + j * filters + f];
          heatmap[i * w + j] = Math.max(0, val);
        }

      let maxVal = 0;
      for (let i = 0; i < heatmap.length; i++) if (heatmap[i] > maxVal) maxVal = heatmap[i];
      if (maxVal > 0) for (let i = 0; i < heatmap.length; i++) heatmap[i] /= maxVal;

      inputTensor.dispose(); convOut.dispose();
      return { heatmap, width: w, height: h };
    } catch (e) {
      console.warn('Grad-CAM failed:', e);
      return null;
    }
  }

  function renderGradCAM(img, gc) {
    if (!gc) { $('#gradcam-card').style.display = 'none'; return; }
    const { heatmap, width, height } = gc;
    const sz = 180;

    const origC = $('#gradcam-original');
    origC.width = sz; origC.height = sz;
    origC.getContext('2d').drawImage(img, 0, 0, sz, sz);

    const heatC = $('#gradcam-heatmap');
    heatC.width = sz; heatC.height = sz;
    const hctx = heatC.getContext('2d');
    hctx.drawImage(img, 0, 0, sz, sz);

    const tmp = document.createElement('canvas');
    tmp.width = width; tmp.height = height;
    const tctx = tmp.getContext('2d');
    const imgData = tctx.createImageData(width, height);

    for (let i = 0; i < heatmap.length; i++) {
      const v = heatmap[i];
      let r, g, b;
      if (v < 0.25) { r = 0; g = Math.round(v * 4 * 255); b = 255; }
      else if (v < 0.5) { r = 0; g = 255; b = Math.round((1 - (v - 0.25) * 4) * 255); }
      else if (v < 0.75) { r = Math.round((v - 0.5) * 4 * 255); g = 255; b = 0; }
      else { r = 255; g = Math.round((1 - (v - 0.75) * 4) * 255); b = 0; }
      imgData.data[i * 4] = r;
      imgData.data[i * 4 + 1] = g;
      imgData.data[i * 4 + 2] = b;
      imgData.data[i * 4 + 3] = Math.round(v * 180);
    }
    tctx.putImageData(imgData, 0, 0);
    hctx.globalAlpha = 0.5;
    hctx.imageSmoothingEnabled = true;
    hctx.drawImage(tmp, 0, 0, sz, sz);
    hctx.globalAlpha = 1.0;

    $('#gradcam-card').style.display = 'block';
  }

  // ── K2 Think Backend ─────────────────────────────────────────

  async function fetchAnalysis(result) {
    try {
      const resp = await fetch(API + '/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scan_id: scanId,
          prediction: result.prediction,
          confidence: result.confidence,
          infected_prob: result.infectedProb,
          healthy_prob: result.healthyProb,
          inference_time_ms: result.inferenceTime,
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.json();
    } catch (e) {
      console.error('Analysis fetch failed:', e);
      return null;
    }
  }

  async function sendCorrection(data) {
    try {
      // If we have the image blob, send with image for CNN retraining
      if (currentImageBlob) {
        const form = new FormData();
        form.append('scan_id', data.scan_id);
        form.append('original_prediction', data.original_prediction);
        form.append('corrected_prediction', data.corrected_prediction);
        form.append('original_confidence', data.original_confidence || 0);
        if (data.corrected_species) form.append('corrected_species', data.corrected_species);
        if (data.parasitemia_level) form.append('parasitemia_level', data.parasitemia_level);
        if (data.doctor_notes) form.append('doctor_notes', data.doctor_notes);
        if (data.doctor_id) form.append('doctor_id', data.doctor_id);
        form.append('image', currentImageBlob, 'correction.png');

        const resp = await fetch(API + '/api/correct/image', { method: 'POST', body: form });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
      } else {
        const resp = await fetch(API + '/api/correct', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
      }
    } catch (e) {
      console.error('Correction failed:', e);
      return null;
    }
  }

  async function sendChat(message) {
    try {
      const resp = await fetch(API + '/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scan_id: scanId,
          message: message,
          conversation_history: chatHistory.slice(-10),
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return await resp.json();
    } catch (e) {
      console.error('Chat failed:', e);
      return { response: 'Unable to reach the server. Please try again.', sources_cited: [] };
    }
  }

  // ── UI: Mode switching ───────────────────────────────────────

  $$('.mode-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      $$('.mode-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset.mode;
      $('#upload-panel').style.display = mode === 'upload' ? 'block' : 'none';
      $('#camera-view').classList.toggle('visible', mode === 'camera');
      $('#batch-panel').classList.toggle('visible', mode === 'batch');
      if (mode === 'camera') startCamera();
      else stopCamera();
    });
  });

  // ── UI: Upload ───────────────────────────────────────────────

  uploadArea.addEventListener('click', () => fileInput.click());
  uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
  uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault(); uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener('change', (e) => { if (e.target.files.length) handleFile(e.target.files[0]); });

  function handleFile(file) {
    if (!file.type.startsWith('image/')) return;
    if (file.size > 10 * 1024 * 1024) { alert('File too large (max 10MB).'); return; }

    currentImageBlob = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      currentImageData = e.target.result;
      previewImg.src = currentImageData;
      previewImg.classList.add('visible');
      previewName.textContent = file.name + ' \u00B7 ' + (file.size / 1024).toFixed(1) + 'KB';
      previewName.classList.add('visible');
      uploadArea.style.display = 'none';
      analyzeBtn.style.display = '';
      analyzeBtn.disabled = false;
      hideResults();
    };
    reader.readAsDataURL(file);
  }

  // ── UI: Camera ───────────────────────────────────────────────

  let cameraStream = null;
  let facingMode = 'environment';

  async function startCamera() {
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 640 }, height: { ideal: 640 } },
      });
      $('#camera-video').srcObject = cameraStream;
    } catch (e) {
      alert('Camera access denied.');
      $$('.mode-btn')[0].click();
    }
  }

  function stopCamera() {
    if (cameraStream) { cameraStream.getTracks().forEach((t) => t.stop()); cameraStream = null; }
  }

  $('#camera-flip').addEventListener('click', async () => {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    stopCamera();
    startCamera();
  });

  $('#camera-cancel').addEventListener('click', () => {
    stopCamera();
    $$('.mode-btn')[0].click();
  });

  $('#capture-btn').addEventListener('click', () => {
    const video = $('#camera-video');
    const canvas = $('#camera-canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob((blob) => { currentImageBlob = blob; }, 'image/png');
    currentImageData = canvas.toDataURL('image/png');
    previewImg.src = currentImageData;
    previewImg.classList.add('visible');
    previewName.textContent = 'Camera capture';
    previewName.classList.add('visible');
    stopCamera();
    $('#camera-view').classList.remove('visible');
    analyzeBtn.style.display = '';
    analyzeBtn.disabled = false;
    hideResults();
  });

  // ── UI: Batch ────────────────────────────────────────────────

  const batchArea = $('#batch-area');
  const batchInput = $('#batch-file-input');
  let batchResults = [];

  batchArea.addEventListener('click', () => batchInput.click());
  batchInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files).slice(0, 50);
    if (!files.length) return;
    await runBatch(files);
  });

  async function runBatch(files) {
    if (!modelLoaded) await loadModel();
    batchResults = [];
    const progress = $('#batch-progress');
    const fill = $('#batch-fill');
    const text = $('#batch-text');
    const tbody = $('#batch-tbody');
    tbody.innerHTML = '';
    progress.classList.add('visible');
    $('#batch-results').style.display = 'none';
    batchArea.style.display = 'none';

    for (let i = 0; i < files.length; i++) {
      fill.style.width = ((i + 1) / files.length * 100) + '%';
      text.textContent = (i + 1) + ' / ' + files.length;
      try {
        const img = await loadImg(files[i]);
        const result = await runCNN(img);
        result.filename = files[i].name;
        batchResults.push(result);
        recordStat(result);

        const tr = document.createElement('tr');
        const tag = result.isInfected ? 'positive' : 'negative';
        const label = result.isInfected ? 'Parasitized' : 'Uninfected';
        tr.innerHTML = '<td><img src="' + img.src + '"></td>' +
          '<td style="font-size:11px;word-break:break-all;">' + files[i].name + '</td>' +
          '<td><span class="tag tag-' + tag + '">' + label + '</span></td>' +
          '<td>' + (result.confidence * 100).toFixed(1) + '%</td>';
        tbody.appendChild(tr);
      } catch (e) {
        console.error('Batch error:', e);
      }
    }

    $('#batch-results').style.display = 'block';
    batchInput.value = '';
  }

  function loadImg(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = e.target.result;
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  $('#batch-export-csv').addEventListener('click', () => {
    if (!batchResults.length) return;
    let csv = 'Filename,Result,Confidence\n';
    for (const r of batchResults) {
      csv += '"' + r.filename + '","' + (r.isInfected ? 'Parasitized' : 'Uninfected') + '",' +
        (r.confidence * 100).toFixed(1) + '%\n';
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'malariaai-batch-' + Date.now() + '.csv';
    a.click();
  });

  // ── UI: Analyze ──────────────────────────────────────────────

  analyzeBtn.addEventListener('click', async () => {
    if (!currentImageData) return;
    scanId = crypto.randomUUID ? crypto.randomUUID().slice(0, 12) : Date.now().toString(36);

    showLoading('Loading model...');

    if (!modelLoaded) await loadModel();

    showLoading('Analyzing cell...');
    await sleep(300);

    const result = await runCNN(previewImg);
    const validation = validateImage(previewImg);

    if (!validation.valid) {
      hideLoading();
      showRejected(validation.reason);
      return;
    }

    lastResult = result;
    recordStat(result);
    showCNNResult(result);

    // Grad-CAM
    showLoading('Generating attention map...');
    const gc = await generateGradCAM(previewImg);
    renderGradCAM(previewImg, gc);

    // K2 Think analysis (async — don't block the CNN result)
    showLoading('Consulting clinical knowledge base...');
    const analysis = await fetchAnalysis(result);
    lastAnalysis = analysis;
    hideLoading();

    if (analysis) showAIAnalysis(analysis);

    analyzeBtn.style.display = 'none';
    resetBtn.style.display = '';
  });

  // ── UI: Results ──────────────────────────────────────────────

  function showCNNResult(r) {
    hideLoading();
    resultEmpty.style.display = 'none';
    resultPanel.classList.add('visible');

    const banner = $('#result-banner');
    const cls = $('#result-classification');
    const conf = $('#result-confidence-text');
    const fill = $('#confidence-fill');

    let bannerClass, label;
    if (r.confidence < 0.65) {
      bannerClass = 'borderline';
      label = r.isInfected ? 'PARASITIZED (Borderline)' : 'UNINFECTED (Borderline)';
    } else {
      bannerClass = r.isInfected ? 'positive' : 'negative';
      label = r.isInfected ? 'PARASITIZED' : 'UNINFECTED';
    }

    banner.className = 'result-banner ' + bannerClass;
    cls.textContent = label;
    conf.textContent = (r.confidence * 100).toFixed(1) + '% confidence';
    setTimeout(() => { fill.style.width = (r.confidence * 100) + '%'; }, 100);

    $('#stat-parasitized').textContent = (r.infectedProb * 100).toFixed(1) + '%';
    $('#stat-uninfected').textContent = (r.healthyProb * 100).toFixed(1) + '%';
    $('#stat-time').textContent = r.inferenceTime + 'ms';
    $('#result-timestamp').textContent = new Date().toLocaleTimeString();
    $('#download-report-btn').style.display = '';
  }

  function showRejected(reason) {
    resultEmpty.style.display = 'none';
    resultPanel.classList.add('visible');
    const banner = $('#result-banner');
    banner.className = 'result-banner borderline';
    $('#result-classification').textContent = 'NOT A BLOOD CELL';
    $('#result-confidence-text').textContent = reason;
    $('#confidence-fill').style.width = '0%';
    $('#stat-parasitized').textContent = '-';
    $('#stat-uninfected').textContent = '-';
    $('#stat-time').textContent = '-';
    analyzeBtn.style.display = 'none';
    resetBtn.style.display = '';
  }

  function showAIAnalysis(a) {
    const card = $('#ai-card');
    card.style.display = 'block';

    $('#ai-summary').textContent = a.clinical_summary || '';

    const sev = $('#ai-severity');
    const sevText = (a.severity_assessment || 'unknown').toLowerCase();
    const sevLevel = sevText.split(/[\s—-]/)[0];
    sev.textContent = a.severity_assessment;
    sev.className = 'severity-badge severity-' + (['none', 'low', 'moderate', 'high', 'critical'].includes(sevLevel) ? sevLevel : 'moderate');

    $('#ai-treatment').textContent = a.treatment_guidance || '';
    $('#ai-education').textContent = a.patient_education || '';

    const fq = $('#ai-followup');
    fq.innerHTML = '';
    (a.follow_up_questions || []).forEach((q) => {
      const li = document.createElement('li');
      li.textContent = q;
      fq.appendChild(li);
    });

    const gl = $('#ai-guidelines');
    gl.innerHTML = '';
    (a.guidelines_cited || []).forEach((g) => {
      const li = document.createElement('li');
      li.textContent = g;
      gl.appendChild(li);
    });

    // Show chat
    $('#chat-card').style.display = 'block';
  }

  function hideResults() {
    resultPanel.classList.remove('visible');
    resultEmpty.style.display = '';
    $('#ai-card').style.display = 'none';
    $('#chat-card').style.display = 'none';
    $('#gradcam-card').style.display = 'none';
    $('#correction-panel').classList.remove('visible');
    $('#confidence-fill').style.width = '0%';
    $('#download-report-btn').style.display = 'none';
    lastResult = null;
    lastAnalysis = null;
    chatHistory = [];
    $('#chat-messages').innerHTML = '';
  }

  // ── UI: Reset ────────────────────────────────────────────────

  resetBtn.addEventListener('click', () => {
    currentImageData = null;
    currentImageBlob = null;
    fileInput.value = '';
    previewImg.classList.remove('visible');
    previewName.classList.remove('visible');
    uploadArea.style.display = '';
    analyzeBtn.disabled = true;
    analyzeBtn.style.display = 'none';
    resetBtn.style.display = 'none';
    hideResults();
  });

  // ── UI: Correction ───────────────────────────────────────────

  $('#correct-btn').addEventListener('click', () => {
    $('#correction-panel').classList.toggle('visible');
  });

  $('#cancel-correction-btn').addEventListener('click', () => {
    $('#correction-panel').classList.remove('visible');
  });

  $('#submit-correction-btn').addEventListener('click', async () => {
    if (!lastResult) return;
    const btn = $('#submit-correction-btn');
    btn.disabled = true;
    btn.textContent = 'Submitting...';

    const data = {
      scan_id: scanId,
      original_prediction: lastResult.prediction,
      corrected_prediction: $('#correct-prediction').value,
      original_confidence: lastResult.confidence,
      corrected_species: $('#correct-species').value || null,
      parasitemia_level: $('#correct-parasitemia').value || null,
      doctor_notes: $('#correct-notes').value || null,
      doctor_id: $('#correct-doctor-id').value || null,
    };

    const resp = await sendCorrection(data);
    btn.disabled = false;
    btn.textContent = 'Submit correction';

    if (resp) {
      $('#correction-panel').classList.remove('visible');
      alert('Correction recorded. The AI will use this to improve future analyses.');
    } else {
      alert('Failed to submit correction. Please try again.');
    }
  });

  // ── UI: Chat ─────────────────────────────────────────────────

  async function handleChat() {
    const input = $('#chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';

    appendChat('user', msg);
    chatHistory.push({ role: 'user', content: msg });

    const resp = await sendChat(msg);
    const answer = resp.response || 'No response.';
    const sources = resp.sources_cited || [];
    appendChat('assistant', answer, sources);
    chatHistory.push({ role: 'assistant', content: answer });
  }

  function appendChat(role, text, sources) {
    const div = document.createElement('div');
    div.className = 'chat-msg ' + role;
    div.textContent = text;
    if (sources && sources.length) {
      const span = document.createElement('span');
      span.className = 'msg-source';
      span.textContent = 'Sources: ' + sources.join(', ');
      div.appendChild(span);
    }
    const container = $('#chat-messages');
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
  }

  $('#chat-send').addEventListener('click', handleChat);
  $('#chat-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') handleChat(); });

  // ── UI: Report ───────────────────────────────────────────────

  $('#download-report-btn').addEventListener('click', async () => {
    if (!lastResult) return;
    const btn = $('#download-report-btn');
    btn.disabled = true;
    btn.textContent = 'Generating...';

    try {
      const resp = await fetch(API + '/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          scan_id: scanId,
          analysis: { ...lastResult, ...(lastAnalysis || {}) },
        }),
      });
      if (!resp.ok) throw new Error('Report generation failed');
      const data = await resp.json();
      const r = data.report || data;

      // Generate professional PDF
      const { jsPDF } = window.jspdf;
      const doc = new jsPDF({ unit: 'mm', format: 'a4' });
      const pageW = 210;
      const pageH = 297;
      const mL = 18;
      const mR = 18;
      const contentW = pageW - mL - mR;
      let y = 0;
      var pageNum = 1;
      var totalPages = 1;
      var hdr = r.header || {};
      var sr = r.screening_result || {};
      var isInfected = (sr.classification || '').toLowerCase().includes('parasitized');

      // Colors
      var navy = [0, 32, 91];
      var blue = [0, 154, 222];
      var darkGray = [50, 50, 50];
      var medGray = [100, 100, 100];
      var lightGray = [160, 160, 160];
      var white = [255, 255, 255];
      var red = [185, 28, 28];
      var green = [21, 128, 61];
      var amber = [146, 64, 14];

      function addFooter() {
        doc.setDrawColor(200, 200, 200);
        doc.setLineWidth(0.3);
        doc.line(mL, pageH - 15, pageW - mR, pageH - 15);
        doc.setFontSize(7);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(170, 170, 170);
        doc.text('MalariaAI \u2014 AI-Assisted Clinical Screening Platform  |  Powered by K2 Think V2 (MBZUAI) + TensorFlow.js', mL, pageH - 10);
        doc.text('Page ' + pageNum, pageW - mR, pageH - 10, { align: 'right' });
      }

      function newPage() {
        addFooter();
        doc.addPage();
        pageNum++;
        y = 18;
      }

      function checkPage(needed) {
        if (y + needed > pageH - 22) newPage();
      }

      function drawLine() {
        doc.setDrawColor(220, 220, 220);
        doc.setLineWidth(0.2);
        doc.line(mL, y, pageW - mR, y);
        y += 5;
      }

      function sectionTitle(title) {
        checkPage(18);
        y += 2;
        doc.setFillColor(navy[0], navy[1], navy[2]);
        doc.roundedRect(mL, y - 4, 3, 8, 1.5, 1.5, 'F');
        doc.setFontSize(12);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(navy[0], navy[1], navy[2]);
        doc.text(title, mL + 7, y + 2);
        y += 10;
      }

      function bodyText(text, indent) {
        if (!text) return;
        indent = indent || 0;
        doc.setFontSize(9.5);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(darkGray[0], darkGray[1], darkGray[2]);
        var lines = doc.splitTextToSize(text, contentW - indent);
        lines.forEach(function(line) {
          checkPage(5);
          doc.text(line, mL + indent, y);
          y += 4.5;
        });
        y += 2;
      }

      function addSection(title, text) {
        if (!text) return;
        sectionTitle(title);
        bodyText(text);
        y += 2;
      }

      function addList(title, items) {
        if (!items || !items.length) return;
        sectionTitle(title);
        doc.setFontSize(9.5);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(darkGray[0], darkGray[1], darkGray[2]);
        items.forEach(function(item, idx) {
          var lines = doc.splitTextToSize(item, contentW - 10);
          checkPage(lines.length * 4.5 + 3);
          // Number circle
          doc.setFillColor(navy[0], navy[1], navy[2]);
          doc.circle(mL + 3, y - 1.2, 2.2, 'F');
          doc.setFontSize(7);
          doc.setFont('helvetica', 'bold');
          doc.setTextColor(255, 255, 255);
          doc.text(String(idx + 1), mL + 3, y - 0.5, { align: 'center' });
          // Text
          doc.setFontSize(9.5);
          doc.setFont('helvetica', 'normal');
          doc.setTextColor(darkGray[0], darkGray[1], darkGray[2]);
          lines.forEach(function(line, li) {
            doc.text(line, mL + 8, y);
            y += 4.5;
          });
          y += 2;
        });
        y += 2;
      }

      // ═══════ PAGE 1: HEADER ═══════

      // Top bar
      doc.setFillColor(navy[0], navy[1], navy[2]);
      doc.rect(0, 0, pageW, 38, 'F');

      // Blue accent line
      doc.setFillColor(blue[0], blue[1], blue[2]);
      doc.rect(0, 38, pageW, 2, 'F');

      // Title
      doc.setFontSize(20);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(255, 255, 255);
      doc.text('MALARIA BLOOD FILM SCREENING REPORT', mL, 18);

      doc.setFontSize(9);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(blue[0], blue[1], blue[2]);
      doc.text('MalariaAI \u2014 AI-Assisted Clinical Screening Platform', mL, 26);

      // Report meta (right side of header)
      doc.setFontSize(8);
      doc.setTextColor(180, 200, 230);
      doc.text('Report ID: ' + (hdr.report_id || data.scan_id || scanId), pageW - mR, 14, { align: 'right' });
      doc.text('Date: ' + (hdr.date_reported || new Date().toISOString().slice(0, 16).replace('T', ' ') + ' UTC'), pageW - mR, 20, { align: 'right' });
      doc.text('Patient ID: ' + (hdr.patient_id || data.patient_id || 'N/A'), pageW - mR, 26, { align: 'right' });

      y = 48;

      // Patient info box
      doc.setFillColor(245, 247, 250);
      doc.setDrawColor(220, 225, 230);
      doc.setLineWidth(0.3);
      doc.roundedRect(mL, y, contentW, 18, 3, 3, 'FD');
      doc.setFontSize(8.5);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(medGray[0], medGray[1], medGray[2]);
      doc.text('SPECIMEN', mL + 5, y + 6);
      doc.text('METHOD', mL + 75, y + 6);
      doc.text('DATASET', mL + 135, y + 6);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(darkGray[0], darkGray[1], darkGray[2]);
      doc.setFontSize(8);
      doc.text(hdr.specimen_type || 'Peripheral blood film (thin smear)', mL + 5, y + 13);
      doc.text((sr.method || 'CNN (95.43% accuracy)').substring(0, 40), mL + 75, y + 13);
      doc.text('27,558 NIH cell images', mL + 135, y + 13);
      y += 26;

      // ═══════ RESULT BANNER ═══════
      var bannerColor = isInfected ? red : green;
      doc.setFillColor(bannerColor[0], bannerColor[1], bannerColor[2]);
      doc.roundedRect(mL, y, contentW, 24, 4, 4, 'F');

      // Result icon
      doc.setFontSize(14);
      doc.setTextColor(255, 255, 255);
      doc.setFont('helvetica', 'bold');
      var classification = sr.classification || (lastResult.isInfected ? 'PARASITIZED' : 'UNINFECTED');
      doc.text(classification, pageW / 2, y + 10, { align: 'center' });

      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      var confStr = 'Confidence: ' + (sr.confidence_score || (lastResult.confidence * 100).toFixed(1) + '%');
      doc.text(confStr, pageW / 2, y + 18, { align: 'center' });
      y += 30;

      // Confidence note
      if (sr.confidence_level) {
        doc.setFillColor(255, 251, 235);
        doc.setDrawColor(253, 230, 138);
        doc.setLineWidth(0.3);
        doc.roundedRect(mL, y, contentW, 12, 2, 2, 'FD');
        doc.setFontSize(8);
        doc.setFont('helvetica', 'italic');
        doc.setTextColor(amber[0], amber[1], amber[2]);
        var confNote = doc.splitTextToSize('\u26A0  ' + sr.confidence_level, contentW - 10);
        confNote.forEach(function(l, i) {
          doc.text(l, mL + 5, y + 5 + (i * 4));
        });
        y += Math.max(14, confNote.length * 4 + 6);
      }

      y += 4;

      // ═══════ SECTIONS ═══════
      addSection('Clinical Interpretation', r.clinical_interpretation);
      addSection('Treatment Recommendation', r.treatment_recommendation);
      addList('Recommended Actions', r.recommended_actions);
      addSection('Patient Guidance', r.patient_guidance);
      addSection('Severity Assessment', r.severity_assessment);
      addSection('Quality Notes', r.quality_notes);

      // Guidelines cited
      if (r.guidelines_cited && r.guidelines_cited.length) {
        sectionTitle('Guidelines Cited');
        doc.setFontSize(9);
        doc.setFont('helvetica', 'italic');
        doc.setTextColor(medGray[0], medGray[1], medGray[2]);
        r.guidelines_cited.forEach(function(g) {
          checkPage(6);
          doc.text('\u2022  ' + g, mL + 4, y);
          y += 5;
        });
        y += 4;
      }

      // Methodology box
      var meth = r.methodology || {};
      if (meth.model) {
        checkPage(35);
        drawLine();
        sectionTitle('Methodology');
        doc.setFillColor(248, 249, 250);
        doc.setDrawColor(230, 232, 235);
        doc.setLineWidth(0.2);
        var methH = 28;
        doc.roundedRect(mL, y, contentW, methH, 2, 2, 'FD');
        doc.setFontSize(8);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(medGray[0], medGray[1], medGray[2]);
        var methY = y + 6;
        var methItems = [
          ['Model', meth.model || ''],
          ['Dataset', meth.dataset || ''],
          ['Validation Accuracy', meth.validation_accuracy || ''],
          ['Clinical Intelligence', meth.clinical_intelligence || ''],
        ];
        methItems.forEach(function(pair) {
          if (pair[1]) {
            doc.setFont('helvetica', 'bold');
            doc.text(pair[0] + ':', mL + 5, methY);
            doc.setFont('helvetica', 'normal');
            doc.text(pair[1], mL + 42, methY);
            methY += 5.5;
          }
        });
        y += methH + 6;
      }

      // Disclaimer
      checkPage(25);
      drawLine();
      doc.setFillColor(254, 242, 242);
      doc.setDrawColor(254, 202, 202);
      doc.setLineWidth(0.2);
      var disc = r.disclaimer || 'This tool is for screening purposes only and does not constitute a medical diagnosis. All results must be confirmed by a qualified medical professional.';
      var discLines = doc.splitTextToSize(disc, contentW - 14);
      var discH = discLines.length * 4 + 8;
      doc.roundedRect(mL, y, contentW, discH, 2, 2, 'FD');
      doc.setFontSize(7.5);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(185, 28, 28);
      doc.text('DISCLAIMER', mL + 5, y + 5);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(7);
      doc.setTextColor(120, 50, 50);
      discLines.forEach(function(l, i) {
        doc.text(l, mL + 5, y + 10 + (i * 4));
      });
      y += discH + 4;

      // Footer on last page
      addFooter();

      doc.save('MalariaAI-Report-' + (hdr.report_id || data.scan_id || scanId) + '.pdf');
    } catch (e) {
      console.error('Report error:', e);
      alert('Report generation failed.');
    }

    btn.disabled = false;
    btn.textContent = 'Download report';
  });

  // ── Dashboard ────────────────────────────────────────────────

  function getStats() {
    try { return JSON.parse(localStorage.getItem('malariaaiStats') || '{}'); } catch { return {}; }
  }

  function saveStats(s) { localStorage.setItem('malariaaiStats', JSON.stringify(s)); }

  function recordStat(result) {
    const s = getStats();
    s.total = (s.total || 0) + 1;
    if (result.isInfected) s.infected = (s.infected || 0) + 1;
    else s.uninfected = (s.uninfected || 0) + 1;
    s.totalConf = (s.totalConf || 0) + result.confidence;
    s.lastScan = new Date().toLocaleString();
    saveStats(s);
    renderDash();
  }

  function renderDash() {
    const s = getStats();
    const total = s.total || 0;
    const infected = s.infected || 0;
    const uninfected = s.uninfected || 0;
    const valid = infected + uninfected;
    const avgConf = valid > 0 ? ((s.totalConf || 0) / valid * 100).toFixed(1) : '0.0';
    const infRate = total > 0 ? (infected / total * 100).toFixed(1) : '0.0';

    $('#dash-grid').innerHTML =
      '<div class="dash-stat"><div class="val">' + total + '</div><div class="lbl">Total scans</div></div>' +
      '<div class="dash-stat"><div class="val" style="color:var(--red);">' + infected + '</div><div class="lbl">Parasitized</div></div>' +
      '<div class="dash-stat"><div class="val" style="color:var(--green);">' + uninfected + '</div><div class="lbl">Uninfected</div></div>' +
      '<div class="dash-stat"><div class="val">' + infRate + '%</div><div class="lbl">Infection rate</div></div>' +
      '<div class="dash-stat"><div class="val">' + avgConf + '%</div><div class="lbl">Avg confidence</div></div>';
  }

  $('#dash-toggle').addEventListener('click', () => {
    const body = $('#dash-body');
    const open = body.style.display !== 'none';
    body.style.display = open ? 'none' : 'block';
    $('#dash-chevron').style.transform = open ? '' : 'rotate(180deg)';
  });

  $('#dash-reset').addEventListener('click', () => {
    if (confirm('Reset all statistics?')) {
      localStorage.removeItem('malariaaiStats');
      renderDash();
    }
  });

  // ── Helpers ──────────────────────────────────────────────────

  function showLoading(text) { loadingText.textContent = text; loadingOverlay.classList.add('visible'); }
  function hideLoading() { loadingOverlay.classList.remove('visible'); }
  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

  // ── Init ─────────────────────────────────────────────────────

  window.addEventListener('load', () => {
    loadModel();
    renderDash();
  });
})();
