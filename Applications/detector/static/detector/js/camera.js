(() => {
  const video = document.getElementById('cameraVideo');
  const poseCanvas = document.getElementById('poseCanvas');
  const poseCtx = poseCanvas.getContext('2d');
  const captureCanvas = document.getElementById('captureCanvas');
  const captureCtx = captureCanvas.getContext('2d', { willReadFrequently: false });
  const startButton = document.getElementById('startButton');
  const uploadButton = document.getElementById('uploadButton');
  const videoFileInput = document.getElementById('videoFileInput');
  const stopButton = document.getElementById('stopButton');
  const resetButton = document.getElementById('resetButton');
  const cameraSelect = document.getElementById('cameraSelect');
  const fpsSelect = document.getElementById('fpsSelect');
  const soundToggle = document.getElementById('soundToggle');
  const placeholder = document.getElementById('cameraPlaceholder');
  const videoStage = document.getElementById('videoStage');
  const fallFlash = document.getElementById('fallFlash');
  const serverChip = document.getElementById('serverChip');
  const statusCard = document.getElementById('statusCard');
  const statusText = document.getElementById('statusText');
  const statusDescription = document.getElementById('statusDescription');
  const probabilityText = document.getElementById('probabilityText');
  const probabilityFill = document.getElementById('probabilityFill');
  const thresholdMark = document.getElementById('thresholdMark');
  const thresholdText = document.getElementById('thresholdText');
  const fallCount = document.getElementById('fallCount');
  const fpsValue = document.getElementById('fpsValue');
  const deviceText = document.getElementById('deviceText');
  const eventList = document.getElementById('eventList');
  const clearLogButton = document.getElementById('clearLogButton');
  const sessionLabel = document.getElementById('sessionLabel');

  let stream = null;
  let videoObjectUrl = null;
  let mediaMode = 'idle'; // idle | camera | video
  let running = false;
  let requestInFlight = false;
  let previousStatus = 'IDLE';
  let audioContext = null;
  let peakProbability = 0;
  let latestFallCount = 0;
  let videoEndLogged = false;
  let scheduleTimer = null;
  let isInitialVideoPlay = false;

  // Accessibility Fix: Gắn aria-live đúng vị trí
  statusCard.removeAttribute('aria-live');
  statusCard.removeAttribute('aria-atomic');
  const statusTextContainer = statusText.parentElement;
  if (statusTextContainer) {
    statusTextContainer.setAttribute('aria-live', 'polite');
    statusTextContainer.setAttribute('aria-atomic', 'true');
  }

  const sessionId = (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`);
  sessionLabel.textContent = `Session: ${sessionId.slice(0, 8)}`;

  function getCookie(name) {
    const entry = document.cookie.split(';').map(v => v.trim()).find(v => v.startsWith(`${name}=`));
    return entry ? decodeURIComponent(entry.split('=').slice(1).join('=')) : '';
  }

  function setServerState(kind, text) {
    serverChip.classList.remove('ready', 'error');
    if (kind) serverChip.classList.add(kind);
    serverChip.querySelector('b').textContent = text;
  }

  async function checkHealth(load = false) {
    try {
      setServerState('', load ? 'Đang tải YOLO + BiGRU…' : 'Đang kiểm tra mô hình');
      const response = await fetch(`/api/health/${load ? '?load=1' : ''}`, { cache: 'no-store' });
      const data = await response.json();
      if (data.model_loaded) {
        const label = data.gpu_name || data.device;
        setServerState('ready', `Model sẵn sàng · ${label}`);
        deviceText.textContent = label;
      } else if (data.load_error) {
        setServerState('error', 'Lỗi tải model');
        addEvent(`Lỗi model: ${data.load_error}`, 'fall');
      } else {
        setServerState('', 'Model sẽ tải khi bắt đầu phân tích');
      }
      const threshold = Number(data.threshold || 0.4577);
      thresholdText.textContent = `${(threshold * 100).toFixed(2)}%`;
      thresholdMark.style.left = `${threshold * 100}%`;
      return data;
    } catch (error) {
      setServerState('error', 'Không kết nối được Django');
      addEvent(`Không kết nối server: ${error.message}`, 'fall');
      return null;
    }
  }

  async function listCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');
    const selected = cameraSelect.value;
    cameraSelect.innerHTML = cameras.length ? '' : '<option value="">Camera mặc định</option>';
    cameras.forEach((camera, index) => {
      const option = document.createElement('option');
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${index + 1}`;
      if (camera.deviceId === selected) option.selected = true;
      cameraSelect.appendChild(option);
    });
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      addEvent('Trình duyệt không hỗ trợ getUserMedia.', 'fall');
      return;
    }

    startButton.disabled = true;
    startButton.classList.add('is-loading');
    uploadButton.disabled = true;

    try {
      stopCurrentMedia(false);
      await resetSession(true);
      const health = await checkHealth(true);
      if (!health || health.load_error) throw new Error(health?.load_error || 'Không tải được model AI.');

      const deviceId = cameraSelect.value;
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 20, max: 30 }
        },
        audio: false
      });
      video.removeAttribute('src');
      video.srcObject = stream;
      video.controls = false;
      await video.play();
      await listCameras();
      resizeCanvases();

      mediaMode = 'camera';
      running = true;
      placeholder.classList.add('hidden');
      startButton.disabled = true;
      uploadButton.disabled = false;
      stopButton.disabled = false;
      addEvent('Đã bật camera và bắt đầu phân tích.', 'normal');
      scheduleNext(0);
    } catch (error) {
      startButton.disabled = false;
      uploadButton.disabled = false;
      addEvent(`Không mở được camera: ${error.message}`, 'fall');
      updateStatus({ status: 'NO_PERSON', persons: [] });
    } finally {
      startButton.classList.remove('is-loading');
    }
  }

  function chooseDemoVideo() {
    videoFileInput.value = '';
    videoFileInput.click();
  }

  async function loadDemoVideo(file) {
    if (!file) return;
    const supportedExtension = /\.(mp4|webm|mov|m4v)$/i.test(file.name);
    if (!file.type.startsWith('video/') && !supportedExtension) {
      addEvent('Định dạng video không được hỗ trợ. Hãy dùng MP4, WebM hoặc MOV.', 'fall');
      return;
    }

    startButton.disabled = true;
    uploadButton.disabled = true;
    try {
      stopCurrentMedia(false);
      await resetSession(true);
      const health = await checkHealth(true);
      if (!health || health.load_error) throw new Error(health?.load_error || 'Không tải được model AI.');

      videoObjectUrl = URL.createObjectURL(file);
      video.srcObject = null;
      video.src = videoObjectUrl;
      video.controls = true;
      video.muted = true;
      videoEndLogged = false;

      await waitForVideoMetadata();
      resizeCanvases();
      mediaMode = 'video';
      running = true;
      placeholder.classList.add('hidden');
      startButton.disabled = false;
      uploadButton.disabled = false;
      stopButton.disabled = false;

      const duration = Number.isFinite(video.duration) ? formatDuration(video.duration) : 'không xác định';
      addEvent(`Đã tải video “${file.name}” · thời lượng ${duration}.`, 'normal');

      isInitialVideoPlay = true;

      try {
        await video.play();
      } catch (_) {
        addEvent('Video đã sẵn sàng. Nhấn nút Play trên video để bắt đầu.', 'normal');
      }
    } catch (error) {
      startButton.disabled = false;
      uploadButton.disabled = false;
      stopButton.disabled = true;
      addEvent(`Không mở được video: ${error.message}`, 'fall');
      stopCurrentMedia(false);
    }
  }

  function waitForVideoMetadata() {
    if (video.readyState >= 1) return Promise.resolve();
    return new Promise((resolve, reject) => {
      const onLoaded = () => { cleanup(); resolve(); };
      const onError = () => { cleanup(); reject(new Error('Trình duyệt không giải mã được video.')); };
      const cleanup = () => {
        video.removeEventListener('loadedmetadata', onLoaded);
        video.removeEventListener('error', onError);
      };
      video.addEventListener('loadedmetadata', onLoaded, { once: true });
      video.addEventListener('error', onError, { once: true });
      video.load();
    });
  }

  function stopCurrentMedia(announce = true) {
    const oldMode = mediaMode;
    running = false;
    requestInFlight = false;
    isInitialVideoPlay = false;
    if (scheduleTimer !== null) {
      window.clearTimeout(scheduleTimer);
      scheduleTimer = null;
    }

    if (stream) stream.getTracks().forEach(track => track.stop());
    stream = null;
    video.pause();
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();
    video.controls = false;

    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    videoObjectUrl = null;
    mediaMode = 'idle';

    poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
    placeholder.classList.remove('hidden');
    startButton.disabled = false;
    uploadButton.disabled = false;
    stopButton.disabled = true;
    videoStage.classList.remove('fall');
    fallFlash.classList.remove('visible');
    setStatusVisual('IDLE', 'CHỜ DỮ LIỆU', 'Hệ thống đã dừng.');

    if (announce && oldMode !== 'idle') {
      addEvent(oldMode === 'video' ? 'Đã dừng video demo.' : 'Đã dừng camera.', 'normal');
    }
  }

  function resizeCanvases() {
    const sourceW = video.videoWidth || 640;
    const sourceH = video.videoHeight || 480;
    const targetW = 640;
    const targetH = Math.round(targetW * sourceH / sourceW);
    captureCanvas.width = targetW;
    captureCanvas.height = targetH;
    poseCanvas.width = targetW;
    poseCanvas.height = targetH;
  }

  function scheduleNext(delay = null) {
    if (!running) return;
    const interval = delay === null ? Number(fpsSelect.value) : delay;
    if (scheduleTimer !== null) window.clearTimeout(scheduleTimer);
    scheduleTimer = window.setTimeout(() => {
      scheduleTimer = null;
      sendFrame();
    }, interval);
  }

  async function sendFrame() {
    if (!running) return;
    if (mediaMode === 'video' && (video.paused || video.ended)) {
      scheduleNext(100);
      return;
    }
    if (requestInFlight || video.readyState < 2) {
      scheduleNext();
      return;
    }

    requestInFlight = true;
    const started = performance.now();
    try {
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      const blob = await new Promise(resolve => captureCanvas.toBlob(resolve, 'image/jpeg', 0.72));
      if (!blob) throw new Error('Không tạo được JPEG từ nguồn hình ảnh.');

      const form = new FormData();
      form.append('image', blob, mediaMode === 'video' ? 'video-frame.jpg' : 'camera.jpg');
      form.append('session_id', sessionId);
      const response = await fetch('/api/predict/', {
        method: 'POST',
        headers: { 'X-CSRFToken': getCookie('csrftoken') },
        body: form
      });
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);

      drawPose(data);
      updateStatus(data);
    } catch (error) {
      addEvent(`Lỗi frame: ${error.message}`, 'fall', true);
    } finally {
      requestInFlight = false;
      const elapsed = performance.now() - started;
      scheduleNext(Math.max(20, Number(fpsSelect.value) - elapsed));
    }
  }

  /**
   * Cập nhật hàm vẽ Canvas hỗ trợ Multi-Person (Đa đối tượng)
   */
  function drawPose(data) {
    poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
    const persons = data.persons || [];
    if (persons.length === 0) return;

    const sx = poseCanvas.width / (data.image_width || 640);
    const sy = poseCanvas.height / (data.image_height || 480);

    const connections = data.connections || [
      [0, 1], [0, 2], [1, 3], [2, 4],
      [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
      [5, 11], [6, 12], [11, 12],
      [11, 13], [13, 15], [12, 14], [14, 16]
    ];

    persons.forEach(person => {
      const isFall = person.is_fall;
      const strokeColor = isFall ? '#ff5267' : '#34d6d1';
      const badgeBg = isFall ? 'rgba(255, 23, 68, 0.9)' : 'rgba(0, 180, 150, 0.9)';
      const jointColor = isFall ? '#ff9aa7' : '#e9ffff';

      // 1. Vẽ Bounding Box & Badge Label cho từng người
      if (person.bbox && person.bbox.length === 4) {
        const x1 = person.bbox[0] * sx;
        const y1 = person.bbox[1] * sy;
        const x2 = person.bbox[2] * sx;
        const y2 = person.bbox[3] * sy;
        const w = x2 - x1;
        const h = y2 - y1;

        // Bounding Box
        poseCtx.strokeStyle = strokeColor;
        poseCtx.lineWidth = isFall ? 3 : 2;
        poseCtx.strokeRect(x1, y1, w, h);

        if (isFall) {
          poseCtx.fillStyle = 'rgba(255, 23, 68, 0.15)';
          poseCtx.fillRect(x1, y1, w, h);
        }

        // Badge Label
        const probPct = Math.round((person.fall_probability || 0) * 100);
        const labelText = `ID: ${person.track_id} | ${person.status} (${probPct}%)`;

        poseCtx.font = 'bold 12px sans-serif';
        const textWidth = poseCtx.measureText(labelText).width;
        const badgeHeight = 20;
        const badgeY = Math.max(0, y1 - badgeHeight);

        poseCtx.fillStyle = badgeBg;
        poseCtx.fillRect(x1, badgeY, textWidth + 10, badgeHeight);

        poseCtx.fillStyle = '#ffffff';
        poseCtx.fillText(labelText, x1 + 5, badgeY + 14);
      }

      // 2. Vẽ đoạn nối Khung xương (Skeleton)
      const kps = person.keypoints;
      if (kps && kps.length > 0) {
        poseCtx.lineWidth = 2.5;
        poseCtx.lineCap = 'round';
        poseCtx.strokeStyle = strokeColor;

        connections.forEach(([a, b]) => {
          const p1 = kps[a];
          const p2 = kps[b];
          if (p1 && p2 && p1.conf > 0.15 && p2.conf > 0.15) {
            poseCtx.beginPath();
            poseCtx.moveTo(p1.x * sx, p1.y * sy);
            poseCtx.lineTo(p2.x * sx, p2.y * sy);
            poseCtx.stroke();
          }
        });

        // 3. Vẽ khớp nối (Joints)
        kps.forEach(pt => {
          if (pt.conf > 0.15) {
            poseCtx.beginPath();
            poseCtx.fillStyle = jointColor;
            poseCtx.arc(pt.x * sx, pt.y * sy, 3.5, 0, Math.PI * 2);
            poseCtx.fill();
          }
        });
      }
    });
  }

  function setStatusVisual(kind, title, description) {
    statusCard.classList.remove('status-idle', 'status-normal', 'status-fall');
    statusCard.classList.add(kind === 'FALL' ? 'status-fall' : kind === 'NORMAL' ? 'status-normal' : 'status-idle');
    statusText.textContent = title;
    statusDescription.textContent = description;
  }

  /**
   * Cập nhật Dashboard UI với thông số Đa đối tượng
   */
  function updateStatus(data) {
    const persons = data.persons || [];
    const personCount = persons.length;

    // Tìm xác suất cao nhất và tổng số vụ té ngã của các đối tượng
    let maxProbability = 0;
    let totalFallCount = 0;
    persons.forEach(p => {
      if (p.fall_probability > maxProbability) maxProbability = p.fall_probability;
      totalFallCount += (p.fall_count || 0);
    });

    const probability = Number(maxProbability);
    const threshold = Number(data.threshold || 0.4577);
    peakProbability = Math.max(peakProbability, probability);
    latestFallCount = totalFallCount;

    probabilityText.textContent = `${(probability * 100).toFixed(1)}%`;
    probabilityFill.style.width = `${Math.min(100, probability * 100)}%`;
    probabilityFill.style.background = probability >= threshold ? '#ff5267' : probability >= threshold * 0.6 ? '#f6bf56' : '#41df8d';
    thresholdMark.style.left = `${threshold * 100}%`;
    thresholdText.textContent = `${(threshold * 100).toFixed(2)}%`;
    fallCount.textContent = latestFallCount;
    fpsValue.textContent = Number(data.server_fps || 0).toFixed(1);
    if (data.device) deviceText.textContent = data.device === 'cuda' ? 'NVIDIA CUDA' : data.device;

    const status = data.status || 'NO_PERSON';
    if (status === 'FALL') {
      const fallPersons = persons.filter(p => p.is_fall);
      const fallIds = fallPersons.map(p => `#${p.track_id}`).join(', ');

      setStatusVisual('FALL', 'FALL DETECTED', `Phát hiện té ngã (${fallPersons.length} người: ID ${fallIds}).`);
      videoStage.classList.add('fall');
      fallFlash.classList.add('visible');

      if (previousStatus !== 'FALL') {
        const timeLabel = mediaMode === 'video' ? ` · mốc ${formatDuration(video.currentTime)}` : '';
        addEvent(`Phát hiện té ngã (ID ${fallIds})${timeLabel} · xác suất ${(probability * 100).toFixed(1)}%`, 'fall');
        beepAlarm();
      }
    } else if (status === 'NORMAL') {
      setStatusVisual('NORMAL', 'NORMAL', `Đã phát hiện ${personCount} người, chưa có dấu hiệu té ngã.`);
      videoStage.classList.remove('fall');
      fallFlash.classList.remove('visible');
      if (previousStatus === 'FALL') addEvent('Trạng thái đã trở lại bình thường.', 'normal');
    } else {
      setStatusVisual('IDLE', 'KHÔNG THẤY NGƯỜI', 'Hãy bảo đảm người nằm trong vùng quan sát.');
      videoStage.classList.remove('fall');
      fallFlash.classList.remove('visible');
    }
    previousStatus = status;
  }

  async function resetSession(silent = false) {
    try {
      const form = new FormData();
      form.append('session_id', sessionId);
      const response = await fetch('/api/reset/', {
        method: 'POST',
        headers: { 'X-CSRFToken': getCookie('csrftoken') },
        body: form
      });
      if (!response.ok) throw new Error('Không đặt lại được phiên phân tích.');
      fallCount.textContent = '0';
      fpsValue.textContent = '—';
      probabilityText.textContent = '0.0%';
      probabilityFill.style.width = '0%';
      previousStatus = 'IDLE';
      peakProbability = 0;
      latestFallCount = 0;
      videoEndLogged = false;
      poseCtx.clearRect(0, 0, poseCanvas.width, poseCanvas.height);
      if (!silent) addEvent('Đã đặt lại phiên phân tích và bộ đếm.', 'normal');
    } catch (error) {
      if (!silent) addEvent(error.message, 'fall');
    }
  }

  function handleVideoEnded() {
    if (mediaMode !== 'video' || videoEndLogged) return;
    running = false;
    if (scheduleTimer !== null) {
      window.clearTimeout(scheduleTimer);
      scheduleTimer = null;
    }
    videoEndLogged = true;
    stopButton.disabled = false;
    videoStage.classList.remove('fall');
    fallFlash.classList.remove('visible');
    setStatusVisual('IDLE', 'VIDEO HOÀN TẤT', `Đã phát hiện ${latestFallCount} sự kiện Fall.`);
    addEvent(`Hoàn tất video · ${latestFallCount} sự kiện Fall · xác suất cao nhất ${(peakProbability * 100).toFixed(1)}%.`, latestFallCount > 0 ? 'fall' : 'normal');
  }

  async function handleVideoPlay() {
    if (mediaMode !== 'video') return;

    if (isInitialVideoPlay) {
      isInitialVideoPlay = false;
    } else if (video.ended || video.currentTime < 0.2) {
      await resetSession(true);
    }

    videoEndLogged = false;
    running = true;
    stopButton.disabled = false;
    scheduleNext(0);
  }

  function formatDuration(seconds) {
    const safeSeconds = Math.max(0, Math.floor(Number(seconds) || 0));
    const minutes = Math.floor(safeSeconds / 60);
    const remainder = safeSeconds % 60;
    return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`;
  }

  function addEvent(message, type = 'normal', dedupe = false) {
    if (dedupe && eventList.firstElementChild?.dataset?.message === message) return;
    const empty = eventList.querySelector('.empty-log');
    if (empty) empty.remove();
    const row = document.createElement('div');
    row.className = `event-item ${type}`;
    row.dataset.message = message;
    const now = new Date().toLocaleTimeString('vi-VN', { hour12: false });
    row.innerHTML = `<i></i><div>${escapeHtml(message)}</div><span>${now}</span>`;
    eventList.prepend(row);
    while (eventList.children.length > 30) eventList.lastElementChild.remove();
  }

  function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = value;
    return div.innerHTML;
  }

  let lastBeepTime = 0;
const BEEP_COOLDOWN_MS = 1000; // Chỉ phát tiếng tít tối đa 1 lần mỗi 3 giây

function beepAlarm() {
  if (!soundToggle.checked) return;

  const now = Date.now();
  if (now - lastBeepTime < BEEP_COOLDOWN_MS) return; // Bỏ qua nếu vừa mới kêu xong
  lastBeepTime = now;

  try {
    audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    [0, .22, .44].forEach(offset => {
      const osc = audioContext.createOscillator();
      const gain = audioContext.createGain();
      osc.frequency.value = 880;
      gain.gain.setValueAtTime(.001, audioContext.currentTime + offset);
      gain.gain.exponentialRampToValueAtTime(.2, audioContext.currentTime + offset + .02);
      gain.gain.exponentialRampToValueAtTime(.001, audioContext.currentTime + offset + .16);
      osc.connect(gain).connect(audioContext.destination);
      osc.start(audioContext.currentTime + offset);
      osc.stop(audioContext.currentTime + offset + .18);
    });
  } catch (_) { /* AudioContext policy */ }
}

  // Gắn các sự kiện điều khiển UI
  startButton.addEventListener('click', startCamera);
  uploadButton.addEventListener('click', chooseDemoVideo);
  videoFileInput.addEventListener('change', event => loadDemoVideo(event.target.files?.[0]));
  stopButton.addEventListener('click', () => stopCurrentMedia(true));
  resetButton.addEventListener('click', () => resetSession(false));
  clearLogButton.addEventListener('click', () => { eventList.innerHTML = '<div class="empty-log">Chưa có sự kiện.</div>'; });
  cameraSelect.addEventListener('change', async () => { if (mediaMode === 'camera') { stopCurrentMedia(false); await startCamera(); } });
  video.addEventListener('ended', handleVideoEnded);
  video.addEventListener('play', handleVideoPlay);
  window.addEventListener('resize', () => { if (video.videoWidth) resizeCanvases(); });
  window.addEventListener('beforeunload', () => stopCurrentMedia(false));

  checkHealth(false);
})();