const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');
const liveCounter = document.getElementById('live-counter');
const statusCamera = document.getElementById('status-camera');
const statusDetection = document.getElementById('status-detection');
const statusPosture = document.getElementById('status-posture');
const feedbackText = document.getElementById('feedback-text');
const startCameraBtn = document.getElementById('start-camera');
const stopCameraBtn = document.getElementById('stop-camera');
const startDetectBtn = document.getElementById('start-detect');
const resetCountBtn = document.getElementById('reset-count');
const countdownOverlay = document.getElementById('countdown-overlay');

const STATE = {
  stream: null,
  detector: {
    running: false,
    pushCount: 0
  }
};

let detector = null;
let rafId = null;
let baseline = null;
let lastPosition = 'up';
let lastRepTime = 0;
let lastDownTime = 0;

const COUNTDOWN_SECONDS = 10;
const MIN_REP_INTERVAL_MS = 800;
const MIN_DOWN_HOLD_MS = 250;
const DOWN_ANGLE_THRESHOLD = 90;
const UP_ANGLE_THRESHOLD = 160;
const SHOULDER_DROP_RATIO = 0.18;

function init() {
  startCameraBtn.addEventListener('click', startCamera);
  stopCameraBtn.addEventListener('click', stopCamera);
  startDetectBtn.addEventListener('click', startOrStopDetection);
  resetCountBtn.addEventListener('click', () => { STATE.detector.pushCount = 0; updateUI(); feedbackText.textContent = 'Contagem zerada.'; });
  updateUI();
}

function updateUI() {
  liveCounter.textContent = STATE.detector.pushCount;
  statusCamera.textContent = STATE.stream ? '✅' : '❌';
  statusDetection.textContent = STATE.detector.running ? '✅' : '❌';
  statusPosture.textContent = (baseline && STATE.detector.running) ? '✅' : (STATE.detector.running ? '...' : '-');
  startDetectBtn.textContent = STATE.detector.running ? 'Parar Detecção' : 'Iniciar Detecção';
  startDetectBtn.classList.toggle('btn-active', STATE.detector.running);
}

async function startCamera() {
  try {
    STATE.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    video.srcObject = STATE.stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    feedbackText.textContent = 'Câmera ativa. Posicione-se para iniciar a deteção.';
    updateUI();
  } catch (err) {
    feedbackText.textContent = 'Erro ao aceder à câmara: ' + err.message;
    console.error(err);
  }
}

function stopCamera() {
  stopDetection();
  if (STATE.stream) {
    STATE.stream.getTracks().forEach(t => t.stop());
    STATE.stream = null;
  }
  video.pause();
  video.srcObject = null;
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  feedbackText.textContent = 'Câmara desativada.';
  updateUI();
}

async function initPoseModel() {
  feedbackText.textContent = 'A carregar modelo (MoveNet) — aguarda alguns segundos...';
  await tf.ready();
  await tf.setBackend('webgl');
  detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, { modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING });
  feedbackText.textContent = 'Modelo carregado. Pronto para calibrar.';
}

async function startOrStopDetection() {
  if (!STATE.stream) { feedbackText.textContent = 'Abre a câmara primeiro.'; return; }
  if (STATE.detector.running) { stopDetection(); return; }
  if (!detector) await initPoseModel();
  await startCountdownAndRun();
}

function stopDetection() {
  STATE.detector.running = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  baseline = null;
  lastPosition = 'up';
  updateUI();
  feedbackText.textContent = 'Detecção parada.';
}

function showCountdown(n) {
  countdownOverlay.textContent = n > 0 ? n : 'COMEÇAR';
  countdownOverlay.classList.add('active');
}

function hideCountdown() {
  countdownOverlay.classList.remove('active');
  countdownOverlay.textContent = '';
}

async function startCountdownAndRun() {
  STATE.detector.running = false;
  updateUI();
  let t = COUNTDOWN_SECONDS;
  showCountdown(t);
  feedbackText.textContent = 'Preparando...';
  while (t > 0) {
    await new Promise(res => setTimeout(res, 1000));
    t--;
    showCountdown(t);
  }
  await new Promise(res => setTimeout(res, 300));
  hideCountdown();
  const ok = await calibrateBaseline();
  if (!ok) { feedbackText.textContent = 'Falha na calibração — tenta posicionar-te melhor.'; return; }
  STATE.detector.running = true;
  updateUI();
  lastPosition = 'up';
  lastRepTime = 0;
  feedbackText.textContent = 'Detecção iniciada. Executa flexões com postura correta.';
  runDetectionLoop();
}

async function calibrateBaseline() {
  const tries = 6;
  let collected = [];
  for (let i = 0; i < tries; i++) {
    const poses = await detector.estimatePoses(video);
    if (poses && poses.length > 0) {
      const p = poses[0];
      const s = getMidpointY(p, 5, 6);
      const hip = getMidpointY(p, 11, 12);
      const shoulderWidth = Math.abs(getKey(p,5).x - getKey(p,6).x) || 1;
      if (s && hip && shoulderWidth > 10) collected.push({ shoulderY: s, hipY: hip, shoulderWidth });
    }
    await new Promise(res => setTimeout(res, 120));
  }
  if (collected.length === 0) return false;
  const avgShoulderY = collected.reduce((a,b)=>a+b.shoulderY,0)/collected.length;
  const avgHipY = collected.reduce((a,b)=>a+b.hipY,0)/collected.length;
  const avgShoulderW = collected.reduce((a,b)=>a+b.shoulderWidth,0)/collected.length;
  const bodyHeight = Math.max(20, Math.abs(avgHipY - avgShoulderY));
  baseline = { shoulderY: avgShoulderY, hipY: avgHipY, shoulderWidth: avgShoulderW, bodyHeight };
  baseline.shoulderDropThreshold = baseline.bodyHeight * SHOULDER_DROP_RATIO;
  return true;
}

function getKey(pose, idx) {
  const k = pose.keypoints[idx];
  return { x: k.x, y: k.y, score: k.score ?? k.probability ?? 0 };
}

function getMidpointY(pose, aIdx, bIdx) {
  const a = getKey(pose, aIdx);
  const b = getKey(pose, bIdx);
  if (a.score < 0.35 || b.score < 0.35) return null;
  return (a.y + b.y) / 2;
}

function angleDeg(a, b, c) {
  const abx = a.x - b.x;
  const aby = a.y - b.y;
  const cbx = c.x - b.x;
  const cby = c.y - b.y;
  const dot = abx * cbx + aby * cby;
  const mag1 = Math.hypot(abx, aby);
  const mag2 = Math.hypot(cbx, cby);
  if (mag1 === 0 || mag2 === 0) return 180;
  const cos = Math.max(-1, Math.min(1, dot / (mag1 * mag2)));
  return Math.acos(cos) * 180 / Math.PI;
}

function drawPose(pose) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  overlayCtx.lineWidth = Math.max(2, Math.round(overlay.width / 280));
  const keypoints = pose.keypoints;
  const strokeLine = (a,b, color) => {
    const A = keypoints[a], B = keypoints[b];
    if ((A.score ?? 0) < 0.35 || (B.score ?? 0) < 0.35) return;
    overlayCtx.beginPath();
    overlayCtx.moveTo(A.x, A.y);
    overlayCtx.lineTo(B.x, B.y);
    overlayCtx.strokeStyle = color;
    overlayCtx.stroke();
  };
  const drawPoint = (i, color, r=4) => {
    const p = keypoints[i];
    if ((p.score ?? 0) < 0.35) return;
    overlayCtx.beginPath();
    overlayCtx.arc(p.x, p.y, r, 0, Math.PI * 2);
    overlayCtx.fillStyle = color;
    overlayCtx.fill();
  };

  const colorGood = '#4ef';
  const colorWarn = '#f55';

  strokeLine(5, 7, colorGood);
  strokeLine(7, 9, colorGood);
  strokeLine(6, 8, colorGood);
  strokeLine(8, 10, colorGood);
  strokeLine(5, 6, '#8af');
  strokeLine(11, 12, '#8af');
  strokeLine(5, 11, '#8af');
  strokeLine(6, 12, '#8af');
  strokeLine(11, 13, '#8af');
  strokeLine(12, 14, '#8af');

  for (let i = 0; i < keypoints.length; i++) drawPoint(i, '#9cf', 3);
}

function evaluateForPushUp(pose) {
  const leftShoulder = getKey(pose, 5);
  const rightShoulder = getKey(pose, 6);
  const leftElbow = getKey(pose, 7);
  const rightElbow = getKey(pose, 8);
  const leftWrist = getKey(pose, 9);
  const rightWrist = getKey(pose, 10);
  if (leftShoulder.score < 0.35 || rightShoulder.score < 0.35) return;

  const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
  const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x);
  const elbowAngleL = angleDeg(leftShoulder, leftElbow, leftWrist);
  const elbowAngleR = angleDeg(rightShoulder, rightElbow, rightWrist);
  const elbowAvg = (elbowAngleL + elbowAngleR) / 2;

  const now = performance.now();
  const wristsUnder = (leftWrist.score >= 0.25 && Math.abs(leftWrist.x - leftShoulder.x) < shoulderWidth * 0.9) &&
                      (rightWrist.score >= 0.25 && Math.abs(rightWrist.x - rightShoulder.x) < shoulderWidth * 0.9);

  const shoulderDrop = baseline ? shoulderMidY - baseline.shoulderY : 0;
  const downDetected = elbowAvg < DOWN_ANGLE_THRESHOLD && shoulderDrop > baseline.shoulderDropThreshold * 0.85 && wristsUnder;
  const upDetected = elbowAvg > UP_ANGLE_THRESHOLD && shoulderDrop < baseline.shoulderDropThreshold * 0.45;

  if (downDetected && lastPosition === 'up') {
    lastPosition = 'down';
    lastDownTime = now;
  }

  if (lastPosition === 'down' && upDetected) {
    const held = now - lastDownTime;
    const sinceLastRep = now - lastRepTime;
    if (held >= MIN_DOWN_HOLD_MS && sinceLastRep >= MIN_REP_INTERVAL_MS) {
      STATE.detector.pushCount++;
      lastRepTime = now;
      lastPosition = 'up';
      feedbackText.textContent = `Flexão válida! Total: ${STATE.detector.pushCount}`;
      updateUI();
    } else {
      if (held < MIN_DOWN_HOLD_MS) feedbackText.textContent = 'Mantém a descida por mais uns ms para contar.';
      else if (sinceLastRep < MIN_REP_INTERVAL_MS) feedbackText.textContent = 'Movimento rápido detectado — desacelera.';
    }
  } else {
    if (!downDetected && !upDetected) {
      feedbackText.textContent = 'Ajusta postura: cotovelos e alinhamento necessários.';
    } else if (downDetected) {
      feedbackText.textContent = 'Descida detectada — mantém posição.';
    } else if (upDetected) {
      feedbackText.textContent = 'Volta à posição alta.';
    }
  }
}

async function runDetectionLoop() {
  if (!STATE.detector.running) return;
  try {
    const poses = await detector.estimatePoses(video, { maxPoses: 1, flipHorizontal: false });
    if (poses && poses.length > 0) {
      const pose = poses[0];
      drawPose(pose);
      if (baseline) evaluateForPushUp(pose);
    } else {
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    }
  } catch (err) {
    console.error('Erro na inferência:', err);
  }
  rafId = requestAnimationFrame(runDetectionLoop);
}

init();
