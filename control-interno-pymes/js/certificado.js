// Certificate generation using Canvas API
const STORAGE_KEY = 'ci_pymes_progress';

function loadProgress() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch { return {}; }
}

function generateCertId(name, date) {
  let hash = 0;
  const str = name + date + 'CIPYMES2025';
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0;
  }
  return 'CP-' + Math.abs(hash).toString(36).toUpperCase().padStart(8, '0');
}

function generateCertificate() {
  const name = document.getElementById('cert-name').value.trim();
  if (!name) {
    alert('Por favor ingresa tu nombre completo.');
    return;
  }
  const org = document.getElementById('cert-org').value.trim();
  const p = loadProgress();
  const score = p.exam_score || 0;
  const total = p.exam_total || 20;
  const date = p.exam_date || new Date().toLocaleDateString('es-PY', { year: 'numeric', month: 'long', day: 'numeric' });
  const pct = Math.round((score / total) * 100);
  const certId = generateCertId(name, date);

  drawCertificate(name, org, date, certId);
  document.getElementById('cert-area').style.display = 'block';
  document.getElementById('cert-area').scrollIntoView({ behavior: 'smooth' });
}

function drawCertificate(name, org, date, certId) {
  const canvas = document.getElementById('cert-canvas');
  const W = 1100, H = 780;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  // ── Background ──────────────────────────────────────────────────────────────
  const bgGrad = ctx.createLinearGradient(0, 0, W, H);
  bgGrad.addColorStop(0, '#FAFCFF');
  bgGrad.addColorStop(1, '#EAF2FF');
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, W, H);

  // ── Decorative border ───────────────────────────────────────────────────────
  // Outer border
  ctx.strokeStyle = '#1B4F72';
  ctx.lineWidth = 8;
  ctx.strokeRect(20, 20, W - 40, H - 40);
  // Inner border
  ctx.strokeStyle = '#C0392B';
  ctx.lineWidth = 2;
  ctx.strokeRect(32, 32, W - 64, H - 64);

  // Corner ornaments
  drawCorner(ctx, 45, 45);
  drawCorner(ctx, W - 45, 45, true);
  drawCorner(ctx, 45, H - 45, false, true);
  drawCorner(ctx, W - 45, H - 45, true, true);

  // ── Top stripe ─────────────────────────────────────────────────────────────
  const stripe = ctx.createLinearGradient(0, 0, W, 0);
  stripe.addColorStop(0, '#1B4F72');
  stripe.addColorStop(0.5, '#2E86C1');
  stripe.addColorStop(1, '#C0392B');
  ctx.fillStyle = stripe;
  ctx.fillRect(32, 32, W - 64, 8);

  // ── Flag colors bar ─────────────────────────────────────────────────────────
  ctx.fillStyle = '#D52B1E'; ctx.fillRect(32, H - 40, (W - 64) / 3, 8);
  ctx.fillStyle = '#FFFFFF'; ctx.fillRect(32 + (W - 64) / 3, H - 40, (W - 64) / 3, 8);
  ctx.fillStyle = '#002D6E'; ctx.fillRect(32 + 2 * (W - 64) / 3, H - 40, (W - 64) / 3, 8);

  // ── Shield emoji ────────────────────────────────────────────────────────────
  ctx.font = '52px serif';
  ctx.textAlign = 'center';
  ctx.fillText('🛡️', W / 2, 115);

  // ── Institution ─────────────────────────────────────────────────────────────
  ctx.fillStyle = '#7F8C8D';
  ctx.font = '500 13px "Segoe UI", sans-serif';
  ctx.letterSpacing = '2px';
  ctx.fillText('PROGRAMA DE CAPACITACIÓN EMPRESARIAL — PARAGUAY', W / 2, 145);

  // ── Certificate label ───────────────────────────────────────────────────────
  ctx.fillStyle = '#1B4F72';
  ctx.font = 'bold 15px "Segoe UI", sans-serif';
  ctx.fillText('CERTIFICADO DE APROBACIÓN', W / 2, 175);

  // ── Divider ──────────────────────────────────────────────────────────────────
  ctx.strokeStyle = '#D5D8DC';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(180, 190); ctx.lineTo(W - 180, 190);
  ctx.stroke();

  // ── "Otorgado a" ────────────────────────────────────────────────────────────
  ctx.fillStyle = '#5D6D7E';
  ctx.font = '15px "Segoe UI", sans-serif';
  ctx.fillText('Se certifica que', W / 2, 225);

  // ── Name ────────────────────────────────────────────────────────────────────
  ctx.fillStyle = '#1C2833';
  const nameSize = name.length > 30 ? 34 : 42;
  ctx.font = `bold italic ${nameSize}px Georgia, serif`;
  ctx.fillText(name, W / 2, 285);

  // Name underline
  const nameWidth = ctx.measureText(name).width;
  ctx.strokeStyle = '#C0392B';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(W / 2 - nameWidth / 2, 295);
  ctx.lineTo(W / 2 + nameWidth / 2, 295);
  ctx.stroke();

  // ── Organization ────────────────────────────────────────────────────────────
  if (org) {
    ctx.fillStyle = '#5D6D7E';
    ctx.font = '14px "Segoe UI", sans-serif';
    ctx.fillText(org, W / 2, 325);
  }

  // ── Completion text ──────────────────────────────────────────────────────────
  ctx.fillStyle = '#2C3E50';
  ctx.font = '16px "Segoe UI", sans-serif';
  ctx.fillText('aprobó exitosamente el curso de auto-instrucción', W / 2, org ? 365 : 345);

  // ── Course title ─────────────────────────────────────────────────────────────
  ctx.fillStyle = '#1B4F72';
  ctx.font = 'bold 26px "Segoe UI", sans-serif';
  ctx.fillText('Control Interno para PYMES', W / 2, org ? 405 : 385);

  // ── Subtitle ────────────────────────────────────────────────────────────────
  ctx.fillStyle = '#5D6D7E';
  ctx.font = '14px "Segoe UI", sans-serif';
  ctx.fillText('Basado en el Marco Integrado de Control Interno COSO 2013', W / 2, org ? 435 : 415);

  // ── Date ────────────────────────────────────────────────────────────────────
  const infoY = org ? 490 : 465;
  ctx.fillStyle = '#7F8C8D';
  ctx.font = '13px "Segoe UI", sans-serif';
  ctx.fillText(`Fecha de aprobación: ${date}`, W / 2, infoY);

  // ── Certificate ID ───────────────────────────────────────────────────────────
  ctx.fillStyle = '#AAB7B8';
  ctx.font = '11px "Segoe UI", monospace';
  ctx.fillText(`Código de verificación: ${certId}`, W / 2, infoY + 22);

  // ── Signature line ───────────────────────────────────────────────────────────
  const sigY = H - 90;
  ctx.strokeStyle = '#1B4F72';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(W / 2 - 180, sigY); ctx.lineTo(W / 2 + 180, sigY); ctx.stroke();
  ctx.fillStyle = '#1B4F72'; ctx.font = 'bold 13px "Segoe UI", sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Jorge Rojas', W / 2, sigY + 18);
  ctx.fillStyle = '#5D6D7E'; ctx.font = '12px "Segoe UI", sans-serif';
  ctx.fillText('Auditor Interno Certificado COSO · Instructor', W / 2, sigY + 34);
}

// ── Helper: corner ornament ──────────────────────────────────────────────────
function drawCorner(ctx, x, y, flipX = false, flipY = false) {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(flipX ? -1 : 1, flipY ? -1 : 1);
  ctx.strokeStyle = '#1B4F72';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, 20); ctx.lineTo(0, 0); ctx.lineTo(20, 0);
  ctx.stroke();
  ctx.fillStyle = '#C0392B';
  ctx.beginPath();
  ctx.arc(0, 0, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// ── Helper: rounded rectangle ───────────────────────────────────────────────
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
  ctx.fill();
}

// ── Download ─────────────────────────────────────────────────────────────────
function downloadCert() {
  const canvas = document.getElementById('cert-canvas');
  const name = document.getElementById('cert-name').value.trim().replace(/\s+/g, '_') || 'certificado';
  const a = document.createElement('a');
  a.download = `Certificado_Control_Interno_${name}.png`;
  a.href = canvas.toDataURL('image/png');
  a.click();
}

function printCert() {
  const canvas = document.getElementById('cert-canvas');
  const img = canvas.toDataURL('image/png');
  const win = window.open('', '_blank');
  win.document.write(`
    <html><head><title>Certificado — Control Interno PYMES</title>
    <style>body{margin:0;display:flex;justify-content:center;align-items:center;min-height:100vh;background:#fff;}
    img{max-width:100%;max-height:100vh;}</style></head>
    <body><img src="${img}" onload="window.print()"></body></html>
  `);
  win.document.close();
}

function shareCert() {
  const name = document.getElementById('cert-name').value.trim();
  const text = `¡Aprobé el curso de Control Interno para PYMES (Paraguay)! 🏆 #ControlInterno #PYMES #Paraguay`;
  if (navigator.share) {
    navigator.share({ title: 'Mi certificado - Control Interno PYMES', text });
  } else {
    navigator.clipboard.writeText(text).then(() => alert('Texto copiado al portapapeles. ¡Compártelo donde quieras!'));
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  const p = loadProgress();
  if (p.exam_passed) {
    document.getElementById('screen-cert').style.display = 'block';
    // Pre-fill score info
    if (p.exam_score !== undefined) {
      const pct = Math.round((p.exam_score / p.exam_total) * 100);
      document.querySelector('.exam-header p').textContent =
        `Aprobaste con ${pct}% (${p.exam_score}/${p.exam_total}) el ${p.exam_date}. ¡Genera tu certificado a continuación!`;
    }
  } else {
    document.getElementById('screen-locked').style.display = 'block';
  }
});
