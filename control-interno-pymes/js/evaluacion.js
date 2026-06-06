// Final evaluation logic
const STORAGE_KEY = 'ci_pymes_progress';
const letters = ['A', 'B', 'C', 'D'];
let examAnswers = {};
let examSubmitted = false;

function loadProgress() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch { return {}; }
}
function saveExamResult(score, total, passed) {
  const p = loadProgress();
  p.exam_score = score;
  p.exam_total = total;
  p.exam_passed = passed;
  p.exam_date = new Date().toLocaleDateString('es-PY', { year: 'numeric', month: 'long', day: 'numeric' });
  p.exam_answers = examAnswers;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
}

function startExam() {
  document.getElementById('screen-intro').style.display = 'none';
  document.getElementById('screen-exam').style.display = 'block';
  renderQuestions();
  window.scrollTo(0, 0);
}

function renderQuestions() {
  const container = document.getElementById('exam-questions-list');
  container.innerHTML = '';

  COURSE.finalExam.forEach((q, qi) => {
    const card = document.createElement('div');
    card.className = 'exam-question-card';
    card.id = `eq-${qi}`;
    card.innerHTML = `
      <div class="exam-q-number">Pregunta ${qi + 1} de ${COURSE.finalExam.length}</div>
      <p>${q.question}</p>
      <div class="exam-options" id="eopts-${qi}">
        ${q.options.map((opt, oi) => `
          <div class="exam-option" id="eopt-${qi}-${oi}" onclick="selectExamOption(${qi},${oi})">
            <span class="exam-option-letter">${letters[oi]}</span>
            <span>${opt}</span>
          </div>
        `).join('')}
      </div>
    `;
    container.appendChild(card);
  });
}

function selectExamOption(qi, oi) {
  if (examSubmitted) return;
  const prev = examAnswers[qi];
  examAnswers[qi] = oi;

  // Update UI for this question
  for (let i = 0; i < COURSE.finalExam[qi].options.length; i++) {
    const el = document.getElementById(`eopt-${qi}-${i}`);
    el.classList.toggle('selected', i === oi);
  }
  document.getElementById(`eq-${qi}`).classList.add('answered');

  updateProgress();
}

function updateProgress() {
  const answered = Object.keys(examAnswers).length;
  const total = COURSE.finalExam.length;
  const pct = Math.round((answered / total) * 100);
  document.getElementById('exam-progress-fill').style.width = pct + '%';
  document.getElementById('answers-count').textContent = `${answered} / ${total} respondidas`;
}

function submitExam() {
  if (examSubmitted) return;
  const answered = Object.keys(examAnswers).length;
  const total = COURSE.finalExam.length;

  if (answered < total) {
    document.getElementById('incomplete-warning').style.display = 'flex';
    if (!submitExam._confirmed) {
      submitExam._confirmed = true;
      return;
    }
  }

  examSubmitted = true;
  let score = 0;
  COURSE.finalExam.forEach((q, qi) => {
    if (examAnswers[qi] === q.correct) score++;
  });

  const pct = Math.round((score / total) * 100);
  const passed = pct >= COURSE.passingScore;

  saveExamResult(score, total, passed);
  showResults(score, total, pct, passed);
}
submitExam._confirmed = false;

function showResults(score, total, pct, passed) {
  document.getElementById('screen-exam').style.display = 'none';
  document.getElementById('screen-results').style.display = 'block';

  const wrongItems = COURSE.finalExam
    .map((q, qi) => ({ q, qi, correct: examAnswers[qi] === q.correct }))
    .filter(i => !i.correct);

  const resultArea = document.getElementById('result-content');
  resultArea.innerHTML = `
    <div class="result-score-circle ${passed ? 'passed' : 'failed'}" style="color:${passed ? 'var(--success)' : 'var(--accent)'};">
      <div class="score-number">${pct}%</div>
      <div class="score-label">${score}/${total} correctas</div>
    </div>

    <h2>${passed ? '🎉 ¡Felicitaciones, aprobaste!' : '📖 No alcanzaste el puntaje mínimo'}</h2>
    <p class="text-muted" style="margin-bottom:1.5rem;">
      ${passed
        ? `Obtuviste <strong>${score} de ${total}</strong> respuestas correctas (${pct}%). El mínimo requerido es 70 % (14/20).`
        : `Obtuviste <strong>${score} de ${total}</strong> respuestas correctas (${pct}%). Necesitas al menos 14/20 para aprobar.`
      }
    </p>

    ${passed ? `
      <a href="certificado.html" class="btn btn-success btn-lg" style="margin-bottom:1.5rem;">
        🏆 Obtener mi certificado →
      </a>
    ` : `
      <div class="flex gap-2" style="justify-content:center;flex-wrap:wrap;margin-bottom:1.5rem;">
        <a href="curso.html" class="btn btn-primary">📚 Repasar módulos</a>
        <button class="btn btn-accent" onclick="retryExam()">🔄 Intentar nuevamente</button>
      </div>
    `}

    ${wrongItems.length > 0 ? `
      <div style="text-align:left;margin-top:2rem;">
        <h3 style="margin-bottom:1rem;font-size:1rem;">Preguntas para repasar (${wrongItems.length}):</h3>
        ${wrongItems.map(item => `
          <div style="background:var(--card);border-radius:8px;padding:1rem;margin-bottom:0.75rem;border-left:3px solid var(--accent);">
            <p style="font-weight:600;font-size:0.9rem;margin-bottom:0.5rem;">${item.qi + 1}. ${item.q.question}</p>
            ${item.q.options.map((opt, oi) => `
              <div style="font-size:0.85rem;padding:0.3rem 0.5rem;border-radius:4px;margin-bottom:0.2rem;
                background:${oi === item.q.correct ? '#D5F5E3' : (oi === examAnswers[item.qi] ? '#FDEDEC' : 'transparent')};
                color:${oi === item.q.correct ? 'var(--success)' : (oi === examAnswers[item.qi] ? 'var(--accent)' : 'var(--text-muted)')};
                font-weight:${oi === item.q.correct ? '600' : '400'};">
                ${letters[oi]}. ${opt}
                ${oi === item.q.correct ? ' ✓' : (oi === examAnswers[item.qi] ? ' ✗' : '')}
              </div>
            `).join('')}
          </div>
        `).join('')}
      </div>
    ` : ''}
  `;

  window.scrollTo(0, 0);
}

function retryExam() {
  examAnswers = {};
  examSubmitted = false;
  submitExam._confirmed = false;
  document.getElementById('screen-results').style.display = 'none';
  document.getElementById('screen-exam').style.display = 'block';
  renderQuestions();
  updateProgress();
  document.getElementById('incomplete-warning').style.display = 'none';
  window.scrollTo(0, 0);
}

// Check if already passed on load
window.addEventListener('DOMContentLoaded', () => {
  const p = loadProgress();
  if (p.exam_passed) {
    const pct = Math.round((p.exam_score / p.exam_total) * 100);
    document.getElementById('screen-intro').innerHTML += `
      <div class="alert alert-success" style="max-width:500px;margin:1rem auto;">
        <span>🏆</span>
        <span>Ya aprobaste esta evaluación con <strong>${pct}%</strong>.
        <a href="certificado.html">Ver mi certificado →</a></span>
      </div>
    `;
  }
});
