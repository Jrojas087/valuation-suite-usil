// Course player logic
const STORAGE_KEY = 'ci_pymes_progress';

function loadProgress() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch { return {}; }
}

function saveProgress(data) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

let progress = loadProgress();
let currentModuleId = parseInt(new URLSearchParams(location.search).get('mod')) || 1;

// ── Sidebar ───────────────────────────────────────────────────────────────────
function renderSidebar() {
  const list = document.getElementById('module-nav-list');
  list.innerHTML = '';
  let completed = 0;
  COURSE.modules.forEach(m => {
    const done = progress[`module_${m.id}_quiz_passed`];
    if (done) completed++;
    const el = document.createElement('a');
    el.className = `module-nav-item${m.id === currentModuleId ? ' active' : ''}${done ? ' completed' : ''}`;
    el.href = '#';
    el.innerHTML = `
      <span class="module-nav-icon">${m.icon}</span>
      <span class="module-nav-text">${m.title}</span>
      <span class="module-nav-check">${done ? '✅' : ''}</span>
    `;
    el.addEventListener('click', e => { e.preventDefault(); navigateTo(m.id); });
    list.appendChild(el);
  });

  const pct = Math.round((completed / COURSE.modules.length) * 100);
  document.getElementById('sidebar-progress').style.width = pct + '%';
  document.getElementById('sidebar-progress-text').textContent =
    `${completed} de ${COURSE.modules.length} módulos completados`;

  const allDone = completed === COURSE.modules.length;
  const evalBtn = document.getElementById('sidebar-eval-btn');
  const navEval = document.getElementById('nav-eval');
  evalBtn.style.display = allDone ? 'flex' : 'none';
  if (navEval) navEval.style.display = allDone ? 'inline-flex' : 'none';
}

// ── Navigate ──────────────────────────────────────────────────────────────────
function navigateTo(id) {
  currentModuleId = id;
  history.replaceState(null, '', `?mod=${id}`);
  renderSidebar();
  renderModule(id);
  window.scrollTo(0, 0);
}

// ── Module content ────────────────────────────────────────────────────────────
function renderModule(id) {
  const mod = COURSE.modules.find(m => m.id === id);
  if (!mod) return;
  const quizPassed = progress[`module_${id}_quiz_passed`];
  const quizAnswers = progress[`module_${id}_quiz_answers`] || {};

  const area = document.getElementById('module-content');
  area.innerHTML = `
    <div class="module-header">
      <div class="meta">Módulo ${mod.id} de ${COURSE.modules.length} &nbsp;·&nbsp; ${mod.duration} &nbsp;·&nbsp;
        ${quizPassed ? '<span class="badge badge-success">✅ Completado</span>' : '<span class="badge badge-info">En progreso</span>'}
      </div>
      <h1 style="margin-top:0.4rem;">${mod.icon} ${mod.title}</h1>
    </div>
    <div class="module-body">${mod.content}</div>
    <div class="quiz-section" id="quiz-section">
      <h3>✏️ Knowledge Check — Módulo ${mod.id}</h3>
      <p style="color:rgba(255,255,255,0.7);font-size:0.88rem;margin-bottom:1.5rem;">
        Responde las 3 preguntas para marcar este módulo como completado.
      </p>
      <div id="quiz-questions"></div>
      <div id="quiz-submit-area" class="quiz-actions" style="display:${quizPassed ? 'none' : 'flex'}">
        <button class="btn btn-accent" id="quiz-submit-btn" onclick="submitQuiz(${id})" disabled>
          Verificar respuestas
        </button>
      </div>
      <div id="quiz-result" style="display:${quizPassed ? 'block' : 'none'}">
        <div class="quiz-result-banner">
          <span class="quiz-result-icon">🎉</span>
          <div>
            <strong style="color:#fff;">¡Módulo completado!</strong>
            <p style="color:rgba(255,255,255,0.75);font-size:0.88rem;margin:0;">
              Puntaje: ${progress[`module_${id}_score`] || '3'}/3
            </p>
          </div>
        </div>
      </div>
    </div>
    <div class="flex gap-2 mt-3" style="margin-bottom:2rem;">
      ${id > 1 ? `<button class="btn btn-outline" onclick="navigateTo(${id-1})">← Módulo anterior</button>` : ''}
      ${id < COURSE.modules.length
        ? `<button class="btn btn-primary" onclick="navigateTo(${id+1})">Siguiente módulo →</button>`
        : `<a href="evaluacion.html" class="btn btn-accent">🎯 Ir a Evaluación Final →</a>`
      }
    </div>
  `;

  renderQuizQuestions(mod, quizAnswers, quizPassed);
}

// ── Quiz questions ────────────────────────────────────────────────────────────
function renderQuizQuestions(mod, savedAnswers, locked) {
  const container = document.getElementById('quiz-questions');
  container.innerHTML = '';

  mod.quiz.forEach((q, qi) => {
    const saved = savedAnswers[qi];
    const div = document.createElement('div');
    div.className = 'quiz-question';
    div.innerHTML = `
      <p>${qi + 1}. ${q.question}</p>
      <div class="quiz-options" id="opts-${qi}"></div>
      <div class="quiz-feedback" id="fb-${qi}"></div>
    `;
    const opts = div.querySelector(`#opts-${qi}`);
    const letters = ['A', 'B', 'C', 'D'];
    q.options.forEach((opt, oi) => {
      let cls = 'quiz-option';
      if (locked) {
        if (oi === q.correct) cls += ' correct';
        else if (oi === saved && oi !== q.correct) cls += ' wrong';
      } else if (oi === saved) cls += ' selected';

      const o = document.createElement('div');
      o.className = cls;
      o.innerHTML = `<span class="quiz-option-letter">${letters[oi]}</span>${opt}`;
      if (!locked) {
        o.addEventListener('click', () => selectOption(mod.id - 1, qi, oi));
      }
      opts.appendChild(o);
    });

    if (locked) {
      const fb = div.querySelector(`#fb-${qi}`);
      fb.style.display = 'block';
      fb.textContent = '💡 ' + q.explanation;
    }
    container.appendChild(div);
  });

  if (!locked) updateSubmitBtn(mod.id - 1);
}

// ── Option selection ──────────────────────────────────────────────────────────
const quizState = {};

function selectOption(modIdx, qi, oi) {
  const modId = modIdx + 1;
  if (!quizState[modId]) quizState[modId] = {};
  quizState[modId][qi] = oi;

  const opts = document.querySelectorAll(`#opts-${qi} .quiz-option`);
  opts.forEach((o, idx) => {
    o.classList.toggle('selected', idx === oi);
  });
  updateSubmitBtn(modIdx);
}

function updateSubmitBtn(modIdx) {
  const mod = COURSE.modules[modIdx];
  const btn = document.getElementById('quiz-submit-btn');
  if (!btn) return;
  const answered = Object.keys(quizState[modIdx + 1] || {}).length;
  btn.disabled = answered < mod.quiz.length;
}

// ── Submit quiz ───────────────────────────────────────────────────────────────
function submitQuiz(modId) {
  const mod = COURSE.modules.find(m => m.id === modId);
  const answers = quizState[modId] || {};
  let score = 0;

  mod.quiz.forEach((q, qi) => {
    if (answers[qi] === q.correct) score++;
  });

  const passed = score === mod.quiz.length;

  // Save
  progress[`module_${modId}_quiz_answers`] = answers;
  progress[`module_${modId}_quiz_passed`] = passed || progress[`module_${modId}_quiz_passed`] || false;
  progress[`module_${modId}_score`] = score;
  if (passed) progress[`module_${modId}_quiz_passed`] = true;
  saveProgress(progress);

  // Show feedback on each option
  mod.quiz.forEach((q, qi) => {
    const opts = document.querySelectorAll(`#opts-${qi} .quiz-option`);
    opts.forEach((o, oi) => {
      if (oi === q.correct) o.classList.add('correct');
      else if (oi === answers[qi] && oi !== q.correct) o.classList.add('wrong');
      o.style.cursor = 'default';
      o.replaceWith(o.cloneNode(true)); // remove click listeners
    });
    const fb = document.getElementById(`fb-${qi}`);
    if (fb) { fb.style.display = 'block'; fb.textContent = '💡 ' + q.explanation; }
  });

  document.getElementById('quiz-submit-area').style.display = 'none';
  const resultDiv = document.getElementById('quiz-result');
  resultDiv.style.display = 'block';

  if (passed) {
    resultDiv.innerHTML = `
      <div class="quiz-result-banner">
        <span class="quiz-result-icon">🎉</span>
        <div>
          <strong style="color:#fff;">¡Excelente! Módulo completado (${score}/${mod.quiz.length})</strong>
          <p style="color:rgba(255,255,255,0.75);font-size:0.88rem;margin:0.3rem 0 0;">
            ${modId < COURSE.modules.length ? 'Puedes continuar con el siguiente módulo.' : 'Completaste todos los módulos. ¡Ve a la evaluación final!'}
          </p>
        </div>
      </div>
    `;
    renderSidebar();
    if (modId === COURSE.modules.length) {
      document.getElementById('sidebar-eval-btn').style.display = 'flex';
    }
  } else {
    resultDiv.innerHTML = `
      <div class="quiz-result-banner" style="background:rgba(231,76,60,0.2);">
        <span class="quiz-result-icon">📖</span>
        <div>
          <strong style="color:#fff;">Obtuviste ${score}/${mod.quiz.length}. Revisa el contenido.</strong>
          <p style="color:rgba(255,255,255,0.75);font-size:0.88rem;margin:0.3rem 0 0;">
            Puedes intentar el quiz nuevamente recargando el módulo.
          </p>
          <button class="btn btn-sm" style="margin-top:0.75rem;background:rgba(255,255,255,0.15);color:#fff;"
            onclick="resetQuiz(${modId})">🔄 Intentar de nuevo</button>
        </div>
      </div>
    `;
  }
}

function resetQuiz(modId) {
  delete progress[`module_${modId}_quiz_passed`];
  delete progress[`module_${modId}_quiz_answers`];
  delete quizState[modId];
  saveProgress(progress);
  renderModule(modId);
  renderSidebar();
}

// ── Init ──────────────────────────────────────────────────────────────────────
renderSidebar();
renderModule(currentModuleId);
