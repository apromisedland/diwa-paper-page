/* All performance values come from the manuscript ledger. Query ranks are schematic. */
(() => {
  'use strict';
  document.body.classList.add('js-ready');
  const $ = (selector) => document.querySelector(selector);
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const data = window.DIWA_DATA;
  const icons = () => window.lucide?.createIcons();
  icons();

  const menu = $('.menu-toggle');
  const nav = $('#navigation');
  function closeMenu() {
    nav.classList.remove('is-open');
    menu.setAttribute('aria-expanded', 'false');
    menu.setAttribute('aria-label', 'Open navigation');
    menu.title = 'Open navigation';
  }
  menu.addEventListener('click', () => {
    const open = nav.classList.toggle('is-open');
    menu.setAttribute('aria-expanded', String(open));
    menu.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    menu.title = open ? 'Close navigation' : 'Open navigation';
  });
  nav.addEventListener('click', (event) => { if (event.target.closest('a')) closeMenu(); });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && nav.classList.contains('is-open')) { closeMenu(); menu.focus(); }
  });
  document.addEventListener('click', (event) => { if (!event.target.closest('.site-header')) closeMenu(); });
  window.matchMedia('(min-width: 521px)').addEventListener('change', closeMenu);

  const dialog = $('#figure-dialog');
  $('#architecture-open').addEventListener('click', (event) => {
    if (typeof dialog.showModal !== 'function') return;
    event.preventDefault();
    dialog.showModal();
  });
  dialog.addEventListener('click', (event) => {
    const box = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) dialog.close();
  });

  let scrollPending = false;
  function updateProgress() {
    const range = document.documentElement.scrollHeight - window.innerHeight;
    $('.scroll-progress').style.transform = `scaleX(${range > 0 ? window.scrollY / range : 0})`;
    scrollPending = false;
  }
  window.addEventListener('scroll', () => {
    if (!scrollPending) { scrollPending = true; requestAnimationFrame(updateProgress); }
  }, { passive: true });
  window.addEventListener('resize', updateProgress);
  updateProgress();

  if ('IntersectionObserver' in window) {
    const reveals = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) { entry.target.classList.add('is-visible'); reveals.unobserve(entry.target); }
      }
    }, { threshold: 0.08 });
    document.querySelectorAll('.reveal').forEach((element) => {
      if (!reduced.matches) element.classList.add('reveal-pending');
      reveals.observe(element);
    });
    const sections = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        nav.querySelectorAll('[aria-current]').forEach((link) => link.removeAttribute('aria-current'));
        nav.querySelector(`a[href="#${entry.target.id}"]`)?.setAttribute('aria-current', 'location');
      }
    }, { rootMargin: '-15% 0px -65% 0px', threshold: 0 });
    document.querySelectorAll('main section[id]').forEach((section) => sections.observe(section));
  }

  if (data) {
    const grid = $('#query-grid');
    // A stable permutation keeps selected sets nested as the budget increases.
    const ranked = Array.from({ length: 48 }, (_, index) => ({ index, score: Math.sin(index * 12.9898 + 4) * 43758.5453 % 1 }))
      .sort((a, b) => b.score - a.score).map((entry) => entry.index);
    const ranks = Array.from({ length: 48 }, (_, index) => ranked.indexOf(index));
    grid.replaceChildren(...ranks.map((rank, index) => {
      const cell = document.createElement('span');
      cell.className = 'query-cell';
      cell.dataset.rank = rank;
      cell.title = `Slot ${index % 16 + 1}, offset ${Math.floor(index / 16) + 1}; illustrative rank ${rank + 1}`;
      cell.setAttribute('aria-hidden', 'true');
      return cell;
    }));
    grid.setAttribute('role', 'img');
    const slider = $('#budget-slider');
    const adaptive = $('#adaptive-toggle');
    const chart = $('#budget-chart');
    const x = (latency) => 40 + (latency - 55) / 170 * 364;
    const y = (success) => 155 - (success - 60) / 16 * 123;
    chart.innerHTML = `<svg viewBox="0 0 440 220" aria-hidden="true">
      ${[60, 65, 70, 75].map((value) => `<path d="M40 ${y(value)}H416" stroke="#dfe8e3"/><text x="28" y="${y(value) + 3}" text-anchor="end">${value}</text>`).join('')}
      ${[60, 100, 150, 200].map((value) => `<text x="${x(value)}" y="179" text-anchor="middle">${value}</text>`).join('')}
      <text x="40" y="15">Success (%)</text><text x="416" y="207" text-anchor="end">Mean latency (ms)</text>
      <polyline fill="none" stroke="#7daba0" stroke-width="2" points="${data.budgets.map((point) => `${x(point.latency_ms)},${y(point.standard_success_pct)}`).join(' ')}"/>
      ${data.budgets.map((point) => `<circle cx="${x(point.latency_ms)}" cy="${y(point.standard_success_pct)}" r="4" fill="#087f72"><title>${point.selected_queries} queries: ${point.standard_success_pct}% success, ${point.latency_ms} ms</title></circle>`).join('')}
      <circle id="budget-halo" class="chart-halo" r="12"/><circle id="budget-point" class="active-point" r="6"/>
    </svg>`;
    function updateBudget() {
      const isAdaptive = adaptive.checked;
      const point = isAdaptive ? data.adaptive : data.budgets[Number(slider.value)];
      const count = isAdaptive ? 12 : point.selected_queries;
      slider.disabled = isAdaptive;
      const fixed = data.budgets[Number(slider.value)];
      slider.setAttribute('aria-valuetext', `${fixed.budget_pct} percent, ${fixed.selected_queries} queries`);
      $('#budget-ratio').textContent = isAdaptive ? '25% ceiling' : `${point.budget_pct}%`;
      $('#setting-label').textContent = isAdaptive ? 'Adaptive / 25% ceiling' : `Fixed ${point.budget_pct}% budget`;
      $('#budget-success').innerHTML = `${point.standard_success_pct.toFixed(1)}<span>%</span>`;
      $('#budget-latency').innerHTML = `${point.latency_ms}<span> ms</span>`;
      $('#query-count').textContent = isAdaptive ? 'Up to 12 / 48' : `${count} / 48`;
      grid.setAttribute('aria-label', isAdaptive ? 'Illustrative selection at the 12-query ceiling. Actual adaptive count varies by context; mean retention is 23.7 percent.' : `${count} of 48 queries selected by illustrative influence rank.`);
      grid.querySelectorAll('.query-cell').forEach((cell) => cell.classList.toggle('selected', Number(cell.dataset.rank) < count));
      $('#budget-note').textContent = isAdaptive
        ? 'Measured adaptive setting: 23.7% mean retention. Grid illustrates the 12-query ceiling; actual counts vary with context.'
        : 'Measured fixed-count settings. Lines connect observations; they do not predict intermediate performance.';
      for (const id of ['#budget-point', '#budget-halo']) {
        $(id).setAttribute('cx', x(point.latency_ms));
        $(id).setAttribute('cy', y(point.standard_success_pct));
      }
      chart.setAttribute('aria-label', `Measured budget frontier. Selected setting: ${point.standard_success_pct}% success at ${point.latency_ms} milliseconds.`);
    }
    slider.addEventListener('input', updateBudget);
    adaptive.addEventListener('change', updateBudget);
    $('.budget-controls').hidden = false;
    updateBudget();

    const resultChart = $('#result-chart');
    const diwa = data.platforms.find((row) => row.method === 'DIWA');
    const baseline = data.platforms.find((row) => row.method === 'DreamVLA');
    function updateResults(category) {
      const entries = category === 'ood' ? data.ood.map((row) => ({ label: row.condition, diwa: row.DIWA, baseline: row.DreamVLA }))
        : [['LIBERO', 'LIBERO'], ['RoboTwin', 'RoboTwin'], ['RoboCasa', 'RoboCasa'], ['Real robot', 'real_robot_pct']]
          .map(([label, key]) => ({ label, diwa: diwa[key], baseline: baseline[key] }));
      resultChart.innerHTML = entries.map((row) => `<div class="result-group"><div class="bar-pair">
        <div class="result-bar" style="--height:${row.diwa}%"><span class="bar-value">${row.diwa.toFixed(1)}%</span></div>
        <div class="result-bar baseline" style="--height:${row.baseline}%"><span class="bar-value">${row.baseline.toFixed(1)}%</span></div>
        </div><h3 class="bar-label">${row.label}</h3><p class="bar-gain">+${(row.diwa - row.baseline).toFixed(1)} points</p></div>`).join('');
      resultChart.setAttribute('aria-label', `Success, DIWA versus DreamVLA. ${entries.map((row) => `${row.label}: ${row.diwa.toFixed(1)} versus ${row.baseline.toFixed(1)} percent`).join('. ')}`);
      $('#results-note').textContent = category === 'ood'
        ? 'Three reported shift conditions and their equal-weight mean. These are task-success percentages; contact-counterfactual success differs from CF decision-label accuracy.'
        : 'Simulation means use training seeds 42, 43, and 44. Physical results use the seed-42 checkpoint. The four-platform macro-average weights platforms equally.';
      document.querySelectorAll('[data-result]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.result === category)));
      if (!reduced.matches && window.Motion?.animate) window.Motion.animate(resultChart.querySelectorAll('.result-bar'), { opacity: [0.35, 1] }, { duration: 0.5 });
    }
    document.querySelectorAll('[data-result]').forEach((button) => button.addEventListener('click', () => updateResults(button.dataset.result)));
    $('.results-controls').hidden = false;
    updateResults('platforms');
  }

  const canvas = $('#flow-canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  let width = 0, height = 0, frame = 0, phase = 0, previousTime = 0;
  let userPaused = false, inView = true;
  const motionButton = $('#motion-toggle');
  motionButton.hidden = false;
  function draw(time = 0) {
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const unit = width / 1100;
    const cy = height * 0.55;
    const left = width * 0.15, gate = width * 0.46, right = width * 0.68, end = width * 0.88;
    const gap = Math.min(21, width * 0.019), tile = gap * 0.52;
    const selected = new Set([1, 4, 7, 10, 15, 18, 21, 25, 30, 34, 39, 44].map((n) => (n + Math.floor(time / 7000) * 5) % 48));
    const paths = [];
    let selectedIndex = 0;
    for (let index = 0; index < 48; index++) {
      const col = index % 8, row = Math.floor(index / 8);
      const sx = left + (col - 3.5) * gap + row * gap * 0.15;
      const sy = cy + (row - 2.5) * gap;
      const active = selected.has(index);
      ctx.fillStyle = active ? '#087f72' : '#dbe6df';
      ctx.beginPath();ctx.roundRect(sx - tile / 2, sy - tile / 2, tile, tile, 2);ctx.fill();
      if (!active) continue;
      const target = selectedIndex++;
      const ex = right + (target % 4 - 1.5) * gap * 1.4;
      const ey = cy + (Math.floor(target / 4) - 1) * gap * 1.4;
      ctx.strokeStyle = '#087f7224';ctx.lineWidth = Math.max(0.5, unit);
      ctx.beginPath();ctx.moveTo(sx + tile, sy);ctx.bezierCurveTo(gate - width * .06, sy, gate + width * .035, ey, ex - tile, ey);ctx.stroke();
      paths.push({ sx: sx + tile, sy, ex: ex - tile, ey });
      ctx.fillStyle = target % 5 === 0 ? '#c96c53' : '#087f72';
      ctx.beginPath();ctx.roundRect(ex - tile * .65, ey - tile * .65, tile * 1.3, tile * 1.3, 2);ctx.fill();
      ctx.strokeStyle = '#087f7224';
      ctx.beginPath();ctx.moveTo(ex + tile, ey);ctx.bezierCurveTo(end - width * .08, ey, end - width * .07, cy, end - gap, cy);ctx.stroke();
    }
    ctx.setLineDash([3, 5]);ctx.strokeStyle = '#8ca69b';ctx.lineWidth = 1;
    ctx.beginPath();ctx.moveTo(gate, cy - gap * 3.1);ctx.lineTo(gate, cy + gap * 3.1);ctx.stroke();ctx.setLineDash([]);
    // Moving packets follow the same Bezier paths used for selected queries.
    paths.forEach((p, index) => {
      const t = ((time / 3600 + index / paths.length) % 1), inv = 1 - t;
      const px = inv ** 3 * p.sx + 3 * inv ** 2 * t * (gate - width * .06) + 3 * inv * t ** 2 * (gate + width * .035) + t ** 3 * p.ex;
      const py = inv ** 3 * p.sy + 3 * inv ** 2 * t * p.sy + 3 * inv * t ** 2 * p.ey + t ** 3 * p.ey;
      ctx.fillStyle = '#087f7288';ctx.fillRect(px - 1.5, py - 1.5, Math.max(2, 3 * unit), Math.max(2, 3 * unit));
    });
    const size = gap * 1.25;
    ctx.strokeStyle = '#087f72';ctx.lineWidth = Math.max(1.5, unit * 2);ctx.fillStyle = '#e3f0e9';
    ctx.beginPath();ctx.roundRect(end - size, cy - size, size * 2, size * 2, 5);ctx.fill();ctx.stroke();
    ctx.beginPath();ctx.moveTo(end - size * .5, cy);ctx.lineTo(end + size * .45, cy);ctx.moveTo(end, cy - size * .45);ctx.lineTo(end + size * .45, cy);ctx.lineTo(end, cy + size * .45);ctx.stroke();
  }
  function running() { return !reduced.matches && !userPaused && inView && !document.hidden; }
  function tick(time) {
    if (!running()) { frame = 0; return; }
    if (previousTime) phase += Math.min(time - previousTime, 50);
    previousTime = time;
    draw(phase);
    frame = requestAnimationFrame(tick);
  }
  function syncMotion() {
    const paused = reduced.matches || userPaused;
    motionButton.innerHTML = `<i data-lucide="${paused ? 'play' : 'pause'}" aria-hidden="true"></i>`;
    motionButton.setAttribute('aria-label', reduced.matches ? 'Animation disabled by reduced-motion preference' : paused ? 'Play animation' : 'Pause animation');
    motionButton.title = motionButton.getAttribute('aria-label');
    motionButton.disabled = reduced.matches;
    icons();
    if (running() && !frame) { previousTime = 0; frame = requestAnimationFrame(tick); }
    if (!running() && frame) { cancelAnimationFrame(frame); frame = 0; }
    draw(phase);
  }
  function resize() {
    const rect = canvas.getBoundingClientRect();
    width = rect.width;height = rect.height;
    const scale = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * scale);canvas.height = Math.round(height * scale);
    draw(phase);
  }
  new ResizeObserver(resize).observe(canvas);
  motionButton.addEventListener('click', () => { userPaused = !userPaused;syncMotion(); });
  reduced.addEventListener('change', syncMotion);
  document.addEventListener('visibilitychange', syncMotion);
  if ('IntersectionObserver' in window) new IntersectionObserver(([entry]) => { inView = entry.isIntersecting;syncMotion(); }).observe(canvas);
  resize();syncMotion();
})();
