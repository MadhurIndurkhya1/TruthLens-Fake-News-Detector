const form = document.getElementById('detectorForm');
const btn = document.getElementById('predictBtn');
const statusEl = document.getElementById('status');
const result = document.getElementById('result');
const labelEl = document.getElementById('label');
const scoreEl = document.getElementById('score');
const messageEl = document.getElementById('message');
const resetBtn = document.getElementById('resetBtn');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const title = document.getElementById('title').value.trim();
  const text = document.getElementById('text').value.trim();
  if (text.length < 10) return;

  btn.disabled = true;
  btn.textContent = 'Analyzing…';
  statusEl.textContent = '';
  result.classList.add('hidden');

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({title, text})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Prediction failed');

    labelEl.textContent = data.label;
    labelEl.className = data.label === 'FAKE' ? 'fake' : 'real';
    scoreEl.textContent = `Score: ${data.decision_score.toFixed(4)}`;
    messageEl.textContent = data.message;
    result.classList.remove('hidden');
  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Analyze article';
  }
});

resetBtn.addEventListener('click', () => {
  result.classList.add('hidden');
  document.getElementById('text').focus();
});
