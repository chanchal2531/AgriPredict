/**
 * app.js — AgriPredict Frontend
 * Connects to Flask backend at http://localhost:5000
 */

'use strict';

// ── Config ───────────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:5000';

// ── Crop & State lists (Indian dataset: 124 crops, 33 states) ────────────────
const CROPS = [
  "Arecanut","Arhar/Tur","Bajra","Banana","Barley","Black pepper","Cardamom",
  "Cashewnut","Castor seed","Coconut","Colocasia","Cotton(lint)","Cowpea(Lobia)",
  "Dry chillies","Dry ginger","Garlic","Ginger","Gram","Grapes","Groundnut",
  "Guar seed","Horse-gram","Jowar","Jute","Khesari","Lemon","Linseed",
  "Litchi","Maize","Mango","Masoor","Mesta","Moong(Green Gram)","Moth",
  "Niger seed","Oilseeds total","Onion","Orange","Other  Rabi pulses",
  "Other Cereals & Millets","Other Kharif pulses","Other Summer Pulses",
  "Papaya","Peas & beans (Pulses)","Pepper","Pome Fruit","Pome Granet",
  "Potato","Ragi","Rapeseed &Mustard","Rice","Rubber","Safflower",
  "Sannhamp","Sesamum","Small millets","Soyabean","Sugarcane","Sunflower",
  "Sweet potato","Tapioca","Tobacco","Tomato","Turmeric","Urad","Varagu",
  "Watermelon","Wheat","Yam"
];

const STATES = [
  "Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chandigarh",
  "Chhattisgarh","Dadra and Nagar Haveli","Goa","Gujarat","Haryana",
  "Himachal Pradesh","Jammu and Kashmir","Jharkhand","Karnataka","Kerala",
  "Madhya Pradesh","Maharashtra","Manipur","Meghalaya","Mizoram",
  "Nagaland","Odisha","Puducherry","Punjab","Rajasthan","Sikkim",
  "Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand",
  "West Bengal","Andaman and Nicobar Islands"
];

// ── DOM refs ──────────────────────────────────────────────────────────────────
const cropSelect     = document.getElementById('cropSelect');
const stateSelect    = document.getElementById('stateSelect');
const areaInput      = document.getElementById('areaInput');
const yearInput      = document.getElementById('yearInput');
const predictForm    = document.getElementById('predictForm');
const submitBtn      = document.getElementById('submitBtn');
const btnSpinner     = document.getElementById('btnSpinner');
const btnLabel       = submitBtn.querySelector('.btn-label');
const apiError       = document.getElementById('apiError');
const apiErrorMsg    = document.getElementById('apiErrorMsg');
const resultsSection = document.getElementById('resultsSection');
const statusDot      = document.getElementById('statusDot');
const statusText     = document.getElementById('statusText');

// ── Populate dropdowns ────────────────────────────────────────────────────────
function populateSelect(selectEl, items) {
  items.slice().sort().forEach(item => {
    const opt = document.createElement('option');
    opt.value = item;
    opt.textContent = item;
    selectEl.appendChild(opt);
  });
}
populateSelect(cropSelect,  CROPS);
populateSelect(stateSelect, STATES);

// ── Background particles ──────────────────────────────────────────────────────
(function spawnParticles() {
  const container = document.getElementById('bgParticles');
  const colors = ['#4ade80', '#22c55e', '#86efac', '#fbbf24', '#34d399'];
  for (let i = 0; i < 25; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 6 + 2;
    p.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random() * 100}%;
      background:${colors[Math.floor(Math.random() * colors.length)]};
      animation-duration:${Math.random() * 18 + 10}s;
      animation-delay:${Math.random() * 12}s;
    `;
    container.appendChild(p);
  }
})();

// ── Health check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      statusDot.className  = 'badge-dot online';
      statusText.textContent = 'Backend Online';
    } else {
      throw new Error();
    }
  } catch {
    statusDot.className  = 'badge-dot offline';
    statusText.textContent = 'Backend Offline';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// ── Validation ────────────────────────────────────────────────────────────────
function setFieldError(id, msg) {
  const el = document.getElementById(id);
  if (el) el.textContent = msg;
}
function clearErrors() {
  ['cropError','stateError','areaError','yearError'].forEach(id => setFieldError(id, ''));
  [cropSelect, stateSelect, areaInput, yearInput].forEach(el => el.classList.remove('error'));
  apiError.hidden = true;
}

function validateForm() {
  let valid = true;
  if (!cropSelect.value) {
    setFieldError('cropError', 'Please select a crop.');
    cropSelect.classList.add('error');
    valid = false;
  }
  if (!stateSelect.value) {
    setFieldError('stateError', 'Please select a state.');
    stateSelect.classList.add('error');
    valid = false;
  }
  const area = parseFloat(areaInput.value);
  if (!areaInput.value || isNaN(area) || area <= 0) {
    setFieldError('areaError', 'Enter a valid area greater than 0.');
    areaInput.classList.add('error');
    valid = false;
  }
  const year = parseInt(yearInput.value);
  if (!yearInput.value || isNaN(year) || year < 1900 || year > 2100) {
    setFieldError('yearError', 'Enter a valid year between 1900 and 2100.');
    yearInput.classList.add('error');
    valid = false;
  }
  return valid;
}

// ── Loading state ─────────────────────────────────────────────────────────────
function setLoading(on) {
  submitBtn.disabled   = on;
  btnSpinner.className = on ? 'btn-spinner visible' : 'btn-spinner';
  btnLabel.textContent = on ? 'Predicting…' : 'Predict Now';
}

// ── Number animation ──────────────────────────────────────────────────────────
function animateNumber(el, target, decimals = 2, duration = 1000) {
  const start = Date.now();
  const step = () => {
    const p = Math.min((Date.now() - start) / duration, 1);
    const ease = 1 - Math.pow(1 - p, 3);
    el.textContent = (target * ease).toFixed(decimals);
    if (p < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ── Confidence ring ───────────────────────────────────────────────────────────
function animateRing(pct) {
  const circumference = 2 * Math.PI * 42; // 263.9
  const ringFill = document.getElementById('ringFill');
  const valEl    = document.getElementById('valConfidence');
  const offset   = circumference * (1 - pct / 100);

  // Trigger in next frame so CSS transition fires
  requestAnimationFrame(() => {
    ringFill.style.strokeDashoffset = offset.toFixed(2);
  });

  // Animate percentage text
  const start = Date.now();
  const step = () => {
    const progress = Math.min((Date.now() - start) / 1200, 1);
    const ease = 1 - Math.pow(1 - progress, 3);
    valEl.textContent = (pct * ease).toFixed(1) + '%';
    if (progress < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ── Render results ────────────────────────────────────────────────────────────
function showResults(data) {
  resultsSection.hidden = false;

  // Scroll to results
  setTimeout(() => resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);

  // Production
  animateNumber(document.getElementById('valProduction'), data.predicted_production, 2);

  // Yield
  animateNumber(document.getElementById('valYield'), data.predicted_yield, 4);

  // Risk
  const risk      = (data.risk_category || 'Unknown').toLowerCase();
  const riskEl    = document.getElementById('valRisk');
  const riskBar   = document.getElementById('riskBar');
  const riskIcon  = document.getElementById('riskIcon');
  riskEl.textContent = data.risk_category;
  riskEl.className   = `result-value risk-value risk-${risk}`;
  riskBar.className  = `risk-bar ${risk}`;
  const icons = { low: '🟢', medium: '🟡', high: '🔴' };
  riskIcon.textContent = icons[risk] || '🛡️';

  // Confidence
  animateRing(data.confidence_score);

  // Model details
  if (data.model_details) {
    document.getElementById('regModel').textContent = data.model_details.regression_model     || '—';
    document.getElementById('clfModel').textContent = data.model_details.classification_model || '—';
  }
}

// ── Form submit ───────────────────────────────────────────────────────────────
predictForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  clearErrors();

  if (!validateForm()) return;

  setLoading(true);

  const payload = {
    crop:  cropSelect.value,
    state: stateSelect.value,
    area:  parseFloat(areaInput.value),
    year:  parseInt(yearInput.value),
  };

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
      signal:  AbortSignal.timeout(90000),
    });

    const json = await res.json();

    if (!res.ok) {
      apiError.hidden   = false;
      apiErrorMsg.textContent = json.error || `Server error (${res.status})`;
    } else {
      showResults(json);
    }
  } catch (err) {
    apiError.hidden  = false;
    if (err.name === 'TimeoutError') {
      apiErrorMsg.textContent = 'Request timed out. Make sure the backend is running on port 5000.';
    } else if (err.name === 'TypeError') {
      apiErrorMsg.textContent = 'Cannot connect to backend. Start the Flask server: python app.py';
    } else {
      apiErrorMsg.textContent = err.message || 'Unexpected error occurred.';
    }
  } finally {
    setLoading(false);
  }
});

// ── Clear errors on input ─────────────────────────────────────────────────────
[cropSelect, stateSelect, areaInput, yearInput].forEach(el => {
  el.addEventListener('change', () => {
    el.classList.remove('error');
    const errId = el.id.replace('Select','Error').replace('Input','Error');
    setFieldError(errId, '');
  });
});
