
// API URL
const API_BASE_URL = 'http://localhost:5000';

// State
let allSymptoms = [];
let selectedSymptoms = new Set();

// DOM
const symptomSearch = document.getElementById('symptomSearch');
const symptomsList = document.getElementById('symptomsList');
const selectedSymptomsDiv = document.getElementById('selectedSymptoms');
const selectedCount = document.getElementById('selectedCount');
const predictBtn = document.getElementById('predictBtn');
const clearAllBtn = document.getElementById('clearAll');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');

// INIT
document.addEventListener("DOMContentLoaded", () => {
    loadSymptoms();

    symptomSearch.addEventListener("input", filterSymptoms);
    clearAllBtn.addEventListener("click", clearAllSymptoms);
    predictBtn.addEventListener("click", predictDisease);
});


// ================= LOAD SYMPTOMS =================
async function loadSymptoms() {
    try {
        const res = await fetch(`${API_BASE_URL}/symptoms`);
        const data = await res.json();

        console.log("API response:", data);

        // ✅ FIX: Direct access (no status check needed)
        allSymptoms = data.symptoms || [];

        renderSymptoms(allSymptoms);

    } catch (err) {
        showError("Failed to load symptoms");
        console.error(err);
    }
}


// ================= FILTER =================
function filterSymptoms() {
    const search = symptomSearch.value.toLowerCase();

    const filtered = allSymptoms.filter(sym =>
        sym.toLowerCase().includes(search)
    );

    renderSymptoms(filtered);
}


// ================= RENDER =================
function renderSymptoms(list) {
    if (list.length === 0) {
        symptomsList.innerHTML = `<p>No symptoms found</p>`;
        return;
    }

    symptomsList.innerHTML = list.map(sym => `
        <div class="symptom-item ${selectedSymptoms.has(sym) ? 'selected' : ''}"
             onclick="toggleSymptom('${sym.replace(/'/g, "\\'")}')">
            ${sym}
        </div>
    `).join('');
}


// ================= TOGGLE =================
function toggleSymptom(sym) {
    if (selectedSymptoms.has(sym)) {
        selectedSymptoms.delete(sym);
    } else {
        selectedSymptoms.add(sym);
    }

    updateSelected();
    renderSymptoms(allSymptoms);
}


// ================= UPDATE SELECTED =================
function updateSelected() {
    selectedCount.textContent = selectedSymptoms.size;

    if (selectedSymptoms.size === 0) {
        selectedSymptomsDiv.innerHTML = `<p>No symptoms selected yet</p>`;
        predictBtn.disabled = true;
        return;
    }

    predictBtn.disabled = false;

    selectedSymptomsDiv.innerHTML = [...selectedSymptoms].map(sym => `
        <div class="selected-tag">
            ${sym}
            <button onclick="removeSymptom('${sym}')">x</button>
        </div>
    `).join('');
}


// ================= REMOVE =================
function removeSymptom(sym) {
    selectedSymptoms.delete(sym);
    updateSelected();
    renderSymptoms(allSymptoms);
}


// ================= CLEAR =================
function clearAllSymptoms() {
    selectedSymptoms.clear();
    updateSelected();
    renderSymptoms(allSymptoms);
}


// ================= PREDICT =================
async function predictDisease() {
    try {
        predictBtn.disabled = true;
        predictBtn.innerHTML = "Predicting...";

        const res = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                symptoms: Array.from(selectedSymptoms)
            })
        });

        const data = await res.json();

        console.log("Prediction:", data);

        displayResults(data);

    } catch (err) {
        showError("Prediction failed");
        console.error(err);
    } finally {
        predictBtn.disabled = false;
        predictBtn.innerHTML = "Predict Disease";
    }
}


// ================= DISPLAY =================
function displayResults(data) {
    const container = document.getElementById("topPredictionsCards");

    const preds = data.top_predictions || [];

    container.innerHTML = preds.map((p, i) => `
        <div class="prediction-card">
            <h3>#${i+1} ${p.disease}</h3>
            <p>${p.probability}%</p>
        </div>
    `).join('');

    document.getElementById("suggestedSteps").innerText =
        data.suggested_steps || "Consult a doctor.";

    document.getElementById("matchedSymptoms").innerHTML =
        (data.matched_symptoms || []).map(s => `<span>${s}</span>`).join('');

    resultSection.classList.remove("hidden");
}


// ================= ERROR =================
function showError(msg) {
    document.getElementById("errorMessage").innerText = msg;
    errorSection.classList.remove("hidden");
}


// GLOBAL FUNCTIONS
window.toggleSymptom = toggleSymptom;
window.removeSymptom = removeSymptom;