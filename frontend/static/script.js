// ============================================================================
// NBA Trade Analyzer - Frontend Integration with ML Backend
// ============================================================================

// Global state
let allTeams = [];
let currentPlayers = {};

// ============================================================================
// Mobile Navigation
// ============================================================================

function toggleMobileNav() {
    const overlay = document.getElementById('mobile-nav-overlay');
    if (overlay) overlay.classList.toggle('active');
}

// Close mobile nav on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const overlay = document.getElementById('mobile-nav-overlay');
        if (overlay) overlay.classList.remove('active');
    }
});

// ============================================================================
// Error Notification System
// ============================================================================

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.innerHTML = `
        <div class="error-content">
            <span class="error-icon">⚠️</span>
            <span class="error-message">${message}</span>
            <button class="error-close" onclick="this.parentElement.parentElement.remove()">✕</button>
        </div>
    `;
    document.body.appendChild(errorDiv);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        errorDiv.classList.add('fade-out');
        setTimeout(() => errorDiv.remove(), 300);
    }, 5000);
}

function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success-notification';
    successDiv.innerHTML = `
        <div class="success-content">
            <span class="success-icon">✓</span>
            <span class="success-message">${message}</span>
        </div>
    `;
    document.body.appendChild(successDiv);

    setTimeout(() => {
        successDiv.classList.add('fade-out');
        setTimeout(() => successDiv.remove(), 300);
    }, 3000);
}

// ============================================================================
// Player Selection Modal
// ============================================================================

// ============================================================================
// Team & Player Selection Logic (Two-Team Constraint)
// ============================================================================

function onMainTeamChange(type) {
    const isOutgoing = type === 'outgoing';
    const selectId = isOutgoing ? 'team-select-outgoing' : 'team-select-incoming';
    const otherSelectId = isOutgoing ? 'team-select-incoming' : 'team-select-outgoing';
    const listId = isOutgoing ? 'outgoing-players' : 'incoming-players';
    const btnId = isOutgoing ? 'btn-add-outgoing' : 'btn-add-incoming';

    const selectedTeam = document.getElementById(selectId).value;
    const otherSelect = document.getElementById(otherSelectId);

    // 1. Clear existing players for this side
    const list = document.getElementById(listId);
    list.innerHTML = '';

    // 2. Manage "Add Player" button state
    const addBtn = document.getElementById(btnId);
    if (selectedTeam) {
        addBtn.disabled = false;
        // Remove empty state message if strictly needed, or let addPlayer handle it
    } else {
        addBtn.disabled = true;
        list.innerHTML = '<div class="empty-state">Select a team above to start</div>';
    }

    // 3. Prevent selecting same team on both sides
    // value constraint: Disable this team in the OTHER dropdown
    Array.from(otherSelect.options).forEach(option => {
        if (option.value === selectedTeam && selectedTeam !== "") {
            option.disabled = true;
        } else {
            option.disabled = false;
        }
    });
}

function openPlayerModal(type) {
    const modal = document.getElementById('player-selector-modal');
    modal.setAttribute('data-selection-type', type);

    // 1. Identify which team is selected for this side
    const selectId = type === 'outgoing' ? 'team-select-outgoing' : 'team-select-incoming';
    const teamId = document.getElementById(selectId).value;

    if (!teamId) {
        showError("Please select a team for this side first.");
        return;
    }

    // 2. Show Modal
    modal.style.display = 'flex';

    // 3. Load Players for this team immediately
    const playerSelect = document.getElementById('player-selector');
    playerSelect.innerHTML = '<option value="">Loading players...</option>';
    document.getElementById('confirm-player-btn').disabled = true;

    loadPlayersForModal(teamId);
}

async function loadPlayersForModal(teamId) {
    const playerSelect = document.getElementById('player-selector');

    try {
        const response = await fetch(`/api/players?team=${encodeURIComponent(teamId)}`);
        const data = await response.json();

        if (data.players.length === 0) {
            playerSelect.innerHTML = '<option value="">No players found</option>';
            return;
        }

        playerSelect.innerHTML = '<option value="">Select a player...</option>';
        data.players.forEach(player => {
            const option = document.createElement('option');
            option.value = player.player_name;
            // Show PPG in dropdown for better UX
            option.textContent = `${player.player_name} (${player.points_per_game.toFixed(1)} PPG)`;
            playerSelect.appendChild(option);
        });

    } catch (error) {
        console.error('Failed to load players:', error);
        playerSelect.innerHTML = '<option value="">Error loading players</option>';
    }
}

function hidePlayerSelector() {
    document.getElementById('player-selector-modal').style.display = 'none';
}

function onPlayerSelected() {
    const playerSelect = document.getElementById('player-selector');
    const confirmBtn = document.getElementById('confirm-player-btn');
    confirmBtn.disabled = !playerSelect.value;
}

async function confirmPlayerSelection() {
    const playerName = document.getElementById('player-selector').value;
    const type = document.getElementById('player-selector-modal').getAttribute('data-selection-type');

    if (!playerName) {
        showError('Please select a player');
        return;
    }

    hidePlayerSelector();
    await addPlayerToTrade(playerName, type);
}

// ============================================================================
// Add Player to Trade
// ============================================================================

async function addPlayerToTrade(playerName, type) {
    // 1. Duplicate guard
    const existingCard = document.querySelector(`.player-card[data-player-name="${playerName}"]`);
    if (existingCard) {
        showError(`${playerName} is already in the trade!`);
        return;
    }

    const listId = type === 'outgoing' ? 'outgoing-players' : 'incoming-players';
    const list = document.getElementById(listId);

    const emptyState = list.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    // Loading skeleton
    const loadingCard = document.createElement('div');
    loadingCard.className = 'player-card loading-card';
    loadingCard.innerHTML = '<div class="loading-spinner"></div><div>Fetching ML prediction…</div>';
    list.appendChild(loadingCard);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ player_name: playerName })
        });

        loadingCard.remove();

        if (!response.ok) {
            const err = await response.json();
            showError(err.error || 'Failed to fetch player data');
            return;
        }

        const p = await response.json();

        // ── Build SHAP reasoning HTML for PPG (primary target) ──────────────
        const ppgShap = p.shap_explanation && p.shap_explanation.ppg;
        const shapHTML = ppgShap ? buildShapSection(ppgShap, p.shap_explanation) : '';

        // ── Confidence badge ──────────────────────────────────────────────
        const ppgConf = p.confidence_ranges && p.confidence_ranges.ppg;
        const confLabel = ppgConf ? ppgConf.confidence_label : 'medium';
        const confBadgeClass = `conf-badge conf-${confLabel}`;
        const confRange = ppgConf
            ? `${ppgConf.lower} – ${ppgConf.upper} PPG`
            : '';

        // ── Player card ───────────────────────────────────────────────────
        const playerCard = document.createElement('div');
        playerCard.className = 'player-card';
        playerCard.style.animation = 'fadeIn 0.35s ease';
        playerCard.setAttribute('data-player-name', p.player_name);
        playerCard.setAttribute('data-predicted-ppg', p.predictions.ppg);
        playerCard.setAttribute('data-predicted-rpg', p.predictions.rpg);
        playerCard.setAttribute('data-predicted-apg', p.predictions.apg);
        playerCard.setAttribute('data-current-age', p.current_age);
        playerCard.setAttribute('data-team', p.team);


        playerCard.innerHTML = `
            <div class="pc-header">
                <div class="pc-name">${p.player_name}</div>
                <span class="${confBadgeClass}">${confLabel.toUpperCase()} CONF</span>
            </div>
            <div class="pc-meta">${p.team} · Age ${p.current_age} · ${p.last_season_year}</div>

            <div class="pc-stats-row">
                <div class="pc-stat-block">
                    <div class="pc-stat-value">${p.current_stats.ppg}</div>
                    <div class="pc-stat-label">Last PPG</div>
                </div>
                <div class="pc-stat-block highlight">
                    <div class="pc-stat-value">${p.predictions.ppg}</div>
                    <div class="pc-stat-label">Pred PPG</div>
                </div>
                <div class="pc-stat-block">
                    <div class="pc-stat-value">${p.predictions.rpg}</div>
                    <div class="pc-stat-label">Pred RPG</div>
                </div>
                <div class="pc-stat-block">
                    <div class="pc-stat-value">${p.predictions.apg}</div>
                    <div class="pc-stat-label">Pred APG</div>
                </div>
            </div>

            ${ppgConf ? `<div class="pc-range">Confidence range: <strong>${confRange}</strong> (MAE ±${ppgConf.mae})</div>` : ''}

            ${shapHTML}

            <button class="remove-btn" onclick="removePlayer(this)">✕ Remove</button>
        `;

        list.appendChild(playerCard);
        showSuccess(`Added ${p.player_name} to trade`);

    } catch (error) {
        loadingCard.remove();
        console.error('Failed to add player:', error);
        showError('Network error. Please try again.');
    }
}

// ── SHAP Section Builder ──────────────────────────────────────────────────────
function buildShapSection(ppgShap, allTargets) {
    const factors = ppgShap.top_factors || [];
    if (factors.length === 0) return '';

    // Max absolute SHAP for normalising bar widths
    const maxAbs = Math.max(...factors.map(f => Math.abs(f.shap)), 0.001);

    const rows = factors.map(f => {
        const barPct = Math.round((Math.abs(f.shap) / maxAbs) * 100);
        const isPos = f.direction === 'positive';
        const sign = isPos ? '+' : '−';
        const barClass = isPos ? 'shap-bar-pos' : 'shap-bar-neg';
        const icon = isPos ? '▲' : '▼';
        return `
            <div class="shap-row">
                <div class="shap-label" title="${f.feature}">${f.label}</div>
                <div class="shap-bar-track">
                    <div class="shap-bar ${barClass}" style="width:${barPct}%"></div>
                </div>
                <div class="shap-value ${isPos ? 'pos' : 'neg'}">${icon} ${sign}${Math.abs(f.shap).toFixed(2)}</div>
            </div>
            <div class="shap-reason">${f.reason}</div>
        `;
    }).join('');

    // Secondary targets summary line
    const secondaryParts = [];
    if (allTargets.rpg) secondaryParts.push(`RPG base: ${allTargets.rpg.base_value}`);
    if (allTargets.apg) secondaryParts.push(`APG base: ${allTargets.apg.base_value}`);
    const secondaryLine = secondaryParts.length
        ? `<div class="shap-secondary">Model baseline — ${secondaryParts.join(' · ')}</div>`
        : '';

    return `
        <div class="shap-section">
            <button class="shap-toggle" onclick="toggleShap(this)">
                <span class="shap-toggle-icon">▼</span> Why this prediction?
            </button>
            <div class="shap-body" style="display:none;">
                <div class="shap-header-row">
                    <span class="shap-header-text">PPG Factors <em>(base: ${ppgShap.base_value} pts avg)</em></span>
                </div>
                ${rows}
                ${secondaryLine}
            </div>
        </div>
    `;
}

function toggleShap(btn) {
    const body = btn.nextElementSibling;
    const icon = btn.querySelector('.shap-toggle-icon');
    const isOpen = body.style.display !== 'none';
    body.style.display = isOpen ? 'none' : 'block';
    icon.textContent = isOpen ? '▼' : '▲';
}



// ============================================================================
// Player Card Management
// ============================================================================

function removePlayer(button) {
    const playerCard = button.parentElement;
    const playerName = playerCard.getAttribute('data-player-name');
    playerCard.style.animation = 'fadeOut 0.3s ease';
    setTimeout(() => {
        playerCard.remove();
        showSuccess(`Removed ${playerName}`);
    }, 300);
}

// ============================================================================
// Trade Analysis (Using Real ML Predictions)
// ============================================================================

async function analyzeTrade() {
    const dashboard = document.getElementById('results-dashboard');

    const outgoingCards = document.querySelectorAll('#outgoing-players .player-card');
    const incomingCards = document.querySelectorAll('#incoming-players .player-card');

    if (outgoingCards.length === 0 || incomingCards.length === 0) {
        showError('Please add players to both teams before analyzing.');
        return;
    }

    dashboard.style.display = 'block';
    document.getElementById('score-value').textContent = '…';

    // ── Gather predicted stats from player cards ──────────────────────────
    const sumCards = (cards, attr) =>
        Array.from(cards).reduce((s, c) => s + parseFloat(c.getAttribute(attr) || 0), 0);

    const outPPG = sumCards(outgoingCards, 'data-predicted-ppg');
    const inPPG = sumCards(incomingCards, 'data-predicted-ppg');
    const outRPG = sumCards(outgoingCards, 'data-predicted-rpg') || 0;
    const inRPG = sumCards(incomingCards, 'data-predicted-rpg') || 0;
    const outAPG = sumCards(outgoingCards, 'data-predicted-apg') || 0;
    const inAPG = sumCards(incomingCards, 'data-predicted-apg') || 0;

    const outAvgAge = sumCards(outgoingCards, 'data-current-age') / outgoingCards.length;
    const inAvgAge = sumCards(incomingCards, 'data-current-age') / incomingCards.length;

    // Average ML confidence: high=1.0, medium=0.6, low=0.3
    const confWeight = (cards) => {
        const confMap = { high: 1.0, medium: 0.6, low: 0.3 };
        let total = 0;
        Array.from(cards).forEach(c => {
            const badge = c.querySelector('.conf-badge');
            if (badge) {
                const cls = Array.from(badge.classList).find(cl => cl.startsWith('conf-') && cl !== 'conf-badge');
                total += confMap[cls?.replace('conf-', '') || 'medium'] || 0.6;
            } else {
                total += 0.6;
            }
        });
        return total / cards.length;
    };
    const outConf = confWeight(outgoingCards);
    const inConf = confWeight(incomingCards);

    // ── Multi-factor scoring (0–100) ───────────────────────────────────────
    // Each component scored 0–25, then multiplied by 4 to give 0–100

    // 1. PPG balance (40 pts): how much better is incoming vs outgoing?
    const ppgDelta = inPPG - outPPG;
    const ppgScore = Math.min(25, Math.max(0, 12.5 + (ppgDelta / Math.max(outPPG, 1)) * 25));

    // 2. Rebounding balance (20 pts)
    const rpgDelta = inRPG - outRPG;
    const rpgScore = Math.min(12.5, Math.max(0, 6.25 + (rpgDelta / Math.max(outRPG, 1)) * 12.5));

    // 3. Playmaking balance (20 pts)
    const apgDelta = inAPG - outAPG;
    const apgScore = Math.min(12.5, Math.max(0, 6.25 + (apgDelta / Math.max(outAPG, 1)) * 12.5));

    // 4. Age / youth factor (20 pts): getting younger is good
    const ageDelta = outAvgAge - inAvgAge;
    const ageScore = Math.min(12.5, Math.max(0, 6.25 + (ageDelta / 5) * 12.5));

    // 5. Confidence bonus (up to 12.5 pts): higher model confidence = more reliable
    const confScore = ((outConf + inConf) / 2) * 12.5;

    // Weighted total out of 100
    const rawScore = (ppgScore + rpgScore + apgScore + ageScore + confScore) * (100 / 75);
    const finalScore = Math.round(Math.max(0, Math.min(100, rawScore)));

    // ── Animate score counter ─────────────────────────────────────────────
    setTimeout(() => dashboard.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);

    let current = 0;
    const step = finalScore / 60;
    const counter = setInterval(() => {
        current = Math.min(current + step, finalScore);
        document.getElementById('score-value').textContent = Math.round(current);
        if (current >= finalScore) clearInterval(counter);
    }, 16);

    // ── Update score meter ────────────────────────────────────────────────
    updateScoreMeter(finalScore);

    // ── Generate sentence-based explanation ───────────────────────────────
    updateTradeAnalysis(finalScore, {
        ppgDelta, outPPG, inPPG,
        rpgDelta, outRPG, inRPG,
        apgDelta, outAPG, inAPG,
        ageDelta, outAvgAge, inAvgAge,
        outConf, inConf,
        outCount: outgoingCards.length,
        inCount: incomingCards.length,
    });
}

function updateScoreMeter(score) {
    const zones = document.querySelectorAll('.guide-zone');
    if (!zones.length) return;

    // Reset categories
    zones.forEach(z => z.classList.remove('active'));

    // Determine active zone index and color — must match HTML zone ranges: 0-30/31-50/51-70/71-100
    let activeIdx = 0;
    let color = '';
    if (score >= 71) { activeIdx = 3; color = '#00ff87'; }
    else if (score >= 51) { activeIdx = 2; color = '#ffd700'; }
    else if (score >= 31) { activeIdx = 1; color = '#ff9f43'; }
    else { activeIdx = 0; color = '#ff4466'; }

    // Highlight active zone
    zones[activeIdx].classList.add('active');

    // Update main score color
    const scoreVal = document.getElementById('score-value');
    if (scoreVal) scoreVal.style.color = color;
}

function updateTradeAnalysis(score, d) {
    const wrap = document.getElementById('trade-explanation');
    if (!wrap) return;

    const sentences = [];

    // ── Sentence 1: Overall verdict — thresholds match HTML zones (71/51/31) ────
    if (score >= 71) {
        sentences.push(`The model rates this as a great trade. The incoming players project solidly across scoring, rebounding, and playmaking — a clear overall improvement.`);
    } else if (score >= 51) {
        sentences.push(`The model rates this as a good trade with some caveats. There are real upsides here, but the move isn't a slam dunk across every category.`);
    } else if (score >= 31) {
        sentences.push(`This trade carries notable risk. The model's projections suggest the incoming package doesn't clearly improve the team's overall output.`);
    } else {
        sentences.push(`This trade looks highly unfavorable. Based on historical data patterns, the outgoing talent significantly outweighs the projected return.`);
    }

    // ── Sentence 2: PPG (scoring) ─────────────────────────────────────────
    if (d.ppgDelta > 3) {
        sentences.push(`On the offensive end, you are acquiring a major scoring upgrade. The incoming players are projected to easily surpass the offensive production you're giving up.`);
    } else if (d.ppgDelta > 0) {
        sentences.push(`Scoring-wise, you gain a slight edge. The model anticipates a marginal but positive increase in overall points per game.`);
    } else if (d.ppgDelta > -3) {
        sentences.push(`The scoring impact is nearly neutral. Essentially, the team is swapping like for like offensively without losing much ground.`);
    } else {
        sentences.push(`Scoring takes a noticeably hard hit in this scenario. You're trading away primary offensive contributors without getting enough firepower in return.`);
    }

    // ── Sentence 3: Rebounding ────────────────────────────────────────────
    if (d.outRPG > 0 || d.inRPG > 0) {
        if (d.rpgDelta > 1.5) {
            sentences.push(`This move also bolsters the team's rebounding depth, translating to an expected improvement in second-chance opportunities and defensive glass coverage.`);
        } else if (d.rpgDelta < -1.5) {
            sentences.push(`Rebounding presents a real concern here, as losing size and hustle on the boards could negatively swing possession metrics.`);
        } else {
            sentences.push(`Rebounding volume stays relatively stable, meaning the team's rebounding identity won't be drastically altered.`);
        }
    }

    // ── Sentence 4: Playmaking — always outputs a sentence ─────────────────
    if (d.apgDelta > 1.5) {
        sentences.push(`Playmaking sees a distinct improvement, signaling better projected ball distribution and team offensive flow.`);
    } else if (d.apgDelta < -1.5) {
        sentences.push(`Playmaking takes a step back. Losing this volume of assists could slow ball movement and stagnate half-court sets.`);
    } else {
        sentences.push(`Assist production holds relatively steady in this deal, so ball movement and half-court offense shouldn't look very different.`);
    }

    // ── Sentence 5: Age/longevity ─────────────────────────────────────────
    if (d.ageDelta > 2) {
        sentences.push(`There is a real youth dividend accompanying this deal. Acquiring younger talent provides a longer contention window and higher development upside.`);
    } else if (d.ageDelta < -2) {
        sentences.push(`Age is a significant risk factor here. Taking on an older average roster increases the likelihood of performance decline and injury issues down the stretch.`);
    } else {
        sentences.push(`The age balance remains competitive, so the team isn't taking on extra longevity or developmental risk.`);
    }

    // ── Sentence 6: Model confidence ──────────────────────────────────────
    const avgConf = (d.outConf + d.inConf) / 2;
    if (avgConf >= 0.8) {
        sentences.push(`Finally, the XGBoost algorithms exhibit high confidence in these projections, as the involved players have robust, predictable multi-year data profiles.`);
    } else if (avgConf >= 0.5) {
        sentences.push(`The model expresses moderate confidence in this analysis. Some players have shorter track records, introducing a wider margin of uncertainty into the forecast.`);
    } else {
        sentences.push(`Please note the model's confidence is relatively low here. Limited historical data for key players means you should treat these projections with extra caution.`);
    }

    // ── Render ─────────────────────────────────────────────────────────────
    wrap.innerHTML = sentences.map((s, i) => `
        <p class="explanation-sentence${i === 0 ? ' lead' : ''}">${s}</p>
    `).join('');
}




// ============================================================================
// Initialize on Page Load
// ============================================================================

window.addEventListener('DOMContentLoaded', async () => {
    try {
        // Check health
        const healthRes = await fetch('/api/health');
        const health = await healthRes.json();

        console.log('Backend health:', health);

        if (!health.model_loaded) {
            showError('ML model not loaded. Using demo mode.');
        }

        // Load teams for dropdown
        const teamsRes = await fetch('/api/teams');
        const teamsData = await teamsRes.json();
        allTeams = teamsData.teams;

        // Populate BOTH team selectors
        const outgoingSelect = document.getElementById('team-select-outgoing');
        const incomingSelect = document.getElementById('team-select-incoming');

        // Helper to populate a select element
        const populateSelect = (selectElement) => {
            selectElement.innerHTML = '<option value="">Select Team...</option>';
            allTeams.forEach(team => {
                const option = document.createElement('option');
                option.value = team.code;
                option.textContent = team.name;
                selectElement.appendChild(option);
            });
        };

        populateSelect(outgoingSelect);
        populateSelect(incomingSelect);

        console.log(`Loaded ${allTeams.length} teams`);
        showSuccess('System ready! Select Team A and Team B to start.');

    } catch (error) {
        console.error('Failed to load backend data:', error);
        showError('Could not connect to backend. Please refresh the page.');
    }
});

// ============================================================================
// Animations & Styles
// ============================================================================

const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        to {
            opacity: 0;
            transform: translateX(-20px);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Error Notifications */
    .error-notification, .success-notification {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10000;
        animation: slideIn 0.3s ease;
    }
    
    .error-content, .success-content {
        background: #1f2937;
        color: white;
        padding: 16px 24px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        display: flex;
        align-items: center;
        gap: 12px;
        min-width: 300px;
    }
    
    .error-notification .error-content {
        border-left: 4px solid #ef4444;
    }
    
    .success-notification .success-content {
        border-left: 4px solid #10b981;
    }
    
    .error-icon, .success-icon {
        font-size: 24px;
    }
    
    .error-close {
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        margin-left: auto;
        padding: 0 4px;
    }
    
    .fade-out {
        animation: fadeOut 0.3s ease;
        opacity: 0;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Loading Card */
    .loading-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 10px;
        opacity: 0.7;
    }
    
    .loading-spinner {
        width: 30px;
        height: 30px;
        border: 3px solid #e5e7eb;
        border-top-color: #3b82f6;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Player stat highlighting */
    .prediction-stat {
        color: #4ade80;
        font-weight: bold;
    }
    
    .player-stat-small {
        font-size: 0.85em;
        color: #94a3b8;
        margin-top: 4px;
    }
`;
document.head.appendChild(style);
