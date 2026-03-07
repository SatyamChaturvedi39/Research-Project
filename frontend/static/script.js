// ============================================================================
// NBA Trade Analyzer - Frontend Integration with ML Backend
// ============================================================================

// Global state
let allTeams = [];
let currentPlayers = {};

// ============================================================================
// Theme Management
// ============================================================================

function initTheme() {
    const themeBtn = document.getElementById('theme-toggle');
    const currentTheme = localStorage.getItem('theme') || 'dark';

    if (currentTheme === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
        if (themeBtn) themeBtn.textContent = '☀️';
    }

    if (themeBtn) {
        themeBtn.addEventListener('click', () => {
            let theme = document.documentElement.getAttribute('data-theme');
            if (theme === 'light') {
                document.documentElement.removeAttribute('data-theme');
                localStorage.setItem('theme', 'dark');
                themeBtn.textContent = '🌙';
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
                localStorage.setItem('theme', 'light');
                themeBtn.textContent = '☀️';
            }
        });
    }
}

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

    // Array.from(otherSelect.options).forEach(option => {
    //     if (option.value === selectedTeam && selectedTeam !== "") {
    //         option.disabled = true;
    //     } else {
    //         option.disabled = false;
    //     }
    // });

    // 4. Fetch team health
    if (selectedTeam) {
        fetchTeamHealth(selectedTeam, type);
    } else {
        document.getElementById(`health-badge-${type}`).style.display = 'none';
    }
}

async function fetchTeamHealth(teamCode, type) {
    try {
        const res = await fetch(`/api/roster/${teamCode}`);
        const data = await res.json();

        const badge = document.getElementById(`health-badge-${type}`);
        const valEl = document.getElementById(`health-val-${type}`);
        const gradeEl = document.getElementById(`health-grade-${type}`);

        badge.style.display = 'flex';
        valEl.textContent = data.roster_health.roster_health_score;
        gradeEl.textContent = data.roster_health.health_grade;

        // Color based on grade
        const colors = {
            'EXCELLENT': '#00ff87',
            'GOOD': '#ffd700',
            'AVERAGE': '#ff9f43',
            'BELOW AVERAGE': '#ff4466',
            'POOR': '#ff0000'
        };
        gradeEl.style.color = colors[data.roster_health.health_grade] || '#fff';

    } catch (e) {
        console.error("Health fetch failed:", e);
    }
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

        // Medical color
        const medColors = {
            'EXCELLENT': '#00ff87',
            'GOOD': '#94fbab',
            'FAIR': '#ffd700',
            'POOR': '#ff9f43',
            'CRITICAL': '#ff4466'
        };
        const medColor = medColors[p.medical_grade] || '#fff';

        playerCard.innerHTML = `
            <div class="pc-header">
                <div class="pc-name-group">
                    <div class="pc-name">${p.player_name}</div>
                    <div class="pc-med-grade" style="color: ${medColor}">Medical: ${p.medical_grade}</div>
                </div>
                <div class="pc-badges">
                    <span class="${confBadgeClass}">${confLabel.toUpperCase()} CONF</span>
                    <span class="inj-badge inj-${p.injury_risk_category.toLowerCase().replace(' ', '-')}">${p.injury_risk_category} RISK</span>
                </div>
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

            <div class="pc-range">Consistency Range: <span>${confRange}</span></div>
            <div class="pc-medical-info">Injury Probability: <strong>${(p.injury_risk_prob * 100).toFixed(1)}%</strong></div>
            <div class="pc-salary-info">Salary: <strong>$${(p.salary / 1e6).toFixed(1)}M</strong> · <strong>${p.years_remaining} yrs</strong></div>

            ${shapHTML}

            <button class="remove-btn" onclick="removePlayer(this)">✕ Remove</button>
        `;

        list.appendChild(playerCard);
        showSuccess(`Added ${p.player_name} to trade`);

    } catch (error) {
        loadingCard.remove();
        console.error('Failed to add player:', error);
        showError(`Error adding player: ${error.message || 'System error'}`);
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

    const teamA = document.getElementById('team-select-outgoing').value;
    const teamB = document.getElementById('team-select-incoming').value;
    const sentA = Array.from(outgoingCards).map(c => c.getAttribute('data-player-name'));
    const sentB = Array.from(incomingCards).map(c => c.getAttribute('data-player-name'));

    dashboard.style.display = 'block';
    dashboard.scrollIntoView({ behavior: 'smooth' });

    // Reset placeholders
    document.getElementById('analytical-dashboard').style.display = 'none';
    document.getElementById('score-value').textContent = '…';
    document.getElementById('fairness-value').textContent = '…';

    try {
        const response = await fetch('/api/trade/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                team_a: teamA,
                team_b: teamB,
                sent_a: sentA,
                sent_b: sentB
            })
        });

        if (!response.ok) {
            const err = await response.json();
            showError(err.error || 'Evaluation failed');
            return;
        }

        const data = await response.json();
        renderTradeEvaluation(data);

    } catch (error) {
        console.error('Failed to analyze trade:', error);
        showError('Network error during evaluation.');
    }
}

function renderTradeEvaluation(data) {
    // 1. Team Names
    document.getElementById('oc-team-a').textContent = data.team_a.code;
    document.getElementById('oc-team-b').textContent = data.team_b.code;
    document.getElementById('tp-team-a').textContent = data.team_a.code;
    document.getElementById('tp-team-b').textContent = data.team_b.code;

    // 2. Metrics for Team A
    animateValue('oc-wins-a', 0, data.team_a.post_wins, 1);
    setDelta('oc-delta-a', data.team_a.win_change, 'wins');
    document.getElementById('oc-ci-a').textContent = `90% CI: [${data.team_a.win_ci[0] > 0 ? '+' : ''}${data.team_a.win_ci[0]}, +${data.team_a.win_ci[1]}]`;

    document.getElementById('oc-playoff-a').textContent = (data.team_a.post_wins >= 42 ? Math.min(99.9, 50 + (data.team_a.post_wins - 42) * 10) : Math.max(0.1, 40 - (42 - data.team_a.post_wins) * 10)).toFixed(1) + '%';
    setDelta('oc-p-delta-a', data.team_a.playoff_change, '%');

    setDelta('oc-inj-delta-a', -data.team_a.injury_risk_change, '%', true); // Invert for "Better/Worse" logic
    setDelta('oc-health-delta-a', data.team_a.health_change, ' pts');
    setDelta('oc-med-delta-a', data.team_a.medical_change, ' pts');

    document.getElementById('oc-grade-a').textContent = data.team_a.grade;
    setGradeColor('oc-grade-a', data.team_a.grade);

    // Financial A
    document.getElementById('oc-salary-total-a').textContent = `$${data.team_a.salary_in.toFixed(1)}M`;
    setDelta('oc-salary-delta-a', data.team_a.salary_delta, 'M', true); // More spending is "red" by default in setDelta(true)

    // 3. Metrics for Team B
    animateValue('oc-wins-b', 0, data.team_b.post_wins, 1);
    setDelta('oc-delta-b', data.team_b.win_change, 'wins');
    document.getElementById('oc-ci-b').textContent = `90% CI: [${data.team_b.win_ci[0] > 0 ? '+' : ''}${data.team_b.win_ci[0]}, +${data.team_b.win_ci[1]}]`;

    document.getElementById('oc-playoff-b').textContent = (data.team_b.post_wins >= 42 ? Math.min(99.9, 50 + (data.team_b.post_wins - 42) * 10) : Math.max(0.1, 40 - (42 - data.team_b.post_wins) * 10)).toFixed(1) + '%';
    setDelta('oc-p-delta-b', data.team_b.playoff_change, '%');

    setDelta('oc-inj-delta-b', -data.team_b.injury_risk_change, '%', true);
    setDelta('oc-health-delta-b', data.team_b.health_change, ' pts');
    setDelta('oc-med-delta-b', data.team_b.medical_change, ' pts');

    document.getElementById('oc-grade-b').textContent = data.team_b.grade;
    setGradeColor('oc-grade-b', data.team_b.grade);

    // Financial B
    document.getElementById('oc-salary-total-b').textContent = `$${data.team_b.salary_in.toFixed(1)}M`;
    setDelta('oc-salary-delta-b', data.team_b.salary_delta, 'M', true);

    // 4. Fairness & Quality
    document.getElementById('fairness-value').textContent = data.assessment.fairness + '%';
    document.getElementById('fairness-label').textContent = data.assessment.fairness > 90 ? 'VERY FAIR' : data.assessment.fairness > 75 ? 'FAIR' : 'ONE-SIDED';

    document.getElementById('oa-quality').textContent = data.assessment.combined_quality;
    document.getElementById('oa-classification').textContent = data.assessment.classification;

    // 5. Main Score (averaged for the top display)
    const meanScore = (data.team_a.score + data.team_b.score) / 2;
    animateValue('score-value', 0, meanScore, 0);
    updateScoreColor(meanScore);

    // 6. Traded Players Summary
    renderTradedPlayers(data.traded_players);

    // 7. Explanation Summary
    const summaryText = document.getElementById('trade-summary-text');
    let summary = `This trade is rated as **${data.assessment.classification}**. `;
    if (data.team_a.win_change > 0 && data.team_b.win_change > 0) {
        summary += `Both teams improve their projected ceilings. `;
    }
    summary += `Winner: **${data.assessment.winner}** by ${data.assessment.win_margin} wins.`;
    summaryText.innerHTML = summary;

    // 8. Detailed Bullets
    const detailWrap = document.getElementById('trade-explanation');
    const sentences = [];

    if (Math.abs(data.team_a.medical_change) > 2) {
        sentences.push(`The simulation identifies a shift in long-term durability profiles between the involved rosters.`);
    }
    sentences.push(`Monte Carlo analysis confirms ${data.team_a.code} has a 90% chance to finish within [${data.team_a.win_ci[0]}, ${data.team_a.win_ci[1]}] wins of their baseline.`);
    sentences.push(`Post-trade roster health for ${data.team_a.code} is graded as **${data.team_a.health_grade_post}**.`);

    detailWrap.innerHTML = sentences.map((s, i) => `
        <p class="explanation-sentence${i === 0 ? ' lead' : ''}">${s}</p>
    `).join('');

    // 9. Analytical Chart
    const dashboard = document.getElementById('analytical-dashboard');
    const chartImg = document.getElementById('trade-analysis-chart');
    if (data.chart_url) {
        chartImg.src = data.chart_url;
        dashboard.style.display = 'block';
    } else {
        dashboard.style.display = 'none';
    }
}

function renderTradedPlayers(players) {
    const listA = document.getElementById('tp-list-a');
    const listB = document.getElementById('tp-list-b');

    const buildMiniItem = (p) => `
        <div class="tp-mini-item">
            <div class="tp-mini-name">${p.player_name}</div>
            <div class="tp-mini-stats">${p.points_per_game.toFixed(1)} PPG · $${(p.salary / 1e6).toFixed(1)}M Salary · <span class="med-text">${p.medical_grade}</span></div>
        </div>
    `;

    listA.innerHTML = players.from_a.map(buildMiniItem).join('');
    listB.innerHTML = players.from_b.map(buildMiniItem).join('');
}

function setDelta(id, val, unit, invertColor = false) {
    const el = document.getElementById(id);
    if (!el) return;
    const isPos = val > 0;
    const isNeg = val < 0;

    let displayVal = `${isPos ? '+' : ''}${val.toFixed(1)}${unit}`;
    el.textContent = displayVal;

    // Logic: for injury risk, positive DELTA is BAD (red), negative is GOOD (green)
    // But we pass inverted delta to make it intuitive (+2% Risk = Red)
    // Wait, let's just use the value directly and a flag
    if (invertColor) {
        el.className = `oc-delta ${isPos ? 'neg' : isNeg ? 'pos' : ''}`;
    } else {
        el.className = `oc-delta ${isPos ? 'pos' : isNeg ? 'neg' : ''}`;
    }
}

function setGradeColor(id, grade) {
    const el = document.getElementById(id);
    const colors = {
        'BENEFICIAL': '#00ff87',
        'SLIGHTLY POSITIVE': '#94fbab',
        'NEUTRAL': '#ffd700',
        'SLIGHTLY NEGATIVE': '#ff9f43',
        'HARMFUL': '#ff4466'
    };
    el.style.color = colors[grade] || '#fff';
}

function updateScoreColor(score) {
    const el = document.getElementById('score-value');
    if (score >= 70) el.style.color = '#00ff87';
    else if (score >= 50) el.style.color = '#ffd700';
    else if (score >= 30) el.style.color = '#ff9f43';
    else el.style.color = '#ff4466';
}

function animateValue(id, start, end, decimals, suffix = '') {
    const el = document.getElementById(id);
    if (!el) return;
    let current = start;
    const range = end - start;
    const step = range / 30;
    const timer = setInterval(() => {
        current += step;
        el.textContent = current.toFixed(decimals) + suffix;
        if ((range > 0 && current >= end) || (range < 0 && current <= end)) {
            el.textContent = end.toFixed(decimals) + suffix;
            clearInterval(timer);
        }
    }, 16);
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
    initTheme();
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
