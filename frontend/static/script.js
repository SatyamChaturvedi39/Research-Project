// ============================================================================
// NBA Trade Analyzer - Frontend Integration with ML Backend
// ============================================================================

// Global state
let allTeams = [];
let currentPlayers = {};

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
    // 1. Check for duplicates
    const existingCard = document.querySelector(`.player-card[data-player-name="${playerName}"]`);
    if (existingCard) {
        showError(`${playerName} is already in the trade!`);
        return;
    }

    const listId = type === 'outgoing' ? 'outgoing-players' : 'incoming-players';
    const list = document.getElementById(listId);

    // Remove empty state if it exists
    const emptyState = list.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    // Show loading indicator
    const loadingCard = document.createElement('div');
    loadingCard.className = 'player-card loading-card';
    loadingCard.innerHTML = '<div class="loading-spinner"></div><div>Loading prediction...</div>';
    list.appendChild(loadingCard);

    try {
        // Fetch player predictions
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ player_name: playerName })
        });

        // Remove loading card
        loadingCard.remove();

        if (!response.ok) {
            const error = await response.json();
            showError(error.error || 'Failed to fetch player data');
            return;
        }

        const prediction = await response.json();

        // Create player card with REAL data from API
        const playerCard = document.createElement('div');
        playerCard.className = 'player-card';
        playerCard.style.animation = 'fadeIn 0.3s ease';
        playerCard.setAttribute('data-player-name', prediction.player_name);
        playerCard.setAttribute('data-predicted-ppg', prediction.predictions.ppg);
        playerCard.setAttribute('data-current-age', prediction.current_age);
        playerCard.setAttribute('data-team', prediction.team);

        playerCard.innerHTML = `
            <div class="player-name">${prediction.player_name}</div>
            <div class="player-stat"><strong>${prediction.team}</strong> | Age: ${prediction.current_age}</div>
            <div class="player-stat">Last Season: ${prediction.current_stats.ppg} PPG</div>
            <div class="player-stat prediction-stat"><strong>Predicted:</strong> ${prediction.predictions.ppg} PPG</div>
            <div class="player-stat-small">
                <span>RPG: ${prediction.predictions.rpg}</span> | 
                <span>APG: ${prediction.predictions.apg}</span>
            </div>
            <div class="player-stat-small">
                <span>MPG: ${prediction.predictions.mpg}</span> | 
                <span>TS%: ${(prediction.predictions.ts_pct * 100).toFixed(1)}%</span>
            </div>
            <button class="remove-btn" onclick="removePlayer(this)">Remove</button>
        `;

        list.appendChild(playerCard);
        showSuccess(`Added ${prediction.player_name} to ${type} players`);

    } catch (error) {
        loadingCard.remove();
        console.error('Failed to add player:', error);
        showError('Network error. Please try again.');
    }
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

    // Get all player cards
    const outgoingPlayers = document.querySelectorAll('#outgoing-players .player-card');
    const incomingPlayers = document.querySelectorAll('#incoming-players .player-card');

    if (outgoingPlayers.length === 0 || incomingPlayers.length === 0) {
        showError('Please add players to both teams before analyzing');
        return;
    }

    // Show loading state
    dashboard.style.display = 'block';
    document.getElementById('score-value').textContent = '...';

    // Calculate trade score based on predicted performance
    let outgoingTotalPPG = 0;
    let incomingTotalPPG = 0;
    let outgoingAvgAge = 0;
    let incomingAvgAge = 0;

    // Calculate totals using data attributes
    outgoingPlayers.forEach(card => {
        outgoingTotalPPG += parseFloat(card.getAttribute('data-predicted-ppg'));
        outgoingAvgAge += parseInt(card.getAttribute('data-current-age'));
    });
    outgoingAvgAge /= outgoingPlayers.length;

    incomingPlayers.forEach(card => {
        incomingTotalPPG += parseFloat(card.getAttribute('data-predicted-ppg'));
        incomingAvgAge += parseInt(card.getAttribute('data-current-age'));
    });
    incomingAvgAge /= incomingPlayers.length;

    // Calculate trade success score
    const ppgDelta = incomingTotalPPG - outgoingTotalPPG;
    let baseScore = 50 + (ppgDelta * 2); // Each PPG = 2 points

    // Age factor (positive if getting younger)
    const ageDelta = outgoingAvgAge - incomingAvgAge;
    baseScore += (ageDelta * 1.5); // Each year younger = 1.5 points

    // Clamp score between 0-100
    const finalScore = Math.max(0, Math.min(100, baseScore));

    // Animate score
    setTimeout(() => {
        dashboard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);

    let currentScore = 0;
    const targetScore = finalScore;
    const increment = targetScore / 50;

    const countInterval = setInterval(() => {
        currentScore += increment;
        if (currentScore >= targetScore) {
            currentScore = targetScore;
            clearInterval(countInterval);
        }
        document.getElementById('score-value').textContent = currentScore.toFixed(1);
    }, 20);

    // Update analysis text based on score
    updateTradeAnalysis(finalScore, ppgDelta, ageDelta, incomingTotalPPG, outgoingTotalPPG);
}

function updateTradeAnalysis(score, ppgDelta, ageDelta, incomingPPG, outgoingPPG) {
    const reasonsList = document.querySelector('.top-reasons ul');
    reasonsList.innerHTML = '';

    // Reason 1: PPG change
    if (ppgDelta > 2) {
        reasonsList.innerHTML += `<li class="positive">✓ Offensive improvement: +${ppgDelta.toFixed(1)} PPG predicted</li>`;
    } else if (ppgDelta < -2) {
        reasonsList.innerHTML += `<li class="warning">⚠ Offensive decline: ${ppgDelta.toFixed(1)} PPG predicted</li>`;
    } else {
        reasonsList.innerHTML += `<li class="neutral">• Similar offensive output predicted</li>`;
    }

    // Reason 2: Age
    if (ageDelta > 2) {
        reasonsList.innerHTML += `<li class="positive">✓ Roster gets ${ageDelta.toFixed(1)} years younger on average</li>`;
    } else if (ageDelta < -2) {
        reasonsList.innerHTML += `<li class="warning">⚠ Roster ages by ${Math.abs(ageDelta).toFixed(1)} years</li>`;
    } else {
        reasonsList.innerHTML += `<li class="neutral">• Similar average age</li>`;
    }

    // Reason 3: Overall recommendation
    if (score >= 70) {
        reasonsList.innerHTML += `<li class="positive">✓ Strong trade recommendation based on ML predictions</li>`;
    } else if (score >= 50) {
        reasonsList.innerHTML += `<li class="neutral">• Moderate trade value - consider team needs</li>`;
    } else {
        reasonsList.innerHTML += `<li class="warning">⚠ Weak trade value - may not improve team</li>`;
    }

    // Reason 4: Total scoring output
    reasonsList.innerHTML += `<li class="neutral">• Outgoing: ${outgoingPPG.toFixed(1)} PPG → Incoming: ${incomingPPG.toFixed(1)} PPG</li>`;

    // Reason 5: ML model confidence
    reasonsList.innerHTML += `<li class="positive">✓ Predictions from 54-feature ML model (R² = 0.826)</li>`;
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
