function removePlayer(button) {
    const playerCard = button.parentElement;
    playerCard.style.animation = 'fadeOut 0.3s ease';
    setTimeout(() => {
        playerCard.remove();
    }, 300);
}

function addPlayer(type) {
    const listId = type === 'outgoing' ? 'outgoing-players' : 'incoming-players';
    const list = document.getElementById(listId);

    const players = [
        { name: 'Anthony Davis', ppg: 24.7, age: 31 },
        { name: 'Jaylen Brown', ppg: 23.1, age: 27 },
        { name: 'Stephen Curry', ppg: 26.4, age: 36 },
        { name: 'Jimmy Butler', ppg: 20.8, age: 34 },
        { name: 'Damian Lillard', ppg: 25.0, age: 33 }
    ];

    const randomPlayer = players[Math.floor(Math.random() * players.length)];

    const playerCard = document.createElement('div');
    playerCard.className = 'player-card';
    playerCard.style.animation = 'fadeIn 0.3s ease';
    playerCard.innerHTML = `
        <div class="player-name">${randomPlayer.name}</div>
        <div class="player-stat">PPG: ${randomPlayer.ppg}</div>
        <div class="player-stat">Age: ${randomPlayer.age}</div>
        <button class="remove-btn" onclick="removePlayer(this)">Remove</button>
    `;

    list.appendChild(playerCard);
}

function analyzeTrade() {
    const dashboard = document.getElementById('results-dashboard');

    dashboard.style.display = 'block';

    setTimeout(() => {
        dashboard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);

    const randomScore = (Math.random() * 30 + 60).toFixed(1);
    const scoreElement = document.getElementById('score-value');

    let currentScore = 0;
    const targetScore = parseFloat(randomScore);
    const increment = targetScore / 50;

    const countInterval = setInterval(() => {
        currentScore += increment;
        if (currentScore >= targetScore) {
            currentScore = targetScore;
            clearInterval(countInterval);
        }
        scoreElement.textContent = currentScore.toFixed(1);
    }, 20);
}

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
`;
document.head.appendChild(style);
