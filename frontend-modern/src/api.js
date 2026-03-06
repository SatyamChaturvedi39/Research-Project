import axios from 'axios';

const API_BASE = 'http://127.0.0.1:5000/api';

export const fetchTeams = async () => {
    const response = await axios.get(`${API_BASE}/teams`);
    return response.data; // { teams: [{code, name}], count, season }
};

export const fetchPlayers = async (teamCode) => {
    const response = await axios.get(`${API_BASE}/players`, { params: { team: teamCode } });
    return response.data; // { players: [{player_name, team, age, points_per_game}] }
};

// The Flask /api/predict takes ONE player_name at a time.
// Call it per player and collect results.
export const fetchPrediction = async (playerNames) => {
    const results = await Promise.all(
        playerNames.map(name =>
            axios.post(`${API_BASE}/predict`, { player_name: name })
                .then(r => r.data)
                .catch(e => ({ error: e.response?.data?.error || e.message, player_name: name }))
        )
    );
    return results;
};
