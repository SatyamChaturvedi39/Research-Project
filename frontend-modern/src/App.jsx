import React, { useState } from 'react';
import Header from './components/Header';
import TeamPanel from './components/TeamPanel';
import TradeBlock from './components/TradeBlock';
import PredictionDashboard from './components/PredictionDashboard';
import TradeScoreMeter from './components/TradeScoreMeter';
import { fetchPrediction } from './api';
import { Loader2 } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

export default function App() {
  const [teamAPlayers, setTeamAPlayers] = useState([]);
  const [teamBPlayers, setTeamBPlayers] = useState([]);
  const [teamASection, setTeamASection] = useState('');
  const [teamBSection, setTeamBSection] = useState('');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const addPlayer = (player, side) => {
    const isDuplicate = teamAPlayers.some(p => p.player_name === player.player_name) ||
      teamBPlayers.some(p => p.player_name === player.player_name);

    if (isDuplicate) {
      setError(`${player.player_name} is already in the trade!`);
      setTimeout(() => setError(null), 3000);
      return;
    }

    if (side === 'A') setTeamAPlayers([...teamAPlayers, player]);
    else setTeamBPlayers([...teamBPlayers, player]);
  };

  const removePlayer = (name, side) => {
    if (side === 'A') setTeamAPlayers(p => p.filter(x => x.player_name !== name));
    else setTeamBPlayers(p => p.filter(x => x.player_name !== name));
  };

  const handleAnalyze = async () => {
    const allPlayers = [...teamAPlayers, ...teamBPlayers];
    if (allPlayers.length === 0) {
      setError('Add at least one player to analyze a trade.');
      return;
    }
    setError(null);
    setLoading(true);
    setResults(null);
    try {
      const data = await fetchPrediction(allPlayers.map(p => p.player_name));
      setResults(data);
    } catch (e) {
      setError(e.message || 'Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  const canAnalyze = teamAPlayers.length > 0 && teamBPlayers.length > 0;

  return (
    <div className="min-h-screen font-sans">
      <Header />

      <main className="max-w-7xl mx-auto px-4 py-8">
        {error && (
          <div className="mb-6 px-4 py-3 bg-red-950/50 border border-red-700/50 rounded-lg text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Three-column workspace */}
        <div className="flex flex-col lg:flex-row gap-5 items-start">
          <TeamPanel
            side="A"
            onAddPlayer={p => addPlayer(p, 'A')}
            otherSelectedTeam={teamBSection}
            onTeamChange={setTeamASection}
          />

          {/* Middle column */}
          <div className="flex-1 flex flex-col items-center w-full min-w-0">
            <TradeBlock
              teamAPlayers={teamAPlayers}
              teamBPlayers={teamBPlayers}
              onRemove={removePlayer}
            />

            <button
              onClick={handleAnalyze}
              disabled={loading || !canAnalyze}
              className="mt-5 px-8 py-2.5 rounded-lg font-semibold text-sm text-white bg-gradient-to-r from-cyan-600 to-blue-700 hover:from-cyan-500 hover:to-blue-600 disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-cyan-900/30 transition-all transform hover:scale-[1.02] active:scale-[0.98] flex items-center gap-2"
            >
              {loading && <Loader2 className="w-4 h-4 animate-spin" />}
              {loading ? 'Running Analysis…' : 'Analyze Trade'}
            </button>

            <AnimatePresence>
              {results && (
                <motion.div
                  className="w-full space-y-4"
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <TradeScoreMeter
                    targetTeam="Team A"
                    incomingResults={results.filter(r => teamBPlayers.some(p => p.player_name === r.player_name))}
                    outgoingResults={results.filter(r => teamAPlayers.some(p => p.player_name === r.player_name))}
                  />
                  <TradeScoreMeter
                    targetTeam="Team B"
                    incomingResults={results.filter(r => teamAPlayers.some(p => p.player_name === r.player_name))}
                    outgoingResults={results.filter(r => teamBPlayers.some(p => p.player_name === r.player_name))}
                  />
                  <PredictionDashboard results={results} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <TeamPanel
            side="B"
            onAddPlayer={p => addPlayer(p, 'B')}
            otherSelectedTeam={teamASection}
            onTeamChange={setTeamBSection}
          />
        </div>
      </main>
    </div>
  );
}
