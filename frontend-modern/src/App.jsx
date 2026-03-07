import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { analyzeTrade, predictPlayer, getTeams } from './api';
import Header from './components/Header';
import TeamPanel from './components/TeamPanel';
import TradeBlock from './components/TradeBlock';
import ResultsDashboard from './components/ResultsDashboard';
import { Loader2, ArrowRightLeft, Activity } from 'lucide-react';

function App() {
  const [teamA, setTeamA] = useState('');
  const [teamB, setTeamB] = useState('');
  const [sentA, setSentA] = useState([]);
  const [sentB, setSentB] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [tradeAnalysis, setTradeAnalysis] = useState(null);
  const [error, setError] = useState('');
  const [allTeams, setAllTeams] = useState([]);

  useEffect(() => {
    const fetchTeams = async () => {
      try {
        const data = await getTeams();
        setAllTeams(data.teams || []);
      } catch (err) {
        console.error("Failed to fetch teams:", err);
      }
    };
    fetchTeams();
  }, []);

  const handleAnalyze = async () => {
    if (!teamA || !teamB || sentA.length === 0 || sentB.length === 0) {
      setError('Please select teams and add players to both sides.');
      return;
    }

    setIsAnalyzing(true);
    setError('');
    setTradeAnalysis(null);

    const namesA = sentA.map(p => p.normalized_name || p.player_name);
    const namesB = sentB.map(p => p.normalized_name || p.player_name);

    try {
      const data = await analyzeTrade(teamA, teamB, namesA, namesB);
      setTradeAnalysis(data);
    } catch (err) {
      setError(err.response?.data?.error || 'Analysis failed due to server error.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleTeamChangeA = (newTeam) => {
    setTeamA(newTeam);
    setSentA([]);
    setTradeAnalysis(null);
  };

  const handleTeamChangeB = (newTeam) => {
    setTeamB(newTeam);
    setSentB([]);
    setTradeAnalysis(null);
  };

  const removePlayerA = (player) => {
    setSentA(sentA.filter(p => p.player_name !== player.player_name));
  };

  const removePlayerB = (player) => {
    setSentB(sentB.filter(p => p.player_name !== player.player_name));
  };

  const handleAddPlayer = async (teamSide, player) => {
    const isTeamA = teamSide === 'A';
    const currentSent = isTeamA ? sentA : sentB;
    const setSent = isTeamA ? setSentA : setSentB;

    if (currentSent.find(p => p.player_name === player.player_name)) return;

    // Add a temporary loading state for this player
    const tempPlayer = { ...player, isLoadingPrediction: true };
    setSent([...currentSent, tempPlayer]);

    try {
      const mlData = await predictPlayer(player.player_name);
      setSent(prev => prev.map(p => p.player_name === player.player_name ? mlData : p));
    } catch (err) {
      setError(`Failed to fetch ML prediction for ${player.player_name}`);
      setSent(prev => prev.filter(p => p.player_name !== player.player_name));
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 p-4 md:p-8">
      <div className="max-w-7xl mx-auto">
        <Header />

        {/* Trade Construction Area */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 relative">

          {/* Team A Panel */}
          <div className="flex flex-col gap-4">
            <TeamPanel
              title="Team A"
              team={teamA}
              setTeam={handleTeamChangeA}
              otherTeam={teamB}
              allTeams={allTeams}
              activePlayers={[...sentA, ...sentB]}
              onAddPlayer={(p) => handleAddPlayer('A', p)}
            />
            <TradeBlock
              title={`Outgoing from ${teamA || 'Team A'}`}
              players={sentA}
              onRemove={removePlayerA}
              color="orange"
            />
          </div>

          {/* Trade Direction Icon */}
          <div className="hidden lg:flex absolute left-1/2 top-32 -translate-x-1/2 z-10 w-16 h-16 bg-slate-900 border border-slate-700 shadow-2xl rounded-full items-center justify-center animate-float">
            <ArrowRightLeft className="w-8 h-8 text-slate-400" />
          </div>

          {/* Team B Panel */}
          <div className="flex flex-col gap-4">
            <TeamPanel
              title="Team B"
              team={teamB}
              setTeam={handleTeamChangeB}
              otherTeam={teamA}
              allTeams={allTeams}
              activePlayers={[...sentA, ...sentB]}
              onAddPlayer={(p) => handleAddPlayer('B', p)}
            />
            <TradeBlock
              title={`Outgoing from ${teamB || 'Team B'}`}
              players={sentB}
              onRemove={removePlayerB}
              color="amber"
            />
          </div>
        </div>

        {/* Action Button & Error */}
        <div className="flex flex-col items-center justify-center mt-12 mb-16">
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="mb-4 text-red-400 bg-red-950/30 px-6 py-3 rounded-xl border border-red-900/50"
              >
                {error}
              </motion.div>
            )}
          </AnimatePresence>
          <button
            onClick={handleAnalyze}
            disabled={isAnalyzing}
            className="group relative px-12 py-4 bg-gradient-to-r from-orange-500 to-rose-500 text-white font-bold text-lg rounded-2xl overflow-hidden hover:scale-[1.02] transition-transform shadow-[0_0_40px_rgba(249,115,22,0.3)] disabled:opacity-50 disabled:hover:scale-100"
          >
            <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
            {isAnalyzing ? (
              <span className="flex items-center gap-2 relative z-10">
                <Loader2 className="w-5 h-5 animate-spin" /> Deep Inference Running...
              </span>
            ) : (
              <span className="relative z-10">Analyze Trade</span>
            )}
          </button>
        </div>

        {/* Results Area */}
        <AnimatePresence mode="wait">
          {tradeAnalysis && !isAnalyzing && (
            <motion.div
              layout
              initial={{ opacity: 0, scale: 0.95, filter: 'blur(10px)' }}
              animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
              exit={{ opacity: 0, scale: 0.95, filter: 'blur(10px)' }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            >
              <h2 className="text-3xl font-black text-white text-center mb-12 tracking-tight">AI Evaluation Report</h2>
              <ResultsDashboard data={tradeAnalysis} />
            </motion.div>
          )}
        </AnimatePresence>

      </div>

      {/* Full Screen Loading Overlay */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-xl"
          >
            <div className="flex flex-col items-center">
              <div className="relative w-24 h-24 mb-8">
                <div className="absolute inset-0 border-4 border-orange-500/20 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-orange-400 rounded-full border-t-transparent animate-spin"></div>
                <div className="absolute inset-2 border-4 border-rose-500/20 rounded-full"></div>
                <div className="absolute inset-2 border-4 border-rose-400 rounded-full border-b-transparent animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Activity className="w-8 h-8 text-orange-400 animate-pulse" />
                </div>
              </div>
              <h2 className="text-2xl font-black text-white bg-clip-text text-transparent bg-gradient-to-r from-orange-400 to-rose-400">
                Processing ML Inference
              </h2>
              <p className="text-slate-400 mt-2 font-mono text-sm">Evaluating 54 complex temporal features...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
}

export default App;
