import React, { useState, useEffect } from 'react';
import { Loader2, PlusCircle, Users } from 'lucide-react';
import { getPlayersByTeam } from '../api';

const TeamPanel = ({ title, team, setTeam, otherTeam, allTeams, activePlayers, onAddPlayer }) => {
    const [roster, setRoster] = useState([]);
    const [isLoadingRoster, setIsLoadingRoster] = useState(false);
    const [selectedPlayerName, setSelectedPlayerName] = useState("");

    // Fetch players automatically when the Team changes
    useEffect(() => {
        if (!team) {
            setRoster([]);
            setSelectedPlayerName("");
            return;
        }

        const fetchRoster = async () => {
            setIsLoadingRoster(true);
            try {
                const data = await getPlayersByTeam(team);
                setRoster(data.players || []);
                setSelectedPlayerName(""); // reset dropdown
            } catch (error) {
                console.error("Failed to load roster:", error);
                setRoster([]);
            } finally {
                setIsLoadingRoster(false);
            }
        };

        fetchRoster();
    }, [team]);

    const handleAddClick = () => {
        if (!selectedPlayerName) return;
        const playerObj = roster.find(p => p.player_name === selectedPlayerName);
        if (playerObj) {
            onAddPlayer(playerObj);
            setSelectedPlayerName(""); // clear selection after adding
        }
    };

    const isPlayerAlreadyInTrade = (playerName) => {
        return activePlayers.some(p => p.player_name === playerName);
    };

    return (
        <div className="glass-panel p-4 md:p-6 flex flex-col gap-4 md:gap-5 border border-slate-800">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3 mb-1">
                <h3 className="text-xl font-bold text-white tracking-wide">{title}</h3>
                <div className="flex items-center gap-2 bg-slate-900/50 px-3 py-1.5 rounded-lg border border-slate-700/50 w-full sm:w-auto">
                    <Users className="w-4 h-4 text-orange-400 shrink-0" />
                    <select
                        className="bg-transparent text-white font-medium focus:outline-none focus:ring-0 appearance-none cursor-pointer w-full"
                        value={team}
                        onChange={(e) => setTeam(e.target.value)}
                    >
                        <option value="" className="bg-slate-900 text-slate-400">Select Franchise...</option>
                        {allTeams.map(t => (
                            <option
                                key={t.code}
                                value={t.code}
                                disabled={t.code === otherTeam}
                                className="bg-slate-900 text-white"
                            >
                                {t.name}
                            </option>
                        ))}
                    </select>
                </div>
            </div>

            <div className="flex flex-col gap-2 relative z-10">
                <label className="text-[10px] md:text-xs uppercase font-bold text-slate-500 tracking-wider">Select Player from Roster</label>
                <div className="flex flex-col sm:flex-row gap-3">
                    <div className="relative flex-1">
                        <select
                            className="w-full bg-slate-900/80 border border-slate-700 text-slate-200 rounded-xl px-4 py-3 appearance-none focus:outline-none focus:border-orange-500 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed text-sm md:text-base"
                            value={selectedPlayerName}
                            onChange={(e) => setSelectedPlayerName(e.target.value)}
                            disabled={!team || isLoadingRoster}
                        >
                            {!team ? (
                                <option value="">Select a team first...</option>
                            ) : isLoadingRoster ? (
                                <option value="">Loading roster...</option>
                            ) : roster.length === 0 ? (
                                <option value="">No players found</option>
                            ) : (
                                <>
                                    <option value="" disabled>Choose a player to trade...</option>
                                    {roster.map(p => {
                                        const isAdded = isPlayerAlreadyInTrade(p.player_name);
                                        return (
                                            <option
                                                key={p.player_name}
                                                value={p.player_name}
                                                disabled={isAdded}
                                                className="bg-slate-900"
                                            >
                                                {p.player_name} {isAdded ? '(Already in Trade)' : `(${p.points_per_game.toFixed(1)} PPG)`}
                                            </option>
                                        );
                                    })}
                                </>
                            )}
                        </select>
                        <div className="absolute inset-y-0 right-0 flex items-center px-4 pointer-events-none text-slate-400">
                            {isLoadingRoster ? <Loader2 className="w-4 h-4 animate-spin text-orange-500" /> : "▼"}
                        </div>
                    </div>

                    <button
                        onClick={handleAddClick}
                        disabled={!selectedPlayerName}
                        className="w-full sm:w-auto px-6 py-3 bg-slate-800 hover:bg-orange-600 border border-slate-700 hover:border-orange-500 text-white rounded-xl transition-all disabled:opacity-40 disabled:hover:bg-slate-800 disabled:hover:border-slate-700 flex items-center justify-center gap-2 font-bold shadow-lg shrink-0"
                    >
                        <PlusCircle className="w-5 h-5" /> Add
                    </button>
                </div>

                {/* Player Preview Selection - Task Request: "whenever a player is selected his headshot should be shown first" */}
                {selectedPlayerName && (
                    <div className="mt-4 flex items-center gap-4 p-4 bg-slate-900/40 rounded-xl border border-slate-800 animate-in fade-in slide-in-from-top-2 duration-300">
                        <div className="relative w-20 h-20 md:w-24 md:h-24 rounded-full overflow-hidden border-2 border-orange-500/30 bg-slate-900 shrink-0">
                            <img 
                                src={roster.find(p => p.player_name === selectedPlayerName)?.photo_url} 
                                alt={selectedPlayerName}
                                className="w-full h-full object-cover"
                                onError={(e) => {
                                    e.target.onerror = null;
                                    e.target.src = "https://www.nba.com/assets/logos/teams/primary/web/NBA.svg"; // Fallback to NBA logo
                                    e.target.style.padding = "1rem";
                                }}
                            />
                        </div>
                        <div className="flex flex-col">
                            <span className="text-lg md:text-xl font-black text-white">{selectedPlayerName}</span>
                            <div className="flex items-center gap-2">
                                <span className="text-xs font-bold text-orange-400 bg-orange-400/10 px-2 py-0.5 rounded uppercase tracking-tighter">
                                    Ready to Trade
                                </span>
                                <span className="text-[10px] text-slate-500 font-medium">Click "Add" to evaluate impact</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default TeamPanel;
