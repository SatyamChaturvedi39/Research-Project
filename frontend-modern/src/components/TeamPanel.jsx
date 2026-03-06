import React, { useState, useEffect } from 'react';
import { fetchTeams, fetchPlayers } from '../api';
import { Search, Plus, ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function TeamPanel({ side, onAddPlayer, otherSelectedTeam, onTeamChange }) {
    const [teams, setTeams] = useState([]);
    const [selectedTeam, setSelectedTeam] = useState('');
    const [players, setPlayers] = useState([]);
    const [search, setSearch] = useState('');
    const [loadingPlayers, setLoadingPlayers] = useState(false);

    useEffect(() => {
        fetchTeams().then(data => {
            if (data.teams) {
                setTeams(data.teams.sort((a, b) => a.name.localeCompare(b.name)));
            }
        });
    }, []);

    useEffect(() => {
        if (!selectedTeam) { setPlayers([]); return; }
        setLoadingPlayers(true);
        fetchPlayers(selectedTeam).then(data => {
            setPlayers(data.players || []);
            setLoadingPlayers(false);
        });
    }, [selectedTeam]);

    const filtered = players.filter(p =>
        p.player_name.toLowerCase().includes(search.toLowerCase())
    );

    const accent = side === 'A' ? 'border-cyan-500/40' : 'border-violet-500/40';
    const labelColor = side === 'A' ? 'text-cyan-400' : 'text-violet-400';

    return (
        <div className={`flex flex-col w-full lg:w-72 glass-panel overflow-hidden h-[680px] ${accent}`}>
            {/* Panel Header */}
            <div className="px-4 pt-4 pb-3 border-b border-slate-700/50">
                <p className={`text-xs font-bold uppercase tracking-widest mb-2 ${labelColor}`}>
                    Team {side}
                </p>

                {/* Team Selector */}
                <div className="relative">
                    <select
                        className="w-full appearance-none bg-slate-800 text-white text-sm border border-slate-600/70 rounded-lg px-3 py-2.5 pr-8 outline-none focus:ring-1 focus:ring-cyan-500/60 transition-all cursor-pointer"
                        value={selectedTeam}
                        onChange={e => {
                            const newTeam = e.target.value;
                            setSelectedTeam(newTeam);
                            setSearch('');
                            if (onTeamChange) onTeamChange(newTeam);
                        }}
                    >
                        <option value="">— Select a team —</option>
                        {teams.map(t => (
                            <option
                                key={t.code}
                                value={t.code}
                                disabled={otherSelectedTeam === t.code}
                            >
                                {t.name}
                            </option>
                        ))}
                    </select>
                    <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
                </div>

                {/* Search box */}
                {selectedTeam && (
                    <div className="relative mt-2">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
                        <input
                            type="text"
                            placeholder="Search players..."
                            className="w-full bg-slate-900/60 border border-slate-700/50 rounded-lg text-sm py-2 pl-8 pr-3 outline-none focus:ring-1 focus:ring-cyan-500/40 text-slate-200 placeholder-slate-600"
                            value={search}
                            onChange={e => setSearch(e.target.value)}
                        />
                    </div>
                )}
            </div>

            {/* Player List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-slate-700">
                {loadingPlayers ? (
                    <div className="flex justify-center py-10">
                        <div className="w-5 h-5 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
                    </div>
                ) : !selectedTeam ? (
                    <p className="text-center text-slate-600 text-sm py-12">Select a team above</p>
                ) : filtered.length === 0 ? (
                    <p className="text-center text-slate-600 text-sm py-12">No players found</p>
                ) : (
                    <AnimatePresence>
                        {filtered.map(player => (
                            <motion.div
                                layout
                                initial={{ opacity: 0, y: 6 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -6 }}
                                key={player.player_name}
                                onClick={() => onAddPlayer(player)}
                                className="flex items-center justify-between px-3 py-2.5 rounded-lg cursor-pointer group hover:bg-slate-700/50 transition-colors"
                            >
                                <div className="min-w-0">
                                    <p className="text-sm font-semibold text-slate-200 group-hover:text-white truncate transition-colors">
                                        {player.player_name}
                                    </p>
                                    <p className="text-xs text-slate-500 mt-0.5">
                                        {player.points_per_game.toFixed(1)} PPG · Age {player.age}
                                    </p>
                                </div>
                                <div className="ml-2 p-1 rounded text-slate-500 group-hover:text-cyan-400 transition-colors flex-shrink-0">
                                    <Plus className="w-4 h-4" />
                                </div>
                            </motion.div>
                        ))}
                    </AnimatePresence>
                )}
            </div>
        </div>
    );
}
