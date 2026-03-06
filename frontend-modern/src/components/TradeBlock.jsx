import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ArrowRightLeft } from 'lucide-react';

function PlayerTag({ player, onRemove, side }) {
    return (
        <motion.div
            layout
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.85 }}
            className={`flex items-center justify-between px-3 py-2 rounded-lg border ${side === 'A'
                    ? 'bg-cyan-950/40 border-cyan-500/30 text-cyan-200'
                    : 'bg-violet-950/40 border-violet-500/30 text-violet-200'
                } text-sm font-medium mb-1.5`}
        >
            <span className="truncate">{player.player_name}</span>
            <button
                onClick={() => onRemove(player.player_name, side)}
                className="ml-2 flex-shrink-0 text-slate-500 hover:text-red-400 transition-colors"
            >
                <X className="w-3.5 h-3.5" />
            </button>
        </motion.div>
    );
}

export default function TradeBlock({ teamAPlayers, teamBPlayers, onRemove }) {
    return (
        <div className="w-full glass-panel p-5">
            <div className="flex items-center justify-center gap-3 mb-5">
                <div className="h-px flex-1 bg-gradient-to-r from-transparent to-slate-700" />
                <div className="flex items-center gap-2 text-slate-400 text-sm font-semibold tracking-widest uppercase">
                    <ArrowRightLeft className="w-4 h-4 text-cyan-500" />
                    Trade Block
                </div>
                <div className="h-px flex-1 bg-gradient-to-l from-transparent to-slate-700" />
            </div>

            <div className="grid grid-cols-2 gap-4">
                {/* Team A giving away */}
                <div>
                    <p className="text-xs font-bold uppercase tracking-widest text-cyan-500/70 mb-2">Team A outgoing</p>
                    <div className="min-h-[80px] bg-slate-900/40 rounded-xl p-2 border border-slate-700/40">
                        <AnimatePresence>
                            {teamAPlayers.length === 0 ? (
                                <p className="text-slate-600 text-xs text-center pt-4">Add players from Team A →</p>
                            ) : (
                                teamAPlayers.map(p => <PlayerTag key={p.player_name} player={p} onRemove={onRemove} side="A" />)
                            )}
                        </AnimatePresence>
                    </div>
                </div>

                {/* Team B giving away */}
                <div>
                    <p className="text-xs font-bold uppercase tracking-widest text-violet-500/70 mb-2">Team B outgoing</p>
                    <div className="min-h-[80px] bg-slate-900/40 rounded-xl p-2 border border-slate-700/40">
                        <AnimatePresence>
                            {teamBPlayers.length === 0 ? (
                                <p className="text-slate-600 text-xs text-center pt-4">← Add players from Team B</p>
                            ) : (
                                teamBPlayers.map(p => <PlayerTag key={p.player_name} player={p} onRemove={onRemove} side="B" />)
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>
        </div>
    );
}
