import React, { useState } from 'react';
import { X, TrendingUp, ShieldAlert, Loader2, ChevronDown, ChevronUp, Activity } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import clsx from 'clsx';
import { twMerge } from 'tailwind-merge';

// Helper function to match Tailwind conditional classes
function cn(...inputs) {
    return twMerge(clsx(...inputs));
}

const ShapFactor = ({ factor, maxAbs }) => {
    const isPos = factor.direction === 'positive';
    const barPct = Math.round((Math.abs(factor.shap) / maxAbs) * 100);

    return (
        <div className="flex flex-col gap-1 mb-2">
            <div className="flex justify-between text-xs items-center">
                <span className="text-slate-300 truncate max-w-[150px]" title={factor.feature}>{factor.label}</span>
                <span className={cn("font-mono font-bold", isPos ? "text-emerald-400" : "text-rose-400")}>
                    {isPos ? '▲ +' : '▼ −'}{Math.abs(factor.shap).toFixed(2)}
                </span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
                <div style={{ width: `${barPct}%` }} className={cn("h-full rounded-full", isPos ? "bg-emerald-500" : "bg-rose-500")}></div>
            </div>
            <div className="text-[10px] text-slate-500 italic leading-tight">{factor.reason}</div>
        </div>
    );
};

const PlayerCard = ({ p, onRemove }) => {
    const [showShap, setShowShap] = useState(false);

    if (p.isLoadingPrediction) {
        return (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                className="bg-slate-800/50 border border-slate-700/50 rounded-xl p-6 flex flex-col items-center justify-center gap-3">
                <div className="flex-1 flex justify-center items-center py-12">
                    <Loader2 className="w-6 h-6 animate-spin text-orange-400" />
                </div>
                <span className="text-sm font-medium text-slate-400 animate-pulse">Running ML Inference on {p.player_name}...</span>
            </motion.div>
        );
    }

    const confObj = p.confidence_ranges?.ppg;
    const confLabel = confObj?.confidence_label || 'medium';

    const shapFactors = p.shap_explanation?.ppg?.top_factors || [];
    const maxAbsShap = shapFactors.length > 0 ? Math.max(...shapFactors.map(f => Math.abs(f.shap)), 0.001) : 1;

    // Medical colors logic
    const medColors = {
        'EXCELLENT': 'text-emerald-400',
        'GOOD': 'text-emerald-300',
        'FAIR': 'text-amber-400',
        'POOR': 'text-rose-400',
        'CRITICAL': 'text-rose-600'
    };
    const medColorClass = medColors[p.medical_grade] || 'text-slate-300';

    return (
        <motion.div layout initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.9 }}
            className="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden group">

            {/* Header Block */}
            <div className="p-4 border-b border-slate-800 bg-slate-800/30 flex justify-between items-start">
                <div>
                    <div className="font-bold text-lg text-white tracking-wide">{p.player_name}</div>
                    <div className="text-xs text-slate-400 flex items-center gap-2 mt-1">
                        <span>{p.team}</span>
                        <span>•</span>
                        <span>Age {p.current_age}</span>
                        <span>•</span>
                        <span className={cn("font-bold flex items-center gap-1", medColorClass)}>
                            <Activity className="w-3 h-3" /> {p.medical_grade}
                        </span>
                    </div>
                </div>
                <div className="flex flex-col items-end gap-2">
                    <button onClick={() => onRemove(p)} className="text-slate-500 hover:text-red-400 transition-colors p-1">
                        <X className="w-5 h-5" />
                    </button>
                    <div className="flex gap-2">
                        <span className={cn("text-[9px] uppercase font-black px-2 py-0.5 rounded border tracking-wider",
                            confLabel === 'low' ? 'bg-rose-500/10 border-rose-500/30 text-rose-400' :
                                confLabel === 'high' ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' :
                                    'bg-amber-500/10 border-amber-500/30 text-amber-400'
                        )}>{confLabel} CONF</span>
                        <span className={cn("text-[9px] uppercase font-black px-2 py-0.5 rounded border tracking-wider",
                            p.injury_risk_category === 'High' ? 'bg-rose-500/20 border-rose-500/50 text-rose-400' :
                                p.injury_risk_category === 'Medium' ? 'bg-amber-500/20 border-amber-500/50 text-amber-400' :
                                    'bg-emerald-500/20 border-emerald-500/50 text-emerald-400'
                        )}>{p.injury_risk_category} RISK</span>
                    </div>
                </div>
            </div>

            {/* Core Stats Block */}
            <div className="p-4 grid grid-cols-4 gap-2 bg-slate-900/50 text-center divide-x divide-slate-800 items-center">
                <div className="flex flex-col">
                    <span className="text-2xl font-light text-slate-300">{p.current_stats?.ppg?.toFixed(1) || '?'}</span>
                    <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mt-1">Prior PPG</span>
                </div>
                <div className="flex flex-col items-center justify-center p-3 relative group/stat cursor-default">
                    <div className="absolute inset-0 bg-orange-500/5 shadow-[0_0_15px_rgba(249,115,22,0.1)] rounded-lg pointer-events-none"></div>
                    <span className="text-3xl font-black text-white relative z-10">{p.predictions?.ppg?.toFixed(1) || '--'}</span>
                    <span className="text-[10px] text-orange-400 uppercase font-bold tracking-wider relative z-10 mt-1">Pred PPG</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-xl font-medium text-slate-200">{p.predictions?.rpg?.toFixed(1) || '?'}</span>
                    <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mt-1">Pred RPG</span>
                </div>
                <div className="flex flex-col">
                    <span className="text-xl font-medium text-slate-200">{p.predictions?.apg?.toFixed(1) || '?'}</span>
                    <span className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mt-1">Pred APG</span>
                </div>
            </div>

            {/* Inference Bounds */}
            {confObj && (
                <div className="px-5 py-3 bg-slate-900 border-t border-slate-800 flex justify-between items-center">
                    <div className="text-xs text-slate-400">90% CI Scoring Range:</div>
                    <div className="font-mono text-sm font-bold text-slate-300">
                        {confObj.lower.toFixed(1)} <span className="text-slate-600 mx-1">—</span> {confObj.upper.toFixed(1)} <span className="text-xs text-slate-500">PPG</span>
                    </div>
                </div>
            )}

            {/* Collapsible SHAP Block */}
            {shapFactors.length > 0 && (
                <div className="border-t border-slate-800/80 bg-[#0f172a]">
                    <button
                        onClick={() => setShowShap(!showShap)}
                        className="w-full flex justify-between items-center px-5 py-3 text-[10px] uppercase font-bold text-slate-500 tracking-widest hover:bg-slate-800/50 transition-colors"
                    >
                        <span><span className="text-orange-300 mr-2">🤖</span> AI Reasoning Factors</span>
                        {showShap ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </button>
                    <AnimatePresence>
                        {showShap && (
                            <motion.div
                                initial={{ height: 0, opacity: 0 }}
                                animate={{ height: 'auto', opacity: 1 }}
                                exit={{ height: 0, opacity: 0 }}
                                className="overflow-hidden bg-slate-950/50 px-5 pb-4 pt-1"
                            >
                                <div className="text-[10px] text-slate-500 mb-3 italic">
                                    Baseline trajectory: {p.shap_explanation.ppg.base_value.toFixed(1)} PPG
                                </div>
                                {shapFactors.map((factor, i) => (
                                    <ShapFactor key={i} factor={factor} maxAbs={maxAbsShap} />
                                ))}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            )}
        </motion.div>
    );
};

const TradeBlock = ({ title, players, onRemove, color = "orange" }) => {
    return (
        <div className={`mt-4 border border-${color}-900/40 rounded-xl bg-slate-900/30 p-4 min-h-[150px]`}>
            <h4 className={`text-xs font-bold uppercase tracking-wider text-${color}-400 mb-4`}>{title}</h4>

            <AnimatePresence mode="popLayout">
                {players.length === 0 && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="text-sm text-slate-500 italic text-center py-4"
                    >
                        No players added yet. Search above to add.
                    </motion.div>
                )}

                <div className="flex flex-col gap-4">
                    {players.map((p) => (
                        <PlayerCard key={p.normalized_name || p.player_name} p={p} onRemove={onRemove} />
                    ))}
                </div>
            </AnimatePresence>
        </div>
    );
};

export default TradeBlock;
