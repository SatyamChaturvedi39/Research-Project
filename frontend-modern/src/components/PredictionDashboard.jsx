import React from 'react';
import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

function StatBar({ label, current, predicted, unit, confidence }) {
    const diff = predicted - current;
    const isUp = diff > 0.1;
    const isDown = diff < -0.1;

    return (
        <div className="py-2.5 border-b border-slate-800 last:border-0">
            <div className="flex justify-between items-center mb-1.5">
                <span className="text-xs text-slate-400 font-medium uppercase tracking-wide">{label}</span>
                <div className="flex items-center gap-2">
                    <span className="text-slate-500 text-xs line-through">{current.toFixed(1)}</span>
                    <span className="text-white text-sm font-bold">{predicted.toFixed(1)} <span className="text-slate-500 font-normal text-xs">{unit}</span></span>
                    <span className={`text-xs font-bold flex items-center gap-0.5 w-12 justify-end ${isUp ? 'text-emerald-400' : isDown ? 'text-red-400' : 'text-slate-500'}`}>
                        {isUp ? <TrendingUp className="w-3 h-3" /> : isDown ? <TrendingDown className="w-3 h-3" /> : <Minus className="w-3 h-3" />}
                        {isUp ? '+' : ''}{diff.toFixed(1)}
                    </span>
                </div>
            </div>
            {/* Simple progress bars showing current vs predicted */}
            <div className="h-1 rounded-full bg-slate-800 overflow-hidden">
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, (predicted / (current * 1.5 + 0.1)) * 100)}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                    className={`h-full rounded-full ${isUp ? 'bg-emerald-500' : isDown ? 'bg-red-500' : 'bg-slate-500'}`}
                />
            </div>
            {confidence && (
                <p className="text-xs text-slate-600 mt-1">
                    Range: {confidence.lower}–{confidence.upper} {unit} · {confidence.confidence_label} confidence
                </p>
            )}
        </div>
    );
}

function ShapChart({ factors }) {
    // Keep original sorting (by absolute SHAP magnitude descending)
    const data = factors.slice(0, 5).map((f) => ({
        name: f.label,
        value: f.shap,
        direction: f.direction,
        raw_value: f.raw_value,
        originalReason: f.reason
    }));

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const data = payload[0].payload;
            return (
                <div className="bg-slate-900 border border-slate-700 p-3 rounded-lg shadow-xl text-xs max-w-xs z-50">
                    <p className="font-bold text-white mb-1">{data.name}</p>
                    <p className="text-slate-400 mb-2">Raw Value: <span className="text-white font-medium">{data.raw_value}</span></p>
                    <p className="text-slate-400 mb-1">
                        Model Impact: <span className={`font-bold ${data.value > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {data.value > 0 ? '+' : ''}{data.value.toFixed(2)}
                        </span>
                    </p>
                    <p className="text-slate-500 italic mt-2 border-t border-slate-800 pt-2">{data.originalReason}</p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="h-56 w-full mt-2">
            <ResponsiveContainer width="100%" height="100%">
                <BarChart
                    layout="vertical"
                    data={data}
                    margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
                >
                    <XAxis type="number" hide domain={['dataMin - 0.5', 'dataMax + 0.5']} />
                    <YAxis
                        type="category"
                        dataKey="name"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#94a3b8', fontSize: 11 }}
                        width={110}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                    <ReferenceLine x={0} stroke="#475569" strokeDasharray="3 3" />
                    <Bar dataKey="value" barSize={16} radius={4}>
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#10b981' : '#f43f5e'} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </div>
    );
}

function PlayerCard({ result, index }) {
    if (result.error) {
        return (
            <div className="glass-panel p-5 flex items-center gap-3 text-red-400">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <p className="text-sm">{result.player_name}: {result.error}</p>
            </div>
        );
    }

    const cs = result.current_stats;
    const p = result.predictions;
    const cr = result.confidence_ranges;
    // shap_explanation is keyed by e.g. 'ppg', 'rpg', etc.
    const shapPpg = result.shap_explanation?.ppg;

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="glass-panel p-5 space-y-4"
        >
            {/* Player header */}
            <div className="flex justify-between items-start">
                <div>
                    <h4 className="text-lg font-bold text-white">{result.player_name}</h4>
                    <p className="text-sm text-slate-400">{result.team} · Age {result.current_age}</p>
                </div>
                <div className="text-right">
                    <p className="text-xs text-slate-500 uppercase tracking-widest">Model Context</p>
                    <p className="text-xs font-semibold text-slate-400">5-Year History</p>
                </div>
            </div>

            {/* Stats */}
            <div className="bg-slate-900/50 rounded-xl px-4 py-1 border border-slate-800">
                <StatBar label="Points" current={cs.ppg} predicted={p.ppg} unit="ppg" confidence={cr?.ppg} />
                <StatBar label="Rebounds" current={cs.rpg} predicted={p.rpg} unit="rpg" confidence={cr?.rpg} />
                <StatBar label="Assists" current={cs.apg} predicted={p.apg} unit="apg" confidence={cr?.apg} />
                <StatBar
                    label="Minutes"
                    current={result.current_stats.mpg ?? 0}
                    predicted={p.mpg}
                    unit="min"
                    confidence={cr?.mpg}
                />
            </div>

            {/* SHAP reasons UI */}
            {shapPpg?.top_factors?.length > 0 && (
                <div className="pt-2">
                    <p className="text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">Key Drivers (PPG Impact)</p>
                    <div className="bg-slate-900/50 rounded-xl p-2 border border-slate-800">
                        <ShapChart factors={shapPpg.top_factors} />
                    </div>
                </div>
            )}
        </motion.div>
    );
}

export default function PredictionDashboard({ results }) {
    if (!results || results.length === 0) return null;

    return (
        <div className="w-full space-y-5 mt-8">
            <div className="flex items-center gap-3">
                <div className="h-px flex-1 bg-slate-800" />
                <h3 className="text-sm font-bold uppercase tracking-widest text-slate-400">Trade Analysis</h3>
                <div className="h-px flex-1 bg-slate-800" />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {results.map((r, i) => <PlayerCard key={r.player_name || i} result={r} index={i} />)}
            </div>
        </div>
    );
}
