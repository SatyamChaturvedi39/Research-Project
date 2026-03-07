import React from 'react';
import { motion } from 'framer-motion';
import { Brain, BarChart3, Scale, Cog, Activity, Target } from 'lucide-react';

const ModelInfo = () => {
    const perfMetrics = [
        { label: 'PPG Prediction', r2: '0.778', mae: '2.49 pts', rmse: '3.24 pts', tag: 'Points/Game' },
        { label: 'RPG Prediction', r2: '0.682', mae: '1.05 reb', rmse: '1.40 reb', tag: 'Rebounds/Game' },
        { label: 'APG Prediction', r2: '0.776', mae: '0.65 ast', rmse: '0.91 ast', tag: 'Assists/Game' },
        { label: 'MPG Prediction', r2: '0.592', mae: '5.09 min', rmse: '6.83 min', tag: 'Minutes/Game' },
        { label: 'TS% Prediction', r2: '0.291', mae: '0.117', rmse: '0.176', tag: 'True Shooting %' },
    ];

    return (
        <div className="text-white space-y-12 pb-20">
            {/* Hero */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center space-y-4"
            >
                <div className="text-orange-400 font-bold tracking-[0.2em] uppercase text-sm">Deep Learning & Interpretability</div>
                <h1 className="text-5xl md:text-7xl font-black">Model Methodology</h1>
                <p className="text-slate-400 max-w-2xl mx-auto text-lg">
                    Inside the 54-feature ensemble engine predicting the future of the NBA.
                </p>
            </motion.div>

            {/* Architecture Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {[
                    { icon: <Brain />, title: 'The Engine', desc: 'A Multi-Output XGBoost GBM that models dependencies between PPG, RPG, APG, MPG, and Shooting Efficiency simultaneously.' },
                    { icon: <BarChart3 />, title: 'Training Data', desc: 'Trained on 2,034 player-season records (2020–2025), with 1,627 training and 407 test samples to learn aging curves and usage transitions.' },
                    { icon: <Scale />, title: 'Explainability', desc: 'Powered by SHAP, calculating the exact contribution of each of the 54 features for every prediction made.' }
                ].map((item, i) => (
                    <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="glass-panel p-8 space-y-4"
                    >
                        <div className="text-orange-400">{item.icon}</div>
                        <h3 className="text-2xl font-bold">{item.title}</h3>
                        <p className="text-slate-400 leading-relaxed">{item.desc}</p>
                    </motion.div>
                ))}
            </div>

            {/* Performance Model Metrics */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="space-y-6"
            >
                <div className="flex items-center gap-3">
                    <Target className="text-orange-400" />
                    <h2 className="text-3xl font-black">Performance Model Evaluation</h2>
                </div>
                <p className="text-slate-400 text-sm max-w-3xl">
                    Evaluated on a held-out test set (20% split, 407 samples) using MAE, RMSE, and R² score. Model: Multi-Output XGBoost Regressor, 54 features, trained on 5 seasons of NBA data (2020–2025).
                </p>

                {/* Metric Table */}
                <div className="overflow-x-auto">
                    <table className="w-full text-sm border-collapse">
                        <thead>
                            <tr className="border-b border-slate-700">
                                <th className="text-left py-3 px-4 text-[10px] uppercase tracking-wider text-slate-500 font-black">Target</th>
                                <th className="text-center py-3 px-4 text-[10px] uppercase tracking-wider text-slate-500 font-black">R² Score</th>
                                <th className="text-center py-3 px-4 text-[10px] uppercase tracking-wider text-slate-500 font-black">MAE (Test)</th>
                                <th className="text-center py-3 px-4 text-[10px] uppercase tracking-wider text-slate-500 font-black">RMSE (Test)</th>
                                <th className="text-left py-3 px-4 text-[10px] uppercase tracking-wider text-slate-500 font-black">Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            {perfMetrics.map((m, i) => (
                                <tr key={i} className={`border-b border-slate-800 ${i % 2 === 0 ? 'bg-slate-900/20' : ''}`}>
                                    <td className="py-3 px-4 font-semibold text-white">{m.label}</td>
                                    <td className="py-3 px-4 text-center">
                                        <span className={`font-black text-lg ${parseFloat(m.r2) >= 0.7 ? 'text-emerald-400' : parseFloat(m.r2) >= 0.5 ? 'text-yellow-400' : 'text-rose-400'}`}>
                                            {m.r2}
                                        </span>
                                    </td>
                                    <td className="py-3 px-4 text-center text-slate-300">{m.mae}</td>
                                    <td className="py-3 px-4 text-center text-slate-300">{m.rmse}</td>
                                    <td className="py-3 px-4 text-slate-500 text-xs">{m.tag}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Legend */}
                <div className="flex flex-wrap gap-4 text-xs text-slate-500">
                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-400 inline-block"></span> R² ≥ 0.7 — Strong fit</span>
                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-400 inline-block"></span> R² 0.5–0.7 — Moderate fit</span>
                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-rose-400 inline-block"></span> R² &lt; 0.5 — Weak predictability</span>
                </div>
            </motion.div>

            {/* Injury Model Metrics */}
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.45 }}
                className="space-y-6"
            >
                <div className="flex items-center gap-3">
                    <Activity className="text-orange-400" />
                    <h2 className="text-3xl font-black">Injury Risk Model Evaluation</h2>
                </div>
                <p className="text-slate-400 text-sm max-w-3xl">
                    XGBoost Classifier trained on injury history data (2020–2025). Predicts whether a player will miss ≥15 days next season. Evaluated on 586 test samples.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    {[
                        { label: 'Accuracy', val: '81%', sub: 'Overall correct predictions' },
                        { label: 'ROC AUC', val: '0.888', sub: 'Discrimination power' },
                        { label: 'F1 (Injured)', val: '0.87', sub: 'Injury class (class 1)' },
                        { label: 'F1 (Healthy)', val: '0.65', sub: 'Healthy class (class 0)' },
                    ].map((m, i) => (
                        <div key={i} className="bg-slate-900/40 p-6 rounded-2xl border-l-4 border-orange-500">
                            <div className="text-[10px] uppercase tracking-wider text-slate-500 mb-1">{m.label}</div>
                            <div className="text-3xl font-black">{m.val}</div>
                            <div className="text-[10px] text-slate-600 mt-1">{m.sub}</div>
                        </div>
                    ))}
                </div>
            </motion.div>

            {/* Monte Carlo */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
                className="glass-panel p-10 border-orange-500/20 bg-gradient-to-br from-slate-900/60 to-orange-500/5"
            >
                <div className="flex items-center gap-4 mb-6">
                    <Cog className="text-orange-400" />
                    <h3 className="text-2xl font-bold">Monte Carlo Simulation</h3>
                </div>
                <p className="text-slate-300 leading-relaxed mb-6">
                    Trade analysis runs 1,000 iterations of the next season, sampling from:
                </p>
                <ul className="space-y-4 text-slate-400">
                    <li className="flex gap-3"><span className="text-orange-500">■</span> <strong>Injury Probabilities:</strong> Derived from the XGBoost classifier (ROC AUC: 0.888) trained on 5 years of medical data.</li>
                    <li className="flex gap-3"><span className="text-orange-500">■</span> <strong>Performance Variance:</strong> Standard error from the multi-output regressor used to sample ceiling and floor outcomes.</li>
                    <li className="flex gap-3"><span className="text-orange-500">■</span> <strong>Roster Synergy:</strong> Fit heuristic evaluating complementarity vs overlap across PPG, RPG, and APG dimensions.</li>
                </ul>
            </motion.div>
        </div>
    );
};

export default ModelInfo;
