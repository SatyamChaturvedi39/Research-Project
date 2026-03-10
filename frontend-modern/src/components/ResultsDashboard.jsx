import React, { useRef, useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { Activity, ShieldAlert, Cpu, ArrowRightLeft } from 'lucide-react';

// Replicating the exact DOM structure and logic from script.js
const ResultsDashboard = ({ data }) => {
    const containerRef = useRef(null);
    const [score, setScore] = useState(0);

    const targetScore = data.team_a.score;

    // Playoff Probability Formula — mirrors backend wins_to_playoff_prob()
    // Backend: if wins >= 42: clip((wins-42)*0.05 + 0.5, 0.5, 1.0)
    //          else:          clip(0.5 - (42-wins)*0.05, 0.0, 0.5)
    const getPlayoffProb = (wins) => {
        let prob;
        if (wins >= 42) {
            prob = Math.min(1.0, (wins - 42) * 0.05 + 0.5);
        } else {
            prob = Math.max(0.0, 0.5 - (42 - wins) * 0.05);
        }
        // Convert to percentage, clamp to [0.1, 99.9] to avoid absolute extremes
        return Math.min(99.9, Math.max(0.1, prob * 100)).toFixed(1);
    };

    useGSAP(() => {
        gsap.to({ val: 0 }, {
            val: targetScore,
            duration: 2.5,
            ease: "power3.out",
            onUpdate: function () {
                setScore(Math.round(this.targets()[0].val));
            }
        });
    }, [targetScore]);

    const formatCI = (ci) => `[${ci[0] > 0 ? '+' : ''}${ci[0]}, +${ci[1]}]`;
    const formatDelta = (val, unit, invert = false) => {
        const isPos = val > 0;
        const isNeg = val < 0;
        let colorClass = "";

        if (invert) {
            colorClass = isPos ? "text-rose-400" : (isNeg ? "text-emerald-400" : "text-slate-400");
        } else {
            colorClass = isPos ? "text-emerald-400" : (isNeg ? "text-rose-400" : "text-slate-400");
        }

        return <span className={`font-mono ${colorClass}`}>{isPos ? '+' : ''}{val.toFixed(1)}{unit}</span>;
    };

    const gradeColor = (grade) => {
        const colors = {
            'BENEFICIAL': 'text-emerald-400',
            'SLIGHTLY POSITIVE': 'text-emerald-300',
            'NEUTRAL': 'text-amber-400',
            'SLIGHTLY NEGATIVE': 'text-orange-400',
            'HARMFUL': 'text-rose-400'
        };
        return colors[grade] || 'text-white';
    };

    const scoreColorClass = score >= 70 ? 'text-emerald-400' : score >= 50 ? 'text-amber-400' : score >= 30 ? 'text-orange-400' : 'text-rose-400';

    const fairnessLabel = data.assessment.fairness > 90 ? 'VERY FAIR' : data.assessment.fairness > 75 ? 'FAIR' : 'ONE-SIDED';

    const renderOutcomeCard = (teamData, label) => (
        <div className="bg-slate-900 border border-slate-700 rounded-xl overflow-hidden shadow-lg flex-1 w-full">
            <div className="p-3 md:p-4 bg-slate-800/50 border-b border-slate-700 flex justify-center text-2xl md:text-3xl font-black text-white tracking-widest">
                {teamData.code}
            </div>
            <div className="p-4 md:p-6 flex flex-col gap-4">
                <div className="flex justify-between items-end">
                    <span className="text-slate-400 font-bold uppercase text-[10px] md:text-xs">Expected Wins</span>
                    <div className="flex flex-col items-end">
                        <span className="text-2xl md:text-3xl font-black text-white">{teamData.post_wins.toFixed(1)}</span>
                        <div className="text-xs md:text-sm font-bold">{formatDelta(teamData.win_change, ' wins')}</div>
                    </div>
                </div>
                <div className="text-right text-[10px] text-slate-500 font-mono tracking-widest mt-[-10px]">90% CI: {formatCI(teamData.win_ci)}</div>

                <div className="flex justify-between items-end pb-4 border-b border-slate-700/50">
                    <span className="text-slate-400 font-bold uppercase text-[10px] md:text-xs">Playoff Prob.</span>
                    <div className="flex flex-col items-end">
                        <span className="text-lg md:text-xl font-bold text-white">{getPlayoffProb(teamData.post_wins)}%</span>
                        <div className="text-xs md:text-sm font-bold">{formatDelta(teamData.playoff_change, '%')}</div>
                    </div>
                </div>

                <div className="flex justify-between items-center text-xs md:text-sm mt-2">
                    <span className="text-slate-400">Injury Risk</span>
                    {formatDelta(teamData.injury_risk_change, '%', true)}
                </div>
                <div className="flex justify-between items-center text-xs md:text-sm">
                    <span className="text-slate-400">Roster Health</span>
                    {formatDelta(teamData.health_change, ' pts')}
                </div>
                <div className="flex justify-between items-center text-xs md:text-sm">
                    <span className="text-slate-400">Medical Index</span>
                    {formatDelta(teamData.medical_change, ' pts')}
                </div>
            </div>
            <div className="p-3 bg-slate-950/40 border-t border-slate-800 text-center">
                <span className={`text-sm md:text-lg tracking-widest font-black ${gradeColor(teamData.grade)}`}>{teamData.grade}</span>
            </div>
        </div>
    );

    return (
        <motion.div ref={containerRef} className="flex flex-col gap-6 w-full max-w-5xl mx-auto mt-4 md:mt-8 relative z-10" layout>

            {/* Core Outcome Row */}
            <div className="flex flex-col lg:flex-row items-stretch justify-center gap-4 md:gap-6">
                {renderOutcomeCard(data.team_a, "A")}

                {/* Fairness Middle Column */}
                <div className="flex flex-col items-center justify-center min-w-[180px] p-6 bg-slate-900 border border-slate-700 rounded-xl shadow-lg order-first lg:order-none">
                    <span className="text-emerald-500/80 mb-2">⚖️</span>
                    <div className="text-[10px] md:text-xs text-slate-500 font-bold uppercase tracking-widest mb-1">FAIRNESS</div>
                    <div className="text-3xl md:text-4xl font-black text-white">{data.assessment.fairness}%</div>
                    <div className="text-xs md:text-sm font-bold text-emerald-400 tracking-widest mt-1 mb-4 md:mb-6">{fairnessLabel}</div>

                    <div className="w-full h-px bg-slate-700 mb-4 md:mb-6 relative">
                        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-orange-500/50 to-transparent"></div>
                    </div>

                    <div className="text-[10px] md:text-xs text-slate-500 font-bold uppercase tracking-widest mb-1">TEAM A SCORE</div>
                    <div className={`text-2xl md:text-3xl font-black ${scoreColorClass}`}>{score}</div>
                    <div className={`text-[10px] md:text-xs font-bold uppercase tracking-widest mt-1 text-center ${gradeColor(data.team_a.grade)}`}>{data.team_a.grade}</div>
                </div>

                {renderOutcomeCard(data.team_b, "B")}
            </div>

            {/* Sim Info Banner */}
            <div className="bg-[#1e293b] border border-slate-700 p-4 rounded-lg flex flex-col sm:flex-row items-center justify-center gap-3 shadow-inner text-center sm:text-left">
                <Cpu className="w-5 h-5 text-rose-400 shrink-0" />
                <span className="text-xs md:text-sm text-slate-300">Analysis based on <strong className="text-white">1,000 Monte Carlo iterations</strong> sampling injury risk and stat variance.</span>
            </div>

            {/* Traded Players Summary */}
            <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 md:p-6 shadow-lg flex flex-col md:flex-row justify-between items-center gap-6 md:gap-8">
                <div className="flex-1 w-full">
                    <h4 className="text-[10px] font-bold tracking-widest uppercase text-slate-500 mb-3 border-b border-slate-800 pb-2">SENT BY <span className="text-orange-400 ml-1">{data.team_a.code}</span></h4>
                    <div className="flex flex-col gap-2">
                        {data.traded_players.from_a.map(p => (
                            <div key={p.player_name} className="flex justify-between items-center text-xs md:text-sm p-2 bg-slate-800/40 rounded">
                                <span className="font-bold text-slate-200 truncate pr-2">{p.player_name}</span>
                                <span className="text-[10px] md:text-xs text-slate-400 whitespace-nowrap">{p.points_per_game.toFixed(1)} PPG · {p.medical_grade}</span>
                            </div>
                        ))}
                    </div>
                </div>
                <ArrowRightLeft className="w-6 h-6 md:w-8 md:h-8 text-slate-600 rotate-90 md:rotate-0" />
                <div className="flex-1 w-full">
                    <h4 className="text-[10px] font-bold tracking-widest uppercase text-slate-500 mb-3 border-b border-slate-800 pb-2">SENT BY <span className="text-emerald-400 ml-1">{data.team_b.code}</span></h4>
                    <div className="flex flex-col gap-2">
                        {data.traded_players.from_b.map(p => (
                            <div key={p.player_name} className="flex justify-between items-center text-xs md:text-sm p-2 bg-slate-800/40 rounded">
                                <span className="font-bold text-slate-200 truncate pr-2">{p.player_name}</span>
                                <span className="text-[10px] md:text-xs text-slate-400 whitespace-nowrap">{p.points_per_game.toFixed(1)} PPG · {p.medical_grade}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Score & Meter Row */}
            <div className="flex flex-col md:flex-row gap-6">
                <div className="bg-slate-900 border border-slate-700 rounded-xl p-8 flex flex-col items-center justify-center min-w-[250px] shadow-lg">
                    <div className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4 text-center">Trade Rating for {data.team_a.code}</div>
                    <div className="flex items-baseline gap-2">
                        <span className={`text-6xl font-black ${scoreColorClass}`}>{score}</span>
                        <span className="text-lg font-bold text-slate-500">/ 100</span>
                    </div>
                </div>
                <div className="bg-slate-900 border border-slate-700 rounded-xl p-8 flex-1 flex flex-col justify-center shadow-lg">
                    <div className="flex items-center gap-2 mb-4">
                        <Activity className="w-5 h-5 text-orange-400" />
                        <h3 className="text-sm font-bold tracking-widest text-slate-300 uppercase">Why this score?</h3>
                    </div>
                    <p className="text-slate-300 leading-relaxed text-sm">
                        From the perspective of <strong className="text-white">{data.team_a.code}</strong>, this trade is graded as <strong className={gradeColor(data.team_a.grade)}>{data.team_a.grade}</strong>.
                        {data.team_a.win_change > 0 && data.team_b.win_change > 0 ? ' Both teams improve their projected ceilings. ' : ' '}
                        Winner: <strong className="text-orange-400">{data.assessment.winner}</strong> by {data.assessment.win_margin} wins.
                    </p>

                    {/* ML Interpretability & Penalty Breakdowns */}
                    <div className="mt-4 pt-4 border-t border-slate-700/50 space-y-3">
                        {data.assessment.explanations?.fit_reasons_a?.length > 0 && (
                            <div className="text-xs text-amber-300">
                                ⚠️ <strong className="text-white">{data.team_a.code} Fit Penalty Triggered:</strong>
                                <ul className="list-disc pl-5 mt-1 space-y-1">
                                    {data.assessment.explanations.fit_reasons_a.map((r, i) => <li key={i}>{r}</li>)}
                                </ul>
                            </div>
                        )}
                        {data.assessment.explanations?.fit_reasons_b?.length > 0 && (
                            <div className="text-xs text-amber-300">
                                ⚠️ <strong className="text-white">{data.team_b.code} Fit Penalty Triggered:</strong>
                                <ul className="list-disc pl-5 mt-1 space-y-1">
                                    {data.assessment.explanations.fit_reasons_b.map((r, i) => <li key={i}>{r}</li>)}
                                </ul>
                            </div>
                        )}

                        {data.assessment.explanations?.shap_incoming_a && (
                            <div className="text-xs text-slate-400">
                                <span className="font-bold text-orange-300 text-[11px] uppercase tracking-wider">{data.team_a.code} acquires {data.assessment.explanations.shap_incoming_a.name}:</span>
                                <ul className="list-disc pl-4 mt-1 space-y-1">
                                    {data.assessment.explanations.shap_incoming_a.reasons.map((r, i) => (
                                        <li key={i}>{r}</li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        {data.assessment.explanations?.shap_incoming_b && (
                            <div className="text-xs text-slate-400">
                                <span className="font-bold text-emerald-400 text-[11px] uppercase tracking-wider">{data.team_b.code} acquires {data.assessment.explanations.shap_incoming_b.name}:</span>
                                <ul className="list-disc pl-4 mt-1 space-y-1">
                                    {data.assessment.explanations.shap_incoming_b.reasons.map((r, i) => (
                                        <li key={i}>{r}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                </div>
            </div>

            {/* ML Explanation Card */}
            <div className="bg-[#0f172a] border border-orange-900/40 shadow-[0_0_30px_rgba(249,115,22,0.05)] rounded-xl p-8 relative overflow-hidden">
                <div className="absolute top-0 right-0 w-64 h-64 bg-orange-500/5 rounded-full blur-3xl mix-blend-screen pointer-events-none"></div>
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-orange-500/20 rounded-lg">
                        <span className="text-xl">🤖</span>
                    </div>
                    <h3 className="text-lg font-bold text-white tracking-wide">Logic-Based Trade Factors</h3>
                </div>
                <div className="space-y-4 relative z-10 text-slate-300 leading-relaxed text-sm">
                    <p className="text-orange-200 text-base font-medium">The simulation projects how this swap affects depth, top-heaviness, and availability.</p>
                    {Math.abs(data.team_a.medical_change) > 2 && (
                        <p>The simulation identifies a shift in long-term durability profiles between the involved rosters.</p>
                    )}
                    <p>Monte Carlo analysis confirms <strong className="text-white">{data.team_a.code}</strong> has a 90% chance to finish within {formatCI(data.team_a.win_ci)} wins of their baseline.</p>
                    <p>Post-trade roster health for <strong className="text-white">{data.team_a.code}</strong> is graded as <strong className="text-white">{data.team_a.health_grade_post}</strong>.</p>
                </div>
            </div>

        </motion.div>
    );
};

export default ResultsDashboard;
