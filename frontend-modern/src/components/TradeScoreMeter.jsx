import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

function generateExplanation(score, stats) {
    const sentences = [];
    const d = stats;

    if (score >= 71) {
        sentences.push(`The model rates this as a great trade. The incoming players project solidly across scoring, rebounding, and playmaking — a clear overall improvement.`);
    } else if (score >= 51) {
        sentences.push(`The model rates this as a good trade with some caveats. There are real upsides here, but the move isn't a slam dunk across every category.`);
    } else if (score >= 31) {
        sentences.push(`This trade carries notable risk. The model's projections suggest the incoming package doesn't clearly improve the team's overall output.`);
    } else {
        sentences.push(`This trade looks highly unfavorable. Based on historical data patterns, the outgoing talent significantly outweighs the projected return.`);
    }

    if (d.ppgDelta > 3) {
        sentences.push(`On the offensive end, you are acquiring a major scoring upgrade. The incoming players are projected to easily surpass the offensive production you're giving up.`);
    } else if (d.ppgDelta > 0) {
        sentences.push(`Scoring-wise, you gain a slight edge. The model anticipates a marginal but positive increase in overall points per game.`);
    } else if (d.ppgDelta > -3) {
        sentences.push(`The scoring impact is nearly neutral. Essentially, the team is swapping like for like offensively without losing much ground.`);
    } else {
        sentences.push(`Scoring takes a noticeably hard hit in this scenario. You're trading away primary offensive contributors without getting enough firepower in return.`);
    }

    if (d.outRPG > 0 || d.inRPG > 0) {
        if (d.rpgDelta > 1.5) {
            sentences.push(`This move also bolsters the team's rebounding depth, translating to an expected improvement in second-chance opportunities and defensive glass coverage.`);
        } else if (d.rpgDelta < -1.5) {
            sentences.push(`Rebounding presents a real concern here, as losing size and hustle on the boards could negatively swing possession metrics.`);
        } else {
            sentences.push(`Rebounding volume stays relatively stable, meaning the team's rebounding identity won't be drastically altered.`);
        }
    }

    if (d.apgDelta > 1.5) {
        sentences.push(`Playmaking sees a distinct improvement, signaling better projected ball distribution and team offensive flow.`);
    } else if (d.apgDelta < -1.5) {
        sentences.push(`Playmaking takes a step back. Losing this volume of assists could slow ball movement and stagnate half-court sets.`);
    } else {
        sentences.push(`Assist production holds relatively steady in this deal, so ball movement and half-court offense shouldn't look very different.`);
    }

    if (d.ageDelta > 2) {
        sentences.push(`There is a real youth dividend accompanying this deal. Acquiring younger talent provides a longer contention window and higher development upside.`);
    } else if (d.ageDelta < -2) {
        sentences.push(`Age is a significant risk factor here. Taking on an older average roster increases the likelihood of performance decline and injury issues down the stretch.`);
    } else {
        sentences.push(`The age balance remains competitive, so the team isn't taking on extra longevity or developmental risk.`);
    }

    const avgConf = (d.outConf + d.inConf) / 2;
    if (avgConf >= 0.8) {
        sentences.push(`Finally, the XGBoost algorithms exhibit high confidence in these projections, as the involved players have robust, predictable multi-year data profiles.`);
    } else if (avgConf >= 0.5) {
        sentences.push(`The model expresses moderate confidence in this analysis. Some players have shorter track records, introducing a wider margin of uncertainty into the forecast.`);
    } else {
        sentences.push(`Please note the model's confidence is relatively low here. Limited historical data for key players means you should treat these projections with extra caution.`);
    }

    return sentences;
}

export default function TradeScoreMeter({ targetTeam, incomingResults, outgoingResults }) {
    const [score, setScore] = useState(0);
    const [animatedScore, setAnimatedScore] = useState(0);

    const sumAttr = (results, attr) => results.reduce((sum, r) => sum + (r.predictions?.[attr] || 0), 0);
    const sumAge = (results) => results.reduce((sum, r) => sum + (r.current_age || 25), 0);

    const confWeight = (results) => {
        if (results.length === 0) return 0.6;
        const confMap = { high: 1.0, medium: 0.6, low: 0.3 };
        const total = results.reduce((sum, r) => {
            const label = r.confidence_ranges?.ppg?.confidence_label || 'medium';
            return sum + (confMap[label] || 0.6);
        }, 0);
        return total / results.length;
    };

    const outPPG = sumAttr(outgoingResults, 'ppg');
    const inPPG = sumAttr(incomingResults, 'ppg');
    const outRPG = sumAttr(outgoingResults, 'rpg');
    const inRPG = sumAttr(incomingResults, 'rpg');
    const outAPG = sumAttr(outgoingResults, 'apg');
    const inAPG = sumAttr(incomingResults, 'apg');

    const outAvgAge = outgoingResults.length ? sumAge(outgoingResults) / outgoingResults.length : 25;
    const inAvgAge = incomingResults.length ? sumAge(incomingResults) / incomingResults.length : 25;

    const outConf = confWeight(outgoingResults);
    const inConf = confWeight(incomingResults);

    const ppgDelta = inPPG - outPPG;
    const rpgDelta = inRPG - outRPG;
    const apgDelta = inAPG - outAPG;
    const ageDelta = outAvgAge - inAvgAge;

    const stats = { ppgDelta, outPPG, inPPG, rpgDelta, outRPG, inRPG, apgDelta, outAPG, inAPG, ageDelta, outAvgAge, inAvgAge, outConf, inConf };

    useEffect(() => {
        // Core performance metrics (out of 50 total)
        // Neutral delta = 12.5 points each (12.5 * 4 = 50 total base score)
        const ppgScore = Math.min(25, Math.max(0, 12.5 + (ppgDelta / Math.max(outPPG, 1)) * 25));
        const rpgScore = Math.min(12.5, Math.max(0, 6.25 + (rpgDelta / Math.max(outRPG, 1)) * 12.5));
        const apgScore = Math.min(12.5, Math.max(0, 6.25 + (apgDelta / Math.max(outAPG, 1)) * 12.5));

        // Age is a smaller factor, max 12.5 points
        const ageScore = Math.min(12.5, Math.max(0, 6.25 + (ageDelta / 5) * 12.5));

        // Add up components which naturally center around exactly 37.5 + 12.5/2 = 43.75 ?
        // Base neutral points: PPG(12.5) + RPG(6.25) + APG(6.25) + Age(6.25) = 31.25. 
        // We want a neutral trade to be exactly 50.
        // Let's rescale the base points to sum to 50 when purely neutral:
        // PPG (max 40, neutral 20), RPG (max 20, neutral 10), APG (max 20, neutral 10), Age (max 20, neutral 10). Total = 100 max, 50 neutral.
        const ppgCore = Math.min(40, Math.max(0, 20 + (ppgDelta / Math.max(outPPG, 1)) * 40));
        const rpgCore = Math.min(20, Math.max(0, 10 + (rpgDelta / Math.max(outRPG, 1)) * 20));
        const apgCore = Math.min(20, Math.max(0, 10 + (apgDelta / Math.max(outAPG, 1)) * 20));
        const ageCore = Math.min(20, Math.max(0, 10 + (ageDelta / 5) * 20));

        let rawScore = ppgCore + rpgCore + apgCore + ageCore;

        // Apply confidence as a penalty to extreme scores if confidence is low, 
        // dragging them closer to 50 to represent uncertainty.
        const confMod = (outConf + inConf) / 2; // e.g. 0.6 to 1.0
        const distanceToNeutral = rawScore - 50;

        // If confidence is 1.0, keep distance identical. If confidence is 0.5, halve the distance to neutral.
        const finalScore = Math.round(Math.max(0, Math.min(100, 50 + (distanceToNeutral * confMod))));

        setScore(finalScore);

        let current = 0;
        const step = finalScore / 60;
        const counter = setInterval(() => {
            current = Math.min(current + step, finalScore);
            setAnimatedScore(Math.round(current));
            if (current >= finalScore) clearInterval(counter);
        }, 16);

        return () => clearInterval(counter);
    }, [outPPG, inPPG, outRPG, inRPG, outAPG, inAPG, outAvgAge, inAvgAge, outConf, inConf, ppgDelta, rpgDelta, apgDelta, ageDelta]);

    let color = 'text-red-400';
    let bgMeter = 'bg-red-500';
    if (animatedScore >= 71) { color = 'text-emerald-400'; bgMeter = 'bg-emerald-500'; }
    else if (animatedScore >= 51) { color = 'text-yellow-400'; bgMeter = 'bg-yellow-500'; }
    else if (animatedScore >= 31) { color = 'text-orange-400'; bgMeter = 'bg-orange-500'; }

    const sentences = generateExplanation(score, stats);

    return (
        <div className="w-full glass-panel p-6 mt-6 mb-8 border border-slate-700/50 shadow-2xl relative overflow-hidden">
            {/* Soft background glow based on score class */}
            <div className={`absolute top-0 right-0 w-64 h-64 blur-3xl opacity-10 pointer-events-none rounded-full ${bgMeter}`}></div>

            <h3 className="text-xl font-black text-center text-white mb-6 tracking-tight uppercase">
                Trade Score for {targetTeam}
            </h3>

            <div className="flex flex-col md:flex-row items-center gap-8">
                {/* Score Circle */}
                <div className="relative w-32 h-32 flex-shrink-0 flex items-center justify-center rounded-full bg-slate-900 border-4 border-slate-800 shadow-[inset_0_2px_10px_rgba(0,0,0,0.5)]">
                    <svg className="absolute inset-0 w-full h-full -rotate-90">
                        <circle cx="64" cy="64" r="58" className="stroke-slate-800" strokeWidth="8" fill="none" />
                        <motion.circle
                            cx="64"
                            cy="64"
                            r="58"
                            strokeWidth="8"
                            fill="none"
                            strokeLinecap="round"
                            strokeDasharray={364}
                            initial={{ strokeDashoffset: 364 }}
                            animate={{ strokeDashoffset: 364 - (364 * animatedScore) / 100 }}
                            transition={{ duration: 1, ease: "easeOut" }}
                            className={`${color.replace('text-', 'stroke-')}`}
                        />
                    </svg>
                    <div className="text-center">
                        <span className={`text-4xl font-black ${color}`}>{animatedScore}</span>
                        <span className="text-xs text-slate-500 block uppercase font-bold tracking-widest mt-1">/ 100</span>
                    </div>
                </div>

                {/* Explanation Text */}
                <div className="flex-1 space-y-3 relative z-10">
                    {sentences.map((s, i) => (
                        <p key={i} className={`text-sm leading-relaxed ${i === 0 ? 'text-white font-semibold' : 'text-slate-400'}`}>
                            {s}
                        </p>
                    ))}
                </div>
            </div>
        </div>
    );
}
