import React from 'react';
import { motion } from 'framer-motion';

const SDG_GOALS = [
    {
        number: '3',
        title: 'Good Health & Well-Being',
        color: 'from-green-500 to-emerald-600',
        borderColor: 'border-green-500',
        textColor: 'text-green-400',
        icon: '🏥',
        relevance: 'Our injury risk model predicts which players are likely to suffer significant injuries next season, helping teams proactively manage athlete health, reduce overuse injuries, and support long-term player well-being through data-informed roster decisions.',
    },
    {
        number: '8',
        title: 'Decent Work & Economic Growth',
        color: 'from-rose-600 to-rose-700',
        borderColor: 'border-rose-500',
        textColor: 'text-rose-400',
        icon: '💼',
        relevance: 'Fair and evidence-based trade analysis supports equitable player valuations in contract negotiations and trades. By removing subjective bias from talent assessment, this helps ensure players are compensated fairly for their projected contributions.',
    },
    {
        number: '9',
        title: 'Industry, Innovation & Infrastructure',
        color: 'from-orange-500 to-amber-600',
        borderColor: 'border-orange-500',
        textColor: 'text-orange-400',
        icon: '🏭',
        relevance: 'This project demonstrates the practical application of interpretable machine learning to sports analytics — a fast-growing industry. By combining XGBoost, SHAP explainability, and Monte Carlo simulation, we advance the infrastructure of AI-driven decision support tools.',
    },
    {
        number: '10',
        title: 'Reduced Inequalities',
        color: 'from-pink-600 to-rose-600',
        borderColor: 'border-pink-500',
        textColor: 'text-pink-400',
        icon: '⚖️',
        relevance: 'Smaller-market NBA teams with limited scouting resources can use this tool to access the same quality of data-driven analysis previously available only to large-budget franchises, levelling the playing field in the league.',
    },
    {
        number: '17',
        title: 'Partnerships for the Goals',
        color: 'from-blue-600 to-indigo-600',
        borderColor: 'border-blue-500',
        textColor: 'text-blue-400',
        icon: '🤝',
        relevance: 'Built as a research project that bridges academia and sports industry, this work demonstrates how open-source tools (Python, XGBoost, SHAP, Flask, React) can foster collaborative innovation, aligning with the SDG spirit of knowledge-sharing for sustainable development.',
    },
];

const About = () => {
    return (
        <div className="text-white space-y-16 pb-20">
            {/* Hero */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-center space-y-4"
            >
                <div className="text-orange-400 font-bold tracking-[0.2em] uppercase text-sm">The Project</div>
                <h1 className="text-5xl md:text-7xl font-black">About the Analyzer</h1>
                <p className="text-slate-400 max-w-2xl mx-auto text-lg text-balance italic">
                    Bridging the gap between raw basketball data and front-office decision making.
                </p>
            </motion.div>

            {/* Grid */}
            <div className="grid grid-cols-1 md:grid-cols-[1fr_300px] gap-12">
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="space-y-8"
                >
                    <div className="space-y-4">
                        <h2 className="text-3xl font-black text-orange-400 uppercase tracking-tighter">Vision</h2>
                        <p className="text-xl text-slate-300 leading-relaxed font-medium">
                            The NBA Trade Analyzer was developed as a research prototype to demonstrate the power of <strong>Interpretable Machine Learning</strong> in professional sports. While most models provide a "black box" prediction, our goal was to build a tool that explains its reasoning in plain English, helping fans and analysts understand the underlying drivers of player performance.
                        </p>
                    </div>

                    <div className="space-y-4">
                        <h2 className="text-3xl font-black text-orange-400 uppercase tracking-tighter">Methodology</h2>
                        <p className="text-xl text-slate-300 leading-relaxed font-medium">
                            By combining traditional box-score statistics with advanced temporal features and injury risk modeling, we provide a holistic view of a player's value. The system doesn't just look at what a player did last week; it looks at their 5-year trajectory, their physical durability, and how their role has evolved over their career.
                        </p>
                    </div>
                </motion.div>

                <motion.aside
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 }}
                    className="space-y-6"
                >
                    <div className="glass-panel p-8 bg-slate-900/40">
                        <h4 className="text-lg font-black uppercase mb-6 pb-2 border-b border-slate-700">Technical Stack</h4>
                        <ul className="space-y-4">
                            {[
                                { k: 'Backend', v: 'Flask (Python)' },
                                { k: 'ML', v: 'XGBoost, SHAP, Monte Carlo Simulation' },
                                { k: 'Frontend', v: 'React, Tailwind v4' },
                                { k: 'Graphics', v: 'Framer Motion' }
                            ].map((item, i) => (
                                <li key={i} className="flex flex-col">
                                    <span className="text-[10px] uppercase text-slate-500 font-black">{item.k}</span>
                                    <span className="text-sm font-semibold">{item.v}</span>
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div className="glass-panel p-8 bg-orange-500/5 border-orange-500/10">
                        <h4 className="text-lg font-black uppercase mb-4">Project Info</h4>
                        <p className="text-sm text-slate-400 leading-relaxed italic">
                            Research Project - Module 1<br />Player Performance Prediction<br />v2.1.0 (Production Build)
                        </p>
                    </div>
                </motion.aside>
            </div>
        </div>
    );
};

export default About;
