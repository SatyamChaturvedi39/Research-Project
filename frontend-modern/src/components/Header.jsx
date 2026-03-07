import React from 'react';
import { Activity } from 'lucide-react';

const Header = () => {
    return (
        <header className="glass-panel text-white p-6 mb-8 flex items-center justify-between animate-float">
            <div className="flex items-center justify-center gap-4 mb-8">
                <div className="bg-orange-500/20 p-3 rounded-lg border border-orange-500/50">
                    <Activity className="w-8 h-8 text-orange-400" />
                </div>
                <div>
                    <h1 className="text-3xl font-black tracking-tight bg-gradient-to-r from-orange-400 to-rose-400 bg-clip-text text-transparent">
                        NBA Trade Analyzer
                    </h1>
                    <p className="text-slate-400 text-sm mt-1 font-medium tracking-wide">
                        Enterprise Simulation & Predictive Analytics
                    </p>
                </div>
            </div>
            <div className="flex items-center gap-2">
                <div className="flex items-center gap-2 px-4 py-2 bg-slate-800/50 rounded-full border border-slate-700">
                    <div className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)]"></div>
                    <span className="text-xs font-semibold text-slate-300">Models Online</span>
                </div>
            </div>
        </header>
    );
};

export default Header;
