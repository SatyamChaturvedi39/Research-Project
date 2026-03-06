import React from 'react';
import { Activity } from 'lucide-react';

export default function Header() {
    return (
        <header className="w-full bg-slate-900/70 backdrop-blur-md border-b border-slate-700/50 px-6 py-4 sticky top-0 z-50">
            <div className="max-w-7xl mx-auto flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="bg-gradient-to-br from-cyan-500 to-blue-700 p-2 rounded-lg shadow-lg shadow-cyan-900/40">
                        <Activity className="w-5 h-5 text-white" />
                    </div>
                    <h1 className="text-lg font-bold tracking-tight text-white">
                        NBA Trade Analyzer
                    </h1>
                </div>
                <span className="text-xs font-medium text-slate-500 hidden sm:block">
                    2025–26 Season · Trade Predictions
                </span>
            </div>
        </header>
    );
}
