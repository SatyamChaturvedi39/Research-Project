import React from 'react';
import { Activity } from 'lucide-react';

const Header = ({ onPageChange, currentPage }) => {
    return (
        <header className="glass-panel text-white p-6 mb-8 flex items-center justify-between">
            <div className="flex items-center gap-4">
                <div className="bg-orange-500/20 p-2 md:p-3 rounded-lg border border-orange-500/50">
                    <Activity className="w-6 h-6 md:w-8 md:h-8 text-orange-400" />
                </div>
                <div>
                    <h1
                        className="text-xl md:text-3xl font-black tracking-tight bg-gradient-to-r from-orange-400 to-rose-400 bg-clip-text text-transparent cursor-pointer"
                        onClick={() => onPageChange('home')}
                    >
                        NBA Trade Analyzer
                    </h1>
                </div>
            </div>

            <nav className="hidden md:flex items-center gap-8 mr-auto ml-12">
                {['home', 'model-info', 'about'].map((page) => (
                    <button
                        key={page}
                        onClick={() => onPageChange(page)}
                        className={`text-sm font-bold uppercase tracking-widest transition-all duration-300 relative group ${currentPage === page ? 'text-orange-400' : 'text-slate-400 hover:text-white'
                            }`}
                    >
                        {page.replace('-', ' ')}
                        <span className={`absolute -bottom-1 left-0 h-0.5 bg-orange-400 transition-all duration-300 ${currentPage === page ? 'w-full' : 'w-0 group-hover:w-full'
                            }`}></span>
                    </button>
                ))}
            </nav>

            <div className="flex items-center gap-2">
                <div className="hidden sm:flex items-center gap-2 px-3 md:px-4 py-1.5 md:py-2 bg-slate-800/50 rounded-full border border-slate-700">
                    <div className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.8)] animate-pulse"></div>
                    <span className="text-[10px] md:text-xs font-semibold text-slate-300">Models Online</span>
                </div>
            </div>
        </header>
    );
};


export default Header;
