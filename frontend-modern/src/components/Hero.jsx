import React from 'react';
import { motion } from 'framer-motion';

const Hero = () => {
    return (
        <section className="h-screen flex items-center px-8 relative">
            <div className="max-w-4xl">
                <motion.div
                    initial={{ opacity: 0, x: -30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.8 }}
                    className="space-y-6"
                >
                    <div className="text-orange-500 font-black tracking-[0.3em] uppercase text-sm">
                        ML-Powered Analytics
                    </div>
                    <h1 className="text-6xl md:text-8xl font-black leading-[0.95] text-white">
                        Predict Trade<br />Performance
                    </h1>
                    <p className="text-slate-400 text-lg md:text-xl max-w-xl leading-relaxed">
                        Multi-output XGBoost models trained on 54 temporal features from 5 seasons of NBA data,
                        predicting PPG, RPG, APG, MPG, and Shooting Efficiency — validated on a held-out test set.
                    </p>
                </motion.div>
            </div>

            <div className="absolute right-12 bottom-12 flex flex-col items-center gap-4">
                <div className="[writing-mode:vertical-rl] text-[10px] font-bold tracking-[0.3em] text-slate-500 uppercase">
                    Scroll to Explore
                </div>
                <div className="w-2 h-2 rounded-full bg-orange-500 animate-bounce"></div>
            </div>
        </section>
    );
};

export default Hero;
