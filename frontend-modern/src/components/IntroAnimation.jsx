import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const IntroAnimation = ({ onComplete }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    // Total animation time: basketball entrance (0.8s) + gate delay (1.2s) + gate open (1s) = ~2.5s
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(onComplete, 500); // Allow exit animation to finish before removing from DOM
    }, 2500);
    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div 
          className="fixed inset-0 z-[100] flex items-center justify-center overflow-hidden"
          exit={{ opacity: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Left Gate */}
          <motion.div
            initial={{ x: 0 }}
            animate={{ x: '-100%' }}
            transition={{ duration: 1.2, delay: 1.2, ease: [0.76, 0, 0.24, 1] }} // Cinematic acceleration
            className="absolute left-0 top-0 bottom-0 w-1/2 bg-slate-950 border-r border-orange-500/30 shadow-[10px_0_30px_rgba(249,115,22,0.2)] z-20 flex justify-end overflow-hidden"
          >
            {/* Subtle tech grid pattern */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
            {/* Center line glow */}
            <div className="absolute right-0 top-0 bottom-0 w-[2px] bg-gradient-to-b from-transparent via-orange-500 to-transparent opacity-50"></div>
          </motion.div>
          
          {/* Right Gate */}
          <motion.div
            initial={{ x: 0 }}
            animate={{ x: '100%' }}
            transition={{ duration: 1.2, delay: 1.2, ease: [0.76, 0, 0.24, 1] }}
            className="absolute right-0 top-0 bottom-0 w-1/2 bg-slate-950 border-l border-orange-500/30 shadow-[-10px_0_30px_rgba(249,115,22,0.2)] z-20 flex justify-start overflow-hidden"
          >
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-[size:24px_24px]"></div>
            <div className="absolute left-0 top-0 bottom-0 w-[2px] bg-gradient-to-b from-transparent via-orange-500 to-transparent opacity-50"></div>
          </motion.div>

          {/* Central Basketball & Text */}
          <motion.div
            initial={{ scale: 0, rotate: -180, opacity: 0 }}
            animate={{ scale: 1, rotate: 0, opacity: 1 }}
            transition={{ 
              scale: { duration: 0.8, type: "spring", bounce: 0.5 },
              rotate: { duration: 0.8, ease: "easeOut" },
              opacity: { duration: 0.3 }
            }}
            className="relative z-30 flex flex-col items-center gap-6"
          >
            <motion.div
                animate={{ 
                    opacity: [1, 1, 0],
                    scale: [1, 1, 1.4] // Expands and vanishes as gates open
                }}
                transition={{
                    duration: 2.2,
                    times: [0, 0.6, 1], 
                    ease: "easeInOut"
                }}
                className="flex flex-col items-center"
            >
                <div className="relative">
                  {/* Outer pulsating glow */}
                  <div className="absolute inset-0 bg-orange-500 blur-3xl opacity-30 rounded-full animate-pulse"></div>
                  
                  {/* Basketball SVG Element */}
                  <svg 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="1.2"
                    className="w-32 h-32 text-orange-400 relative z-10 drop-shadow-[0_0_15px_rgba(249,115,22,0.6)]"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <path d="M5.4 5.4c1.8 1.8 2.6 4.3 2.6 6.6s-.8 4.8-2.6 6.6" />
                    <path d="M18.6 5.4c-1.8 1.8-2.6 4.3-2.6 6.6s.8 4.8 2.6 6.6" />
                    <path d="M12 2v20" />
                    <path d="M2 12h20" />
                  </svg>
                </div>
                
                <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5, duration: 0.6 }}
                    className="mt-6 text-2xl md:text-4xl font-black tracking-[0.25em] text-transparent bg-clip-text bg-gradient-to-r from-orange-300 via-rose-400 to-orange-300 drop-shadow-sm"
                >
                TRADE ANALYZER
                </motion.div>
            </motion.div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default IntroAnimation;
