import { motion } from 'framer-motion';
import { ReactNode } from 'react';
import { useLocation } from 'react-router-dom';

type TransitionWrapperProps = {
  children: ReactNode;
  direction?: 'up' | 'down';
};

export const TransitionWrapper = ({
  children,
  direction = 'up'
}: TransitionWrapperProps) => {
  const location = useLocation();

  return (
    <motion.div
      key={location.pathname}
      initial={{ y: direction === 'up' ? '100%' : '-100%' }}
      animate={{ y: 0 }}
      exit={{ y: direction === 'up' ? '-100%' : '100%' }}
      transition={{ 
        duration: 1.5,
        ease: [0.22, 1, 0.36, 1]
      }}
      style={{
        position: 'relative',
        width: '100%',
        height: '100%',
        overflow: 'hidden',
        backgroundColor: 'inherit'
      }}
    >
      {children}
    </motion.div>
  );
};
