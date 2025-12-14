import './App.css';
import { BrowserRouter as Router } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import Header from './shared/components/header/Header';
import AppRoutes from './routes/AppRoutes';
import './App.css';

// ... imports
import React, { useState } from 'react';

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
    <div className={`App ${isDarkMode ? 'dark-mode' : ''}`}>
      <Header isDarkMode={isDarkMode} toggleDarkMode={toggleDarkMode}></Header>
      <Router>

        <AnimatePresence mode="wait">
          <AppRoutes />
        </AnimatePresence>
      </Router>

    </div>
  );
}

export default App;
