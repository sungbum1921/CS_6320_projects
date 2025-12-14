import React, { useState } from 'react';
import { Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { TransitionWrapper} from '../shared/components/transitionWrapper' ;
import NotFound from '../components/not-found/NotFound';
import Login from '../components/login/Login';
import Home from '../components/home/Home';


const AppRoutes: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [direction, setDirection] = useState<'up'|'down'>('up');
  const [isTransitioning, setIsTransitioning] = useState(false);

  const handleNavigation = (newPath: string, dir: 'up'|'down') => {
    if (!isTransitioning) {
      setDirection(dir);
      setIsTransitioning(true);
      navigate(newPath);
    }
  };
  return (
    <Routes location={location} key={location.pathname}>
        <Route path="/" element={
          <TransitionWrapper direction={direction}>
            <Login onNavigate={() => handleNavigation('/home', 'up')}/>
          </TransitionWrapper>} />
        <Route path="/home" element={
          <TransitionWrapper direction={direction}>
            <Home />
          </TransitionWrapper>
          } />
        {/* <Route path="/search" element={<Search />} />
        <Route path="/history" element={<History />} /> */}
        <Route path="*" element={<NotFound />} />
    </Routes>
  );
};

export default AppRoutes;
