import React from 'react';
import './Header.css';

interface HeaderProps {
    isDarkMode: boolean;
    toggleDarkMode: () => void;
}

function Header({ isDarkMode, toggleDarkMode }: HeaderProps) {
    return (
        <div className='header'>
            <div style={{ flex: 1, textAlign: 'center', color: 'white' }}>
                <h3>NLP2SQL</h3>
            </div>
            <div style={{ position: 'absolute', right: '20px', top: '15px' }}>
                <label className="switch">
                    <input type="checkbox" checked={isDarkMode} onChange={toggleDarkMode} />
                    <span className="slider round"></span>
                </label>
            </div>
        </div>
    );
}

export default Header;
