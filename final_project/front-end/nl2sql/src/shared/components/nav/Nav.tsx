import { useState } from 'react';
import './Nav.css';

interface MenuItem {
  id: string;
  label: string;
  onClick: () => void;
}

interface NavProps {
  items: MenuItem[];
  spacing?: number;
  radius?: number;
}

export const Nav: React.FC<NavProps> = ({
  items,
  spacing = 10,
  radius = 120,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const totalItems = items.length;
  const totalSpacing = (totalItems - 1) * spacing;
  const availableAngle = 180 - totalSpacing;
  const anglePerItem = availableAngle / totalItems;

  return (
    <div className="menu-container">
      {/* <button
        className={`center-button ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? '−' : '+'}
      </button> */}
      <button
        className={`center-button ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label={isOpen ? 'Close menu' : 'Open menu'}
      >
        <span className="button-icon"></span>
      </button>
      {items.map((item, index) => {
        const startAngle = -90;
        const angle = startAngle + 
          (anglePerItem * index) + 
          (spacing * index) +
          anglePerItem / 2;
        return (<button
          key={item.id}
          className={`menu-item ${isOpen ? 'visible' : ''}`}
          style={{
            transform: `rotate(${angle}deg) translate(${radius}px) rotate(${-angle}deg)`,
            transitionDelay: `${index * 50}ms`
          }}
          onClick={item.onClick}
        >
          {item.label}
        </button>
        );
      })}
      {/* {items.map((item, index) => {
        const startAngle = -90;
        const angle = startAngle + 
          (anglePerItem * index) + 
          (spacing * index) +
          anglePerItem / 2;

        return (
          <button
            key={item.id}
            className={`menu-item ${isOpen ? 'visible' : ''}`}
            style={{
              transform: `rotate(${angle}deg) translate(${radius}px) rotate(${-angle}deg)`,
              transitionDelay: `${index * 50}ms`,
            }}
            onClick={item.onClick}
          >
            {item.label}
          </button>
        );
      })} */}
    </div>
  );
};
export default Nav;
