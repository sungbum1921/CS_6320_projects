import React, { useState } from 'react';
import './Login.css';
import { Snackbar, Alert, AlertColor } from '@mui/material';
import main_logo from '../../assets/images/main_logo.png';
import eyeHide from '../../assets/images/eye-hide.png';
import eyeShow from '../../assets/images/eye-show.png';
import emailIcon from '../../assets/images/email_img.png';
import passwordIcon from '../../assets/images/password_img.png';

interface LoginProps {
  onNavigate: () => void;
}

function Login({ onNavigate }: LoginProps) {
  const [passwordFieldType, setPasswordFieldType] = useState<'password' | 'text'>('password');
  const toggleViewPassword = () => {
    setPasswordFieldType((prevType) => (prevType === 'password' ? 'text' : 'password'));
  }
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState('');
  const [severity, setSeverity] = useState<AlertColor>('success');

  const showMessage = (newMessage: string, newSeverity: AlertColor = 'success') => {
    setMessage(newMessage);
    setSeverity(newSeverity);
    setOpen(true);
  };

  const handleClose = () => setOpen(false);


  const handleLoginClick = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.preventDefault();
    console.log('Username:', username, 'Password:', password);
    if (username.trim() === 'user@example.com' && password.trim() === 'password') {
      onNavigate();
    }
    else {
      showMessage("Incorrect username or password!!!!", "error")
    }
  };

  return (
    <div className="login-container">
      {/* <Nav items={test}></Nav> */}
      <div className="left-section">
        <h1>Your Data, <br /> Your Language.</h1>
        <p>Seamlessly transform natural langiage into powerful SQL queries.</p>
      </div>
      <div className="right-section">
        <div style={{ alignContent: 'center' }}>
          <img src={main_logo} alt="logo" width={200} height={200} />
        </div>
        <h2>Sign In</h2>
        <p style={{ color: '#666', marginBottom: '20px' }}>Access your account.</p>
        <form>
          <label>Username</label>
          <div className="input-container">
            <input
              className="email-input"
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="user@example.com"
            />
          </div>

          <label>Password</label><br></br>
          <div className="input-container">
            <input
              className="password-input"
              type={passwordFieldType}
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password" />
            <button type="button" id="toggle-password" onClick={toggleViewPassword} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>
              <img
                src={passwordFieldType === 'text' ? eyeHide : eyeShow}
                alt={passwordFieldType === 'text' ? 'View password' : 'Hide password'}
                style={{ width: '25px', height: '25px', verticalAlign: 'middle' }}
              />
            </button>
          </div>

          <div className="form-links">
            <a href="#" className="forgot-password">Forgot Password?</a>
          </div>

          <button type="button" onClick={handleLoginClick} className="login-button">Login</button>
        </form>
      </div>
      <Snackbar
        open={open}
        autoHideDuration={6000}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleClose} severity={severity} sx={{ width: '100%' }}>
          {message}
        </Alert>
      </Snackbar>
    </div>

  );
}
export default Login;
