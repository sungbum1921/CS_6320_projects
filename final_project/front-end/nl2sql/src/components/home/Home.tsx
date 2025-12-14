// Home.tsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import sendIcon from '../../assets/images/send.png';
import attachIcon from '../../assets/images/attach.png';
import './Home.css';
import { Snackbar, Alert, AlertColor, Dialog } from '@mui/material';

import { useAppDispatch, useAppSelector } from '../../shared/hooks';
import { queryRequest } from '../../shared/slices/query.slice';
import { RootState } from '../../shared/store/store';

type Message = {
  message: string;
  timestamp: string;
  type: 'user' | 'response';
};

const MessageDisplay = ({ msg }: { msg: Message }) => {
  const [displayedText, setDisplayedText] = useState('');

  const typeWriter = useCallback((text: string, i = 0) => {
    if (i < text.length) {
      setDisplayedText(text.slice(0, i + 1));
      setTimeout(() => typeWriter(text, i + 1), 40);
    }
  }, []);

  useEffect(() => {
    if (msg.type === 'response') {
      typeWriter(msg.message);
    } else {
      setDisplayedText(msg.message);
    }
  }, [msg.message, msg.type, typeWriter]);

  return (
    <div className={`message ${msg.type}`}>
      {displayedText}
      {msg.type === 'response' && displayedText.length < msg.message.length && (
        <span className="cursor">|</span>
      )}
    </div>
  );
};

function Home() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    setMessages(prev =>
      [...prev].sort((a, b) => Number(a.timestamp) - Number(b.timestamp))
    );
  }, [messages.length]);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = () => {
    if (!input.trim()) return;

    // Add user message
    const newUserMessage: Message = {
      message: input,
      timestamp: Date.now().toString(),
      type: 'user'
    };

    setMessages(prev => [...prev, newUserMessage]);

    // Simulate API response
    // setTimeout(() => {
    //   const newResponse: Message = {
    //     message: `Response to: ${input}`,
    //     timestamp: (Date.now() + 1).toString(),
    //     type: 'response'
    //   };
    //   setMessages(prev => [...prev, newResponse]);
    // }, 1000);
    handleSubmit(input);

    setInput('');
  };


  // snackbar
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState('');
  const [severity, setSeverity] = useState<AlertColor>('success');

  const showMessage = (newMessage: string, newSeverity: AlertColor = 'success') => {
    setMessage(newMessage);
    setSeverity(newSeverity);
    setOpen(true);
  };

  const handleClose = () => setOpen(false);
  const attachClicked = () => {
    showMessage('Functionality not available yet.', 'info');
  }

  //API call
  const dispatch = useAppDispatch();
  const { data, loading, error } = useAppSelector((state: RootState) => state.query);
  const [inputText, setInputText] = useState('');

  const handleSubmit = (inputText: string) => {
    dispatch(queryRequest({ query: inputText, model_id: selectedModel }));
  };
  useEffect(() => {
    if (data) {
      const newResponseMessage: Message = {
        message: data.message,
        timestamp: data.timestamp,
        type: 'response'
      };

      setMessages(prev => [...prev, newResponseMessage]);
    }
  }, [data]);

  // Model selection state
  const [selectedModel, setSelectedModel] = useState('t5_small');
  const [warningOpen, setWarningOpen] = useState(false);

  const models = [
    { id: 't5_small', label: 'T5 Small' },
    { id: 't5_base', label: 'T5 Base' },
    { id: 'codellama_7b_QLoRA', label: 'Code Llama' }
  ];

  const handleModelClick = (modelId: string) => {
    if (modelId === 'codellama_7b_QLoRA') {
      setWarningOpen(true);
    } else {
      setSelectedModel(modelId);
    }
  };

  const handleConfirmLlama = () => {
    setSelectedModel('codellama_7b_QLoRA');
    setWarningOpen(false);
  };

  const [clearConfirmOpen, setClearConfirmOpen] = useState(false);

  const handleClearClick = () => {
    setClearConfirmOpen(true);
  };

  const handleConfirmClear = () => {
    setMessages([]);
    setClearConfirmOpen(false);
  };

  const TrashIcon = () => (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M3 6H5H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M8 6V4C8 3.44772 8.44772 3 9 3H15C15.5523 3 16 3.44772 16 4V6M19 6V20C19 20.5523 18.5523 21 18 21H6C5.44772 21 5 20.5523 5 20V6H19Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M10 11V17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M14 11V17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );

  return (
    <div className="elevated-container">
      <div className="model-selector">
        {models.map((model) => (
          <button
            key={model.id}
            className={`model-btn ${selectedModel === model.id ? 'active' : ''}`}
            onClick={() => handleModelClick(model.id)}
          >
            {model.label}
          </button>
        ))}
      </div>

      <Dialog
        open={warningOpen}
        onClose={() => setWarningOpen(false)}
        PaperProps={{
          style: {
            borderRadius: '16px',
            padding: '10px',
            maxWidth: '500px'
          }
        }}
      >
        <div style={{ padding: '20px' }}>
          <h3 style={{ marginTop: 0, marginBottom: '15px' }}>⚠️ High Performance Required</h3>
          <p style={{ color: '#555', lineHeight: '1.6', marginBottom: '25px' }}>
            Code Llama requires significant GPU memory (VRAM).
            <br /><br />
            If you are running on <b>CPU only</b> or have <b>low VRAM</b>, the application may freeze or crash.
            <br /><br />
            We strongly recommend using <b>T5 Small</b> or <b>T5 Base</b> for better stability on standard hardware.
          </p>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
            <button
              onClick={() => setWarningOpen(false)}
              style={{
                padding: '10px 20px',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                background: '#e0e0e0',
                color: '#333',
                fontWeight: 'bold'
              }}
            >
              Use T5 Instead
            </button>
            <button
              onClick={handleConfirmLlama}
              style={{
                padding: '10px 20px',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                background: '#333',
                color: 'white',
                fontWeight: 'bold'
              }}
            >
              Proceed Anyway
            </button>
          </div>
        </div>
      </Dialog>
      <div className="content-wrapper">
        <div className="elevated-box scrollable-hidden" style={{ position: 'relative' }}>
          {messages.length > 0 && (
            <button className="clear-chat-btn" onClick={handleClearClick} title="Clear Chat">
              <TrashIcon />
            </button>
          )}
          <div className="chat-container scrollable-hidden">
            {messages.map((msg, index) => (
              <MessageDisplay key={index} msg={msg} />
            ))}
            <div ref={chatEndRef} />
          </div>

          <div className="input-container">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              placeholder="Type a message..."
            />
            <div className="button-wrapper">
              <button className="icon-button-1" onClick={attachClicked}>
                <img src={attachIcon} alt="Attach file" className="icon-image" />
              </button>
              <button className="icon-button" onClick={handleSend}>
                <img src={sendIcon} alt="Send message" className="icon-image" />
              </button>
            </div>
          </div>
        </div>

        <div className="memo-box elevated-box">
          <div className="memo-header">Memo</div>
          <textarea
            className="memo-textarea"
            placeholder="Type anything here... (e.g., copied SQL, notes)"
          />
        </div>
      </div>


      {/* Clear Confirmation Dialog */}
      <Dialog
        open={clearConfirmOpen}
        onClose={() => setClearConfirmOpen(false)}
        PaperProps={{
          style: {
            borderRadius: '16px',
            padding: '10px',
            maxWidth: '400px'
          }
        }}
      >
        <div style={{ padding: '20px' }}>
          <h3 style={{ marginTop: 0, marginBottom: '15px' }}>Clear Chat?</h3>
          <p style={{ color: '#555', lineHeight: '1.6', marginBottom: '25px' }}>
            Are you sure you want to delete all messages? This action cannot be undone.
          </p>
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
            <button
              onClick={() => setClearConfirmOpen(false)}
              style={{
                padding: '10px 20px',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                background: '#e0e0e0',
                color: '#333',
                fontWeight: 'bold'
              }}
            >
              Cancel
            </button>
            <button
              onClick={handleConfirmClear}
              style={{
                padding: '10px 20px',
                border: 'none',
                borderRadius: '8px',
                cursor: 'pointer',
                background: '#ff4d4d',
                color: 'white',
                fontWeight: 'bold'
              }}
            >
              Clear All
            </button>
          </div>
        </div>
      </Dialog>

      <Snackbar
        open={open}
        autoHideDuration={1500}
        onClose={handleClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleClose} severity={severity} sx={{ width: '100%' }}>
          {message}
        </Alert>
      </Snackbar>
    </div >
  );
}

export default Home;
