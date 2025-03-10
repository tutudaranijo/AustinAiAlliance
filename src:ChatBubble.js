// src/ChatBubble.js
import React from 'react';
import './ChatBubble.css';

function ChatBubble({ role, content, isTyping }) {
  return (
    <div className={`chat-bubble ${role}`}>
      <div className="bubble-content">
        {content.split('\n').map((line, index) => (
          <p key={index}>{line}</p>
        ))}
      </div>
      {isTyping && <div className="typing-indicator">...</div>}
    </div>
  );
}

export default ChatBubble;
