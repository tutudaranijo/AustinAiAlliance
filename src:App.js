// src/App.js
import React, { useState, useEffect, useRef } from 'react';
import ChatBubble from './ChatBubble';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Load chat history from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem('chatHistory');
    if (saved) {
      setMessages(JSON.parse(saved));
    }
  }, []);

  // Save chat history to localStorage on messages update
  useEffect(() => {
    localStorage.setItem('chatHistory', JSON.stringify(messages));
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userInput.trim()) return;

    // Append user message
    const newMessages = [...messages, { role: 'user', content: userInput }];
    setMessages(newMessages);
    const query = userInput;
    setUserInput('');

    // Append a placeholder assistant message
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      if (!response.ok) throw new Error("Network error");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let content = '';
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunk = decoder.decode(value);
        content += chunk;
        // Update last assistant message with streamed tokens
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1].content = content;
          return updated;
        });
      }
    } catch (error) {
      console.error(error);
      setMessages(prev => [...prev, { role: 'assistant', content: "Error: Unable to fetch response" }]);
    }
    setIsLoading(false);
  };

  return (
    <div className="chat-container">
      <header className="chat-header">Austin AI Alliance Chatbot</header>
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <ChatBubble key={index} role={msg.role} content={msg.content} />
        ))}
        {isLoading && <ChatBubble role="assistant" content="Typing..." isTyping={true} />}
        <div ref={messagesEndRef} />
      </div>
      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Ask me anything..."
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>Send</button>
      </form>
    </div>
  );
}

export default App;
