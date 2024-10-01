import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

console.log('index.js executing');

const rootElement = document.getElementById('root');
if (rootElement) {
  console.log('Root element found, creating React root');
  const root = ReactDOM.createRoot(rootElement);
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
} else {
  console.error('Root element not found');
}