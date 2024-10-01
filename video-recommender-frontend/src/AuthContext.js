import React, { createContext, useState } from 'react';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [authTokens, setAuthTokens] = useState(() => localStorage.getItem('tokens'));
  console.log('AuthProvider rendering, authTokens:', authTokens);

  const setTokens = (data) => {
    localStorage.setItem('tokens', data);
    setAuthTokens(data);
  };

  return (
    <AuthContext.Provider value={{ authTokens, setAuthTokens: setTokens }}>
      {children}
    </AuthContext.Provider>
  );
};
