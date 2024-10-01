import React, { useContext } from 'react';
import { AuthContext } from '../AuthContext';
import { useNavigate } from 'react-router-dom';

function Logout() {
  const { setAuthTokens } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleLogout = () => {
    setAuthTokens(null);
    localStorage.removeItem('tokens');
    navigate('/login');
  };

  return <button onClick={handleLogout}>Logout</button>;
}

export default Logout;
