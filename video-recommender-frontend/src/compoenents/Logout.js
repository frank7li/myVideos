import React, { useContext } from 'react';
import { AuthContext } from '../AuthContext';
import { useHistory } from 'react-router-dom';

function Logout() {
  const { setAuthTokens } = useContext(AuthContext);
  const history = useHistory();

  const handleLogout = () => {
    setAuthTokens(null);
    localStorage.removeItem('tokens');
    history.push('/login');
  };

  return <button onClick={handleLogout}>Logout</button>;
}

export default Logout;
