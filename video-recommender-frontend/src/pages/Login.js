import React, { useState, useContext } from 'react';
import axios from 'axios';
import { AuthContext } from '../AuthContext';
import { useNavigate, Link } from 'react-router-dom';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { setAuthTokens } = useContext(AuthContext);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('/api/login', { email, password });
      setAuthTokens(response.data.access_token);
      navigate('/dashboard');
    } catch (error) {
      console.error(error);
      alert('Invalid email or password');
    }
  };

  return (
    <div>
      <h2>Login</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>Email:</label><br />
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
        </div>
        <div>
          <label>Password:</label><br />
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required />
        </div><br />
        <button type="submit">Login</button>
        <p>
          Don't have an account? <Link to="/register">Register here</Link>
        </p>
      </form>
    </div>
  );
}

export default Login;
