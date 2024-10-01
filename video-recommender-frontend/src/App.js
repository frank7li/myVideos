import React from 'react';
import { HashRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import { AuthProvider, AuthContext } from './AuthContext';

function AppRoutes() {
  const { authTokens } = React.useContext(AuthContext);
  console.log('AppRoutes rendering, authTokens:', authTokens);

  return (
    <Routes>
      <Route path="/" element={
        <>
          {console.log('Root route rendering')}
          {authTokens ? <Navigate to="/dashboard" /> : <Navigate to="/login" />}
        </>
      } />
      <Route path="/login" element={
        <>
          {console.log('Login route rendering')}
          {authTokens ? <Navigate to="/dashboard" /> : <Login />}
        </>
      } />
      <Route path="/register" element={
        <>
          {console.log('Register route rendering')}
          {authTokens ? <Navigate to="/dashboard" /> : <Register />}
        </>
      } />
      <Route path="/dashboard" element={
        <>
          {console.log('Dashboard route rendering')}
          {authTokens ? <Dashboard /> : <Navigate to="/login" />}
        </>
      } />
      <Route path="*" element={<div>Page not found</div>} />
    </Routes>
  );
}

function App() {
  console.log('App rendering');
  return (
    <AuthProvider>
      <Router>
        <div className="App">
          <AppRoutes />
        </div>
      </Router>
    </AuthProvider>
  );
}

export default App;
