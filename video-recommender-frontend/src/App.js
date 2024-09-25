import React from 'react';
import { Switch, Route, Redirect } from 'react-router-dom';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import { AuthContext } from './AuthContext';

function App() {
  const { authTokens } = React.useContext(AuthContext);

  return (
    <div className="App">
      <Switch>
        <Route exact path="/">
          {authTokens ? <Redirect to="/dashboard" /> : <Redirect to="/login" />}
        </Route>
        <Route path="/login">
          {authTokens ? <Redirect to="/dashboard" /> : <Login />}
        </Route>
        <Route path="/register">
          {authTokens ? <Redirect to="/dashboard" /> : <Register />}
        </Route>
        <Route path="/dashboard">
          {authTokens ? <Dashboard /> : <Redirect to="/login" />}
        </Route>
      </Switch>
    </div>
  );
}

export default App;
