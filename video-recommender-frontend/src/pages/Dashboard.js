import React from 'react';
import UploadVideo from '../components/UploadVideo';
import Recommendations from '../components/Recommendations';
import Logout from '../components/Logout';

function Dashboard() {
  return (
    <div>
      <h1>Welcome to Video Recommender</h1>
      <Logout />
      <UploadVideo />
      <Recommendations />
    </div>
  );
}

export default Dashboard;
