import React, { useEffect, useState, useContext } from 'react';
import axios from '../axiosConfig';
import { AuthContext } from '../AuthContext';

function Recommendations() {
  const [videos, setVideos] = useState([]);
  const { authTokens } = useContext(AuthContext);

  useEffect(() => {
    if (authTokens) {
      const fetchRecommendations = async () => {
        try {
          const response = await axios.get('/api/recommendations', {
            headers: {
              'Authorization': `Bearer ${authTokens}`
            }
          });
          setVideos(response.data.videos);
        } catch (error) {
          console.error(error);
        }
      };
  
      fetchRecommendations();
    }
}, [authTokens]);
  

  return (
    <div>
      <h2>Recommended Videos</h2>
      {videos.map((video) => (
        <div key={video.id}>
          <video width="320" height="240" controls>
            <source src={video.url} type="video/mp4" />
          </video>
        </div>
      ))}
    </div>
  );
}

export default Recommendations;
