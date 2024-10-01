import React, { useEffect, useState, useContext } from 'react';
import axios from '../axiosConfig';
import { AuthContext } from '../AuthContext';

function Recommendations() {
  const [videos, setVideos] = useState([]);
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
  const { authTokens } = useContext(AuthContext);

  const fetchRecommendation = async () => {
    if (authTokens) {
      try {
        const response = await axios.get('/api/recommendations', {
          headers: {
            'Authorization': `Bearer ${authTokens}`
          }
        });
        setVideos(prevVideos => [...prevVideos, ...response.data.videos]);
      } catch (error) {
        console.error(error);
      }
    }
  };

  useEffect(() => {
    fetchRecommendation();
  }, [authTokens]);

  const handleNextVideo = () => {
    setCurrentVideoIndex(prevIndex => prevIndex + 1);
    if (currentVideoIndex + 1 >= videos.length) {
      fetchRecommendation();
    }
  };

  const currentVideo = videos[currentVideoIndex];

  return (
    <div>
      <h2>Recommended Videos</h2>
      {currentVideo ? (
        <div>
          <video width="320" height="240" controls>
            <source src={currentVideo.url} type="video/mp4" />
          </video>
          <button onClick={handleNextVideo}>Next Video</button>
        </div>
      ) : (
        <p>Loading recommendations...</p>
      )}
    </div>
  );
}

export default Recommendations;
