import React, { useState, useContext } from 'react';
import axios from '../axiosConfig';
import { AuthContext } from '../AuthContext';

function UploadVideo() {
  const [videoFile, setVideoFile] = useState(null);
  const { authTokens } = useContext(AuthContext);

  const handleFileChange = (e) => {
    setVideoFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!videoFile) return;
    const formData = new FormData();
    formData.append('video', videoFile);

    try {
      await axios.post('/api/upload', formData, {
        headers: {
          'Authorization': `Bearer ${authTokens}`
        }
      });
      alert('Video uploaded successfully!');
    } catch (error) {
      console.error(error);
      alert('Failed to upload video.');
    }
  };

  return (
    <div>
      <h2>Upload Video</h2>
      <input type="file" accept="video/*" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>
    </div>
  );
}

export default UploadVideo;
