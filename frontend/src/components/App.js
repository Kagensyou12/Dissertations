import React, { useState } from "react";
import axios from "axios";

function App() {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState("");
    const [heatmapUrl, setHeatmapUrl] = useState("");

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) {
            alert("Please select a file first.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await axios.post("http://127.0.0.1:5000/upload", formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });

            setResult(response.data.result);
            setHeatmapUrl(`http://127.0.0.1:5000${response.data.heatmap_url}`);
        } catch (error) {
            console.error("Error uploading file", error);
            setResult("Error processing image");
        }
    };

    return (
        <div style={{ textAlign: "center", padding: "20px" }}>
            <h1>AI Image Detection</h1>
            <input type="file" onChange={handleFileChange} />
            <button onClick={handleUpload}>Upload</button>
            {result && <h2>Result: {result}</h2>}
            {heatmapUrl && <img src={heatmapUrl} alt="Grad-CAM Heatmap" style={{ width: "224px" }} />}
        </div>
    );
}

export default App;
