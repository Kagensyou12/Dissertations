import React, { useState } from "react";
import axios from "axios";

function App() {
    const [file, setFile] = useState(null);
    const [imagePreview, setImagePreview] = useState(null);
    const [result, setResult] = useState("");

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        setFile(selectedFile);

        if (selectedFile) {
            const previewUrl = URL.createObjectURL(selectedFile);
            setImagePreview(previewUrl);
        } else {
            setImagePreview(null);
        }
        setResult(""); // Clear previous result when new file selected
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
        } catch (error) {
            console.error("Error uploading file", error);
            setResult("Error processing image");
        }
    };

    const handleReset = () => {
        setFile(null);
        setImagePreview(null);
        setResult("");
        // Also reset the file input field value manually:
        document.getElementById("fileInput").value = null;
    };

    return (
        <div style={{ textAlign: "center", padding: "20px" }}>
            <h1>AI Image Detection</h1>
            <input id="fileInput" type="file" onChange={handleFileChange} />
            <br />
            <button onClick={handleUpload} style={{ margin: "10px" }}>
                Upload
            </button>
            <button onClick={handleReset} style={{ margin: "10px" }}>
                Reset
            </button>

            {imagePreview && (
                <>
                    <h3>Uploaded Image</h3>
                    <img
                        src={imagePreview}
                        alt="Uploaded Preview"
                        style={{ width: "224px", marginTop: "10px" }}
                    />
                </>
            )}

            {result && <h2>Result: {result}</h2>}
        </div>
    );
}

export default App;
