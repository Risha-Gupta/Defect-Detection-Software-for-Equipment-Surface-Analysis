import { useState } from "react";

function UploadForm() {
    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [selectedModel, setSelectedModel] = useState("simple_cnn");

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleModelChange = (e) => {
        setSelectedModel(e.target.value);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!file) return alert("Please select an image first!");

        const formData = new FormData();
        formData.append("file", file);
        formData.append("model", selectedModel);

        try {
            setLoading(true);
            setResult(null);

            const response = await fetch("http://localhost:8000/predict", {
                method: "POST",
                body: formData,
            });
            const data = await response.json();
            setResult(data);
        } catch (err) {
            console.error("Error uploading file:", err);
            alert("Upload failed");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex items-center justify-center min-h-screen bg-gray-50">
            <div className="w-full max-w-sm p-6 bg-white border border-gray-200 rounded-lg">
                <h2 className="text-xl font-medium text-gray-800 text-center mb-4">
                    Equipment Defect Detection
                </h2>

                <div className="space-y-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                            Select Model:
                        </label>
                        <select
                            value={selectedModel}
                            onChange={handleModelChange}
                            className="w-full p-2 border border-gray-300 rounded-md text-sm"
                        >
                            <option value="simple_cnn">Simple CNN</option>
                            <option value="complex_cnn">Complex CNN</option>
                            <option value="mobilenet_v2">MobileNetV2</option>
                        </select>
                    </div>
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        className="block w-full text-sm text-gray-600
                                file:mr-3 file:py-1 file:px-3
                                file:rounded-md file:border-0
                                file:bg-gray-100 file:text-gray-700
                                hover:file:bg-gray-200"
                    />

                    <button
                        onClick={handleSubmit}
                        className="w-full bg-gray-800 hover:bg-gray-700 text-white py-2 rounded-md text-sm"
                    >
                        {loading ? "Analyzing..." : "Upload & Predict"}
                    </button>
                </div>

                {result && (
                    <div className="mt-4 p-3 border border-gray-200 rounded-md bg-gray-50 text-sm">
                        <p>
                            <span className="font-semibold">Model Used:</span> {selectedModel}
                        </p>
                        <p>
                            <span className="font-semibold">Status:</span>{" "}
                            <span
                                className={
                                    result.label === "damaged"
                                        ? "text-red-600 font-medium"
                                        : "text-green-600 font-medium"
                                }
                            >
                                {result.label}
                            </span>
                        </p>
                        <p>
                            <span className="font-semibold">Confidence:</span>{" "}
                            {(result.confidence * 100).toFixed(2)}%
                        </p>
                        {result.defect_type && (
                            <p>
                                <span className="font-semibold">Defect Type:</span> {result.defect_type}
                            </p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export default UploadForm;