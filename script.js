document.getElementById("predictBtn").addEventListener("click", async () => {
    const inputText = document.getElementById("inputText").value.trim();
    const predictionEl = document.getElementById("prediction");

    if (!inputText) {
        predictionEl.textContent = "Please enter some text first!";
        return;
    }

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: inputText })
   });


        const data = await response.json();
        predictionEl.textContent = `Next word might be: "${data.prediction}"`;

    } catch (error) {
        predictionEl.textContent = "Error: Unable to connect to backend.";
        console.error(error);
    }
});
