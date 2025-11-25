document.getElementById("recommend-form").addEventListener("submit", function(e) {
    e.preventDefault();
    const queryVal = document.getElementById("query").value.trim();
    const resultsDiv = document.getElementById("results");
    if(!queryVal) {
        resultsDiv.innerHTML = "<span style='color:red'>Please enter a requirement.</span>";
        return;
    }
    resultsDiv.innerHTML = "<em>Loading recommendations...</em>";
    fetch("http://localhost:8000/recommend", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({"query": queryVal})
    })
    .then(r => {
        if (!r.ok) throw new Error("Server error");
        return r.json();
    })
    .then(data => {
        if (!data || !Array.isArray(data.recommended_assessments) || data.recommended_assessments.length === 0) {
            resultsDiv.innerHTML = "<span>No recommendations found.</span>";
            return;
        }
        let html = "";
        data.recommended_assessments.forEach(assess => {
            html += `
            <div class="assessment">
                <h3>
                    ${assess.url ? `<a href="${assess.url}" target="_blank">${assess.name || "Unnamed"}</a>` : (assess.name || "Unnamed")}
                </h3>
                <ul>
                    <li><b>Adaptive Support:</b> ${assess.adaptive_support || "-"}</li>
                    <li><b>Duration:</b> ${assess.duration || "-"}</li>
                    <li><b>Remote Support:</b> ${assess.remote_support || "-"}</li>
                    <li><b>Type:</b> ${Array.isArray(assess.test_type) ? assess.test_type.join(", ") : (assess.test_type || "-")}</li>
                </ul>
                <p>${assess.description || ""}</p>
            </div>
            `;
        });
        resultsDiv.innerHTML = html;
    })
    .catch(error => {
        resultsDiv.innerHTML = "<span style='color:red'>Error fetching recommendations.</span>";
        console.error(error);
    });
});