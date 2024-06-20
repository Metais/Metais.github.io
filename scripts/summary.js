document.addEventListener("DOMContentLoaded", function() {
    // Generate summary
    function generateSummary() {
        const drug = sessionStorage.getItem('drug');
        const ageGroup = sessionStorage.getItem('ageGroup');
        const indication = sessionStorage.getItem('indication');
        const answers = JSON.parse(sessionStorage.getItem('answers'));
        const summaryContainer = document.getElementById('summaryContainer');

        let summary = `<p>Drug: ${drug}</p>`;
        summary += `<p>Age Group: ${ageGroup}</p>`;
        summary += `<p>Indication: ${indication}</p>`;
        summary += '<p>Answers:</p><ul>';
        for (const question in answers) {
            summary += `<li>${question}: ${answers[question]}</li>`;
        }
        summary += '</ul>';

        summaryContainer.innerHTML = summary;
    }

    generateSummary();
});