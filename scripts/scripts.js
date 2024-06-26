document.addEventListener("DOMContentLoaded", function() {
    // Function to populate drug options
    function populateDrugOptions(drugs) {
        const drugSelect = document.getElementById('drug');
        for (const drug in drugs) {
            const option = document.createElement('option');
            option.value = drug;
            option.textContent = drug;
            drugSelect.appendChild(option);
        }
    }

    // Function to update indications based on selected drug
    function updateIndications(drugs) {
        const drugSelect = document.getElementById('drug');
        const indicationSelect = document.getElementById('indication');
        
        drugSelect.addEventListener('change', function() {
            indicationSelect.innerHTML = '<option value="" selected disabled>Select an indication</option>';
            const selectedDrug = drugSelect.value;
            if (selectedDrug) {
                drugs[selectedDrug].forEach(indication => {
                    const option = document.createElement('option');
                    option.value = indication;
                    option.textContent = indication;
                    indicationSelect.appendChild(option);
                });
            }
        });
    }

    // Recursive function to render questions based on the tree structure
    function renderQuestions(container, questionsTree) {
        const ageGroup = sessionStorage.getItem('ageGroup');

        for (const key in questionsTree) {
            const questionData = questionsTree[key];
            const div = document.createElement('div');
            div.classList.add('question');

            if (questionData.type === "header") {
                const header = document.createElement('h2');
                header.textContent = questionData.question;
                div.appendChild(header);
            } else if (questionData.type === "subheader") {
                const subheader = document.createElement('h3');
                subheader.textContent = questionData.question.replace("children", ageGroup);
                div.appendChild(subheader);
            } else {
                const label = document.createElement('label');
                label.textContent = questionData.question;
                div.appendChild(label);

                let input;
                if (questionData.type === "text") {
                    input = document.createElement('textarea');
                    input.rows = 5;
                } else if (questionData.type === "number") {
                    input = document.createElement('input');
                    input.type = 'number';
                } else if (questionData.type === "yesno") {
                    input = document.createElement('select');
                    const optionYes = document.createElement('option');
                    optionYes.value = "yes";
                    optionYes.textContent = "Yes";
                    input.appendChild(optionYes);

                    const optionNo = document.createElement('option');
                    optionNo.value = "no";
                    optionNo.textContent = "No";
                    input.appendChild(optionNo);
                }

                input.dataset.question = questionData.question; // Store the question text in data attribute
                div.appendChild(input);
            }

            container.appendChild(div);

            if (Object.keys(questionData.children).length > 0) {
                const childContainer = document.createElement('div');
                childContainer.classList.add('child-questions');
                container.appendChild(childContainer);

                if (questionData.type !== "header" && questionData.type !== "subheader") {
                    input.addEventListener('keypress', function(event) {
                        if (event.key === 'Enter' && !event.shiftKey) {
                            event.preventDefault();
                            // Render child questions only if there's an input value
                            if (input.value.trim() !== "") {
                                childContainer.innerHTML = ''; // Clear previous questions
                                renderQuestions(childContainer, questionData.children);
                            }
                        }
                    });
                } else {
                    renderQuestions(childContainer, questionData.children);
                }
            }
        }
    }

    // Function to initialize assessment questions
    function initializeAssessment(questionsTree) {
        const questionsContainer = document.getElementById('questionsContainer');
        renderQuestions(questionsContainer, questionsTree);

        document.getElementById('assessmentForm').addEventListener('submit', function(event) {
            event.preventDefault();
            // Collect answers and generate summary
            const answers = {};
            questionsContainer.querySelectorAll('textarea, input, select').forEach(input => {
                if (input.value.trim() !== "") {
                    answers[input.dataset.question] = input.value;
                }
            });
            sessionStorage.setItem('answers', JSON.stringify(answers));
            window.location.href = 'summary.html';
        });
    }

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

    if (!window.location.pathname.endsWith('assessment.html') && !window.location.pathname.endsWith('summary.html')) {
        // Fetch drug data from JSON file
        fetch('drugs.json')
            .then(response => response.json())
            .then(drugs => {
                populateDrugOptions(drugs);
                updateIndications(drugs);

                // Handle form submission
                document.getElementById('selectionForm').addEventListener('submit', function(event) {
                    event.preventDefault();
                    const drug = document.getElementById('drug').value;
                    const ageGroup = document.getElementById('ageGroup').value;
                    const indication = document.getElementById('indication').value;
                    sessionStorage.setItem('drug', drug);
                    sessionStorage.setItem('ageGroup', ageGroup);
                    sessionStorage.setItem('indication', indication);
                    window.location.href = 'assessment.html';
                });
            })
            .catch(error => console.error('Error loading drugs:', error));
    }

    // Fetch questions data from JSON file
    fetch('questions.json')
        .then(response => response.json())
        .then(questionsTree => {
            if (window.location.pathname.endsWith('assessment.html')) {
                initializeAssessment(questionsTree);
            } else if (window.location.pathname.endsWith('summary.html')) {
                generateSummary();
            }
        })
        .catch(error => console.error('Error loading questions:', error));
});
