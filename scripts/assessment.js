document.addEventListener("DOMContentLoaded", function() {
    // The questions that if answered with 'yes', will show the child questions
    const questionDependencies = {
        "1.1.5": ["1.1.6", "1.1.7", "1.2"]
    };

    var questionsAssociations = {};

    // Recursive function to render questions based on the tree structure
    function renderQuestions(container, questionsTree) {
        const ageGroup = sessionStorage.getItem('ageGroup');

        for (const key in questionsTree) {
            questionsAssociations[key] = questionsTree[key].question;

            const questionData = questionsTree[key];
            const div = document.createElement('div');
            div.classList.add('question');
            div.dataset.key = key; // Store the question key in data attribute

            if (questionData.type === "header") {
                const header = document.createElement('h2');
                header.textContent = questionData.question;
                div.appendChild(header);
            } else if (questionData.type === "subheader") {
                const subheader = document.createElement('h3');
                subheader.textContent = questionData.question.replace("children", ageGroup.toLowerCase());
                div.appendChild(subheader);
            } else {
                const label = document.createElement('label');
                label.textContent = questionData.question;
                div.appendChild(label);

                let inputContainer;
                if (questionData.type === "text") {
                    inputContainer = document.createElement('textarea');
                    inputContainer.rows = 5;
                } else if (questionData.type === "number") {
                    inputContainer = document.createElement('input');
                    inputContainer.type = 'number';
                } else if (questionData.type === "yesno") {
                    inputContainer = document.createElement('div');

                    const labelYes = document.createElement('label');
                    labelYes.textContent = 'Yes';
                    labelYes.htmlFor = `${key}-yes`;

                    const inputYes = document.createElement('input');
                    inputYes.type = 'radio';
                    inputYes.name = key;
                    inputYes.value = 'yes';
                    inputYes.id = `${key}-yes`;

                    const labelNo = document.createElement('label');
                    labelNo.textContent = 'No';
                    labelNo.htmlFor = `${key}-no`;                    

                    const inputNo = document.createElement('input');
                    inputNo.type = 'radio';
                    inputNo.name = key;
                    inputNo.value = 'no';
                    inputNo.id = `${key}-no`;

                    const yesContainer = document.createElement('div');
                    yesContainer.classList.add('radio-container');
                    yesContainer.appendChild(labelYes);
                    yesContainer.appendChild(inputYes);

                    const noContainer = document.createElement('div');
                    noContainer.classList.add('radio-container');
                    noContainer.appendChild(labelNo);
                    noContainer.appendChild(inputNo);

                    inputContainer.appendChild(yesContainer);
                    inputContainer.appendChild(noContainer);

                    inputContainer.querySelectorAll('input').forEach(input => {
                        input.addEventListener('change', function() {
                            handleDependencies(key, input.value);
                        });
                    });
                }

                inputContainer.dataset.question = questionData.question; // Store the question text in data attribute
                inputContainer.dataset.key = key; // Store the question key in data attribute
                div.appendChild(inputContainer);
            }

            // Hide dependent questions initially
            if (Object.values(questionDependencies).flat().includes(key)) {
                div.style.display = "none";
            }

            container.appendChild(div);

            if (Object.keys(questionData.children).length > 0) {
                const childContainer = document.createElement('div');
                childContainer.classList.add('child-questions');
                container.appendChild(childContainer);

                renderQuestions(childContainer, questionData.children);
            }
        }
    }

    // Function to handle showing questions dependent on answers given in previous questions
    function handleDependencies(questionKey, value) {
        const questionsContainer = document.getElementById('questionsContainer');
        const dependentQuestions = questionDependencies[questionKey];

        if (dependentQuestions) {
            dependentQuestions.forEach(depKey => {
                const questionDiv = questionsContainer.querySelector(`[data-key="${depKey}"]`);
                if (questionDiv) {
                    if (value === "no") {
                        questionDiv.style.display = "block";
                    } else {
                        questionDiv.style.display = "none";
                    }
                }
            });
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
                if (input.type === 'radio') {
                    if (input.checked) {
                        // Check if dataset.question exists, if not, take from questionsAssociations
                        if (typeof input.dataset.question === "undefined") {
                            questionId = input.id.split('-')[0]
                            answers[questionsAssociations[questionId]] = input.value
                        }
                    }
                } else if (input.value.trim() !== "") {
                    answers[input.dataset.question] = input.value;
                }
            });
            sessionStorage.setItem('answers', JSON.stringify(answers));
            window.location.href = 'summary.html';
        });
    }

    fetch('questions.json')
        .then(response => response.json())
        .then(questionsTree => {
            initializeAssessment(questionsTree);
        })
        .catch(error => console.error('Error loading questions:', error));
});
