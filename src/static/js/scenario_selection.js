document.addEventListener('DOMContentLoaded', () => {
    const selectButtons = document.querySelectorAll('.select-btn');
    const loadingOverlay = document.getElementById('loading-overlay');

    selectButtons.forEach(button => {
        button.addEventListener('click', async (e) => {
            const scenarioType = e.target.dataset.scenario;
            await selectScenario(scenarioType);
        });
    });

    async function selectScenario(scenarioType) {
        try {
            // Show loading overlay
            loadingOverlay.classList.remove('hidden');

            // Call API to get random case for selected scenario
            const response = await fetch(`/api/select-scenario?scenario_type=${encodeURIComponent(scenarioType)}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const caseData = data.case;

            // Store case data in sessionStorage for game initialization
            sessionStorage.setItem('selectedCase', JSON.stringify(caseData));

            // Redirect to game page
            window.location.href = '/game';

        } catch (error) {
            console.error('Error selecting scenario:', error);
            alert('시나리오 선택 중 오류가 발생했습니다. 다시 시도해주세요.');
            loadingOverlay.classList.add('hidden');
        }
    }
});
