document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('contract-file');
    const fileNameDisplay = document.getElementById('file-name');
    const uploadForm = document.querySelector('.upload-form');
    const resultContent = document.getElementById('result-content');
    const emptyState = document.getElementById('empty-state');
    const submitBtn = document.querySelector('.submit-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('spinner');

    function setLoading(isLoading) {
        submitBtn.disabled = isLoading;
        btnText.textContent = isLoading ? 'Analisando contrato...' : 'Analisar Contrato';
        spinner.style.display = isLoading ? 'inline-block' : 'none';
    }

    fileInput.addEventListener('change', function () {
        fileNameDisplay.textContent = this.files.length > 0 ? this.files[0].name : 'Nenhum arquivo selecionado';
    });

    uploadForm.addEventListener('submit', async function (e) {
        e.preventDefault();

        if (!fileInput.files.length) {
            alert('Por favor, selecione um arquivo');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        setLoading(true);
        resultContent.style.display = 'none';
        resultContent.textContent = '';
        emptyState.style.display = 'none';

        try {
            const response = await fetch('http://localhost:8080/analisar', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (response.ok) {
                resultContent.textContent = JSON.stringify(data.analise, null, 2);
                resultContent.style.display = 'block';
            } else {
                alert(data.mensagem || 'Erro ao analisar contrato');
                emptyState.style.display = 'flex';
            }
        } catch (err) {
            console.error(err);
            alert('Erro de rede ou servidor');
            emptyState.style.display = 'flex';
        } finally {
            setLoading(false);
        }
    });
});
