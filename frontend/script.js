async function uploadFile() {
    const fileInput = document.getElementById('audioFile')
    const progressDiv = document.getElementById('progress')
    const resultDiv = document.getElementById('result')

    const formData = new FormData();
    formData.append('file', fileInput.files[0])
    
    progressDiv.innerHTML = 'Обработка...'

    resultDiv.innerHTML = ''

    try {
        const response = await fetch('http://localhost:8000:transcribe/', {
            method: 'POST',
            body: formData
        })

        if (!response.ok) throw new Error('Ошибка сервера')

        const data = await response.json()
        displayResults(data);
        progressDiv.innerHTML = 'Ошибка' + error.message
    } catch (error) {
        progressDiv.innerHTML = 'Ошибка' + error.message
    }
}

function displayResults(data) {
    const resultDiv = document.getElementById('result')
    let html = '<h2>Результат:</h2>'

    html += `<p>${data.result}</p>`
    resultDiv.innerHTML = html
}

