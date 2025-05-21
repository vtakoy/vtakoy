document.addEventListener('DOMContentLoaded', () => {
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const resultsContainer = document.getElementById('results-container');
    const resultsList = document.getElementById('results-list');
    const loadingElement = document.getElementById('loading');
    const noResultsElement = document.getElementById('no-results');
    
    // Добавляем переменные для таймера загрузки и хранения истории
    let loadingTimer = null;
    const MAX_LOADING_TIME = 40000; // Максимальное время загрузки (40 секунд)
    let sessionHistory = []; // Массив для хранения истории запросов и ответов

    console.log('Инициализация JavaScript...');

    // Обработка отправки формы
    queryForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const query = queryInput.value.trim();
        if (!query) return;
        
        console.log('Отправка запроса:', query);
        
        // Показываем индикатор загрузки
        showLoading();
        
        // Устанавливаем таймер для принудительного скрытия загрузки
        if (loadingTimer) clearTimeout(loadingTimer);
        loadingTimer = setTimeout(() => {
            console.log('Превышено максимальное время загрузки, принудительно скрываем индикатор');
            hideLoading();
            showError('Превышено время ожидания ответа от сервера. Попробуйте упростить запрос.');
        }, MAX_LOADING_TIME);
        
        try {
            const formData = new FormData();
            formData.append('query', query);
            
            console.log('Отправка POST запроса на /analyze');
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            
            // Очищаем таймер после получения ответа
            if (loadingTimer) {
                clearTimeout(loadingTimer);
                loadingTimer = null;
            }
            
            if (!response.ok) {
                throw new Error(`Ошибка HTTP: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Получен ответ:', data);
            
            // Очищаем поле ввода после получения результатов
            queryInput.value = '';
            
            // Передаем запрос для отображения в заголовке результатов и сохранения в историю
            displayResults(data, query, true);
            
        } catch (error) {
            console.error('Ошибка при запросе:', error);
            
            // Очищаем таймер при ошибке
            if (loadingTimer) {
                clearTimeout(loadingTimer);
                loadingTimer = null;
            }
            
            showError(error.message);
        }
    });

    // Функция отображения результатов
    function displayResults(data, query, addToHistory = false) {
        console.log('Отображение результатов:', data);
        
        // Скрываем индикатор загрузки
        hideLoading();
        
        // Если не удалось найти информацию, показываем соответствующее сообщение
        if (!data || data.status === 'not_found') {
            // Показываем сообщение "Не найдено"
            noResultsElement.classList.remove('hidden');
            
            // Если есть результаты в виде текста, отображаем их
            if (data && data.results) {
                const message = typeof data.results === 'string' 
                    ? data.results 
                    : 'Информация по запросу не найдена в документах';
                    
                noResultsElement.querySelector('p').textContent = message;
            }
            
            // Если есть другие результаты, продолжаем их показывать
            if (sessionHistory.length > 0) {
                resultsContainer.classList.remove('hidden');
            } else {
                resultsContainer.classList.add('hidden');
            }
            
            return;
        }
        
        // Скрываем сообщение о ненайденных результатах
        noResultsElement.classList.add('hidden');
        
        // Создаем новый элемент результата запроса
        const resultElement = document.createElement('div');
        resultElement.className = 'search-result-item';
        
        // Добавляем заголовок с текстом запроса
        if (query) {
            const queryHeader = document.createElement('h3');
            queryHeader.className = 'query-header';
            queryHeader.textContent = query;
            resultElement.appendChild(queryHeader);
        }
        
        // Добавляем результаты поиска
        const resultItemsList = document.createElement('div');
        resultItemsList.className = 'result-items-list';
        
        // Если результатов нет в ожидаемом формате, проверяем есть ли текст в data.results
        if (!data.results || (Array.isArray(data.results) && data.results.length === 0)) {
            // Проверяем, является ли results строкой
            if (typeof data.results === 'string' && data.results.trim()) {
                // Создаем один результат из текста
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'result-content';
                
                // Разбиваем на абзацы
                const paragraphs = data.results.split('\n').filter(p => p.trim() !== '');
                
                if (paragraphs.length > 0) {
                    paragraphs.forEach(paragraph => {
                        const p = document.createElement('p');
                        p.textContent = paragraph;
                        contentDiv.appendChild(p);
                    });
                } else {
                    contentDiv.textContent = data.results;
                }
                
                resultItem.appendChild(contentDiv);
                resultItemsList.appendChild(resultItem);
            } else {
                const noResultsDiv = document.createElement('div');
                noResultsDiv.className = 'no-results';
                noResultsDiv.textContent = 'Результаты получены, но не удалось структурировать ответ';
                resultItemsList.appendChild(noResultsDiv);
            }
        } else {
            // Добавляем каждый результат в список
            data.results.forEach((result, index) => {
                console.log(`Обработка результата ${index}:`, result);
                
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                
                // Метаданные результата
                const metaDiv = document.createElement('div');
                metaDiv.className = 'result-meta';
                
                // Документ
                if (result.document) {
                    console.log(`Отображение документа: "${result.document}"`);
                    const docMeta = document.createElement('div');
                    docMeta.className = 'meta-item document';
                    docMeta.innerHTML = `<i class="fas fa-file-alt"></i> ${result.document}`;
                    metaDiv.appendChild(docMeta);
                }
                
                // Раздел
                if (result.section) {
                    const sectionMeta = document.createElement('div');
                    sectionMeta.className = 'meta-item section';
                    sectionMeta.innerHTML = `<i class="fas fa-bookmark"></i> ${result.section}`;
                    metaDiv.appendChild(sectionMeta);
                }
                
                // Страница
                if (result.page && result.page !== "N/A") {
                    const pageMeta = document.createElement('div');
                    pageMeta.className = 'meta-item page';
                    pageMeta.innerHTML = `<i class="fas fa-file-alt"></i> Страница ${result.page}`;
                    metaDiv.appendChild(pageMeta);
                }
                
                // Содержимое результата
                const contentDiv = document.createElement('div');
                contentDiv.className = 'result-content';
                
                // Обработка содержимого в зависимости от его типа
                if (typeof result.content === 'string') {
                    // Разбиваем на абзацы для лучшей читаемости
                    const paragraphs = result.content.split('\n').filter(p => p.trim() !== '');
                    
                    if (paragraphs.length > 0) {
                        paragraphs.forEach(paragraph => {
                            const p = document.createElement('p');
                            p.textContent = paragraph;
                            contentDiv.appendChild(p);
                        });
                    } else {
                        contentDiv.textContent = result.content;
                    }
                } else {
                    contentDiv.textContent = 'Содержимое не доступно';
                }
                
                // Добавляем все элементы в карточку результата
                resultItem.appendChild(metaDiv);
                resultItem.appendChild(contentDiv);
                
                // Добавляем карточку в список результатов
                resultItemsList.appendChild(resultItem);
            });
        }
        
        // Добавляем список результатов в общий контейнер результата запроса
        resultElement.appendChild(resultItemsList);
        
        // Сохраняем результат в истории сессии и отображаем все результаты
        if (addToHistory) {
            // Создаем объект с запросом и ответом для сохранения в истории
            const historyItem = {
                query: query,
                element: resultElement
            };
            
            // Добавляем новый элемент в начало массива истории
            sessionHistory.unshift(historyItem);
            
            // Обновляем отображение всех результатов из истории
            updateResultsDisplay();
        } else {
            // Просто добавляем результат к текущему отображению
            resultsList.appendChild(resultElement);
        }
        
        // Показываем контейнер результатов
        resultsContainer.classList.remove('hidden');
    }
    
    // Функция обновления отображения всех результатов из истории
    function updateResultsDisplay() {
        // Очищаем список результатов
        resultsList.innerHTML = '';
        
        // Добавляем все результаты из истории сессии (новые в начале)
        sessionHistory.forEach(item => {
            resultsList.appendChild(item.element.cloneNode(true));
        });
    }

    // Функция отображения индикатора загрузки
    function showLoading() {
        console.log('Показываем индикатор загрузки');
        loadingElement.classList.remove('hidden');
        noResultsElement.classList.add('hidden');
    }

    // Функция скрытия индикатора загрузки
    function hideLoading() {
        console.log('Скрываем индикатор загрузки');
        loadingElement.classList.add('hidden');
    }

    // Функция отображения ошибки
    function showError(message) {
        console.error('Отображение ошибки:', message);
        hideLoading();
        
        noResultsElement.querySelector('h2').textContent = 'Произошла ошибка';
        noResultsElement.querySelector('p').textContent = message || 'Не удалось обработать запрос. Попробуйте позже.';
        noResultsElement.classList.remove('hidden');
        
        // Если есть результаты истории, продолжаем их показывать
        if (sessionHistory.length > 0) {
            resultsContainer.classList.remove('hidden');
        } else {
            resultsContainer.classList.add('hidden');
        }
    }
    
    console.log('JavaScript инициализирован успешно');
}); 