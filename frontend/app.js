const DETAILED_API_URL = '/similarity/detailed';
const BASIC_API_URL = '/similarity';
const MODELS_API_URL = '/models';

let attentionData = null;
let availableModels = [];
let defaultModelId = null;

document.addEventListener('DOMContentLoaded', async () => {
    document.getElementById('analyzeButton').addEventListener('click', calculateSimilarity);
    document.getElementById('compareButton').addEventListener('click', compareModels);
    document.getElementById('modelSelect').addEventListener('change', updateModelHint);
    await fetchModels();
});

async function fetchModels() {
    const errorDiv = document.getElementById('error');

    try {
        const response = await fetch(MODELS_API_URL);
        if (!response.ok) {
            throw new Error('模型列表加载失败');
        }

        const data = await response.json();
        availableModels = data.models || [];
        defaultModelId = data.default_model_id;
        populateModelSelect();
        updateModelHint();
    } catch (error) {
        errorDiv.textContent = `初始化失败: ${error.message}`;
        errorDiv.classList.remove('hidden');
    }
}

function populateModelSelect() {
    const select = document.getElementById('modelSelect');
    select.innerHTML = '';

    for (const model of availableModels) {
        const option = document.createElement('option');
        option.value = model.model_id;
        option.textContent = `${model.name}${model.available ? '' : '（未就绪）'}`;
        option.disabled = !model.available;
        if (model.model_id === defaultModelId) {
            option.selected = true;
        }
        select.appendChild(option);
    }
}

function updateModelHint() {
    const currentModel = getSelectedModel();
    const hint = document.getElementById('modelHint');

    if (!currentModel) {
        hint.textContent = '当前没有可用模型。';
        return;
    }

    hint.textContent = currentModel.description;
}

function getSelectedModel() {
    const selectedId = document.getElementById('modelSelect').value;
    return availableModels.find((model) => model.model_id === selectedId) || null;
}

function getInputPair() {
    const sentence1 = document.getElementById('sentence1').value.trim();
    const sentence2 = document.getElementById('sentence2').value.trim();

    if (!sentence1 || !sentence2) {
        throw new Error('请输入两个句子');
    }

    return { sentence1, sentence2 };
}

function setLoading(isLoading) {
    document.getElementById('loading').classList.toggle('hidden', !isLoading);
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function clearState() {
    document.getElementById('error').classList.add('hidden');
    document.getElementById('result').classList.add('hidden');
    document.getElementById('comparison').classList.add('hidden');
    document.getElementById('visualization').classList.add('hidden');
}

async function calculateSimilarity() {
    clearState();

    let payload;
    try {
        payload = {
            ...getInputPair(),
            model_id: document.getElementById('modelSelect').value,
        };
    } catch (error) {
        showError(error.message);
        return;
    }

    setLoading(true);
    try {
        const response = await fetch(DETAILED_API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || '请求失败');
        }

        renderSingleResult(data);
        if (data.attentions && data.layer_scores && data.layer_scores.length > 0) {
            initVisualization(data);
        }
    } catch (error) {
        showError(`计算失败: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

async function compareModels() {
    clearState();

    let inputPair;
    try {
        inputPair = getInputPair();
    } catch (error) {
        showError(error.message);
        return;
    }

    if (availableModels.length === 0) {
        showError('没有可用于对比的模型');
        return;
    }

    setLoading(true);
    try {
        const results = await Promise.all(
            availableModels.map(async (model) => {
                if (!model.available) {
                    return {
                        model_id: model.model_id,
                        name: model.name,
                        error: '模型未就绪，请先完成微调并导出到本地目录。',
                    };
                }

                const response = await fetch(BASIC_API_URL, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...inputPair, model_id: model.model_id }),
                });

                const data = await response.json();
                if (!response.ok) {
                    return {
                        model_id: model.model_id,
                        name: model.name,
                        error: data.detail || '请求失败',
                    };
                }

                return {
                    ...data,
                    name: model.name,
                };
            })
        );

        renderComparison(results);
    } catch (error) {
        showError(`对比失败: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

function renderSingleResult(data) {
    const model = availableModels.find((item) => item.model_id === data.model_id);
    const normalized = toPercentage(data.normalized_score);

    document.getElementById('resultModelName').textContent = model ? model.name : data.model_id;
    document.getElementById('scoreValue').textContent = formatScore(data.score);
    document.getElementById('scoreAux').textContent = `归一化分数: ${normalized.toFixed(1)}%`;
    document.getElementById('scoreFill').style.width = `${normalized}%`;
    document.getElementById('timeValue').textContent = `计算耗时: ${data.time_ms} ms`;
    document.getElementById('result').classList.remove('hidden');
}

function renderComparison(results) {
    const list = document.getElementById('comparisonList');
    list.innerHTML = '';

    results.sort((left, right) => {
        const leftScore = left.normalized_score ?? -1;
        const rightScore = right.normalized_score ?? -1;
        return rightScore - leftScore;
    });

    for (const item of results) {
        const card = document.createElement('div');
        card.className = 'comparison-card';

        if (item.error) {
            card.innerHTML = `
                <div class="comparison-row">
                    <div>
                        <div class="comparison-name">${item.name}</div>
                        <div class="comparison-status error-text">${item.error}</div>
                    </div>
                </div>
            `;
            list.appendChild(card);
            continue;
        }

        const normalized = toPercentage(item.normalized_score);
        card.innerHTML = `
            <div class="comparison-row">
                <div>
                    <div class="comparison-name">${item.name}</div>
                    <div class="comparison-status">model_id: ${item.model_id}</div>
                </div>
                <div class="comparison-score">${formatScore(item.score)}</div>
            </div>
            <div class="mini-bar">
                <div class="mini-bar-fill" style="width: ${normalized}%"></div>
            </div>
            <div class="comparison-meta">
                <span>归一化: ${normalized.toFixed(1)}%</span>
                <span>耗时: ${item.time_ms} ms</span>
            </div>
        `;
        list.appendChild(card);
    }

    document.getElementById('comparison').classList.remove('hidden');
}

function initVisualization(data) {
    attentionData = data;
    drawScoreChart(data.layer_scores);

    const layerSelect = document.getElementById('layerSelect');
    layerSelect.innerHTML = '';
    for (let i = 0; i < data.attentions.length; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `第 ${i + 1} 层`;
        layerSelect.appendChild(option);
    }

    layerSelect.value = data.attentions.length - 1;
    layerSelect.onchange = () => drawAttention(parseInt(layerSelect.value, 10));

    drawAttention(data.attentions.length - 1);
    document.getElementById('visualization').classList.remove('hidden');
}

function drawScoreChart(scores) {
    const canvas = document.getElementById('scoreChart');
    const ctx = canvas.getContext('2d');
    canvas.width = 800;
    canvas.height = 300;

    const padding = 50;
    const width = canvas.width - 2 * padding;
    const height = canvas.height - 2 * padding;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, padding + height);
    ctx.lineTo(padding + width, padding + height);
    ctx.stroke();

    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 2;
    ctx.beginPath();
    const stepX = scores.length > 1 ? width / (scores.length - 1) : width;
    for (let i = 0; i < scores.length; i++) {
        const x = padding + i * stepX;
        const y = padding + height - (scores[i] * height);
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    ctx.stroke();

    ctx.fillStyle = '#667eea';
    for (let i = 0; i < scores.length; i++) {
        const x = padding + i * stepX;
        const y = padding + height - (scores[i] * height);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();
    }

    ctx.fillStyle = '#666';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    for (let i = 0; i < scores.length; i += 3) {
        const x = padding + i * stepX;
        ctx.fillText(`L${i + 1}`, x, padding + height + 20);
    }

    ctx.textAlign = 'right';
    ctx.fillText('0.0', padding - 10, padding + height);
    ctx.fillText('1.0', padding - 10, padding);
}

function drawAttention(layerIndex) {
    const canvas = document.getElementById('attentionCanvas');
    const ctx = canvas.getContext('2d');
    const attention = attentionData.attentions[layerIndex];
    const tokens = attentionData.tokens;
    const len = attention.length;

    const cellSize = 25;
    const labelWidth = 80;
    const labelHeight = 60;

    canvas.width = len * cellSize + labelWidth;
    canvas.height = len * cellSize + labelHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < len; i++) {
        for (let j = 0; j < len; j++) {
            const value = attention[i][j];
            ctx.fillStyle = valueToColor(value);
            ctx.fillRect(j * cellSize + labelWidth, i * cellSize + labelHeight, cellSize, cellSize);
        }
    }

    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= len; i++) {
        ctx.beginPath();
        ctx.moveTo(labelWidth, i * cellSize + labelHeight);
        ctx.lineTo(len * cellSize + labelWidth, i * cellSize + labelHeight);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(i * cellSize + labelWidth, labelHeight);
        ctx.lineTo(i * cellSize + labelWidth, len * cellSize + labelHeight);
        ctx.stroke();
    }

    ctx.fillStyle = '#333';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i < Math.min(len, 20); i++) {
        ctx.fillText(tokens[i].substring(0, 8), labelWidth - 5, i * cellSize + labelHeight + cellSize / 2 + 4);
    }

    canvas.onmousemove = (event) => handleHover(event, canvas, layerIndex, cellSize, labelWidth, labelHeight);
    canvas.onmouseleave = () => document.getElementById('hoverInfo').classList.add('hidden');
}

function valueToColor(value) {
    const normalized = Math.max(0, Math.min(1, value));
    const hue = (1 - normalized) * 240;
    return `hsl(${hue}, 100%, ${50 + normalized * 20}%)`;
}

function handleHover(event, canvas, layerIndex, cellSize, labelWidth, labelHeight) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left - labelWidth;
    const y = event.clientY - rect.top - labelHeight;

    if (x < 0 || y < 0) {
        return;
    }

    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);
    const attention = attentionData.attentions[layerIndex];

    if (row >= 0 && row < attention.length && col >= 0 && col < attention.length) {
        const value = attention[row][col];
        const tokens = attentionData.tokens;
        const info = document.getElementById('hoverInfo');
        info.textContent = `${tokens[row]} → ${tokens[col]}: ${value.toFixed(4)}`;
        info.style.left = `${event.clientX + 10}px`;
        info.style.top = `${event.clientY + 10}px`;
        info.classList.remove('hidden');
    }
}

function formatScore(value) {
    return Number(value).toFixed(4);
}

function toPercentage(value) {
    return Math.max(0, Math.min(100, Number(value) * 100));
}
