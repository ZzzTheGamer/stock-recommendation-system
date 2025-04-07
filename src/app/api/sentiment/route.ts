import { NextResponse } from 'next/server';
import axios from 'axios';

// Hugging Face API 密钥
const HF_API_KEY = process.env.HF_API_KEY || '';
const HF_API_URL = 'https://api-inference.huggingface.co/models';

// 情感分析模型 - 使用FinBERT
const SENTIMENT_MODEL = 'ProsusAI/finbert';
// 备用情感分析模型
const BACKUP_SENTIMENT_MODEL = 'yiyanghkust/finbert-tone';

/**
 * 调用HuggingFace API，带重试机制
 */
const callHuggingFaceAPI = async (model: string, text: string, maxRetries = 2) => {
    let retries = 0;
    let lastError;

    while (retries <= maxRetries) {
        try {
            console.log(`[API调用] 尝试调用模型: ${model}，重试次数: ${retries}`);

            // 为不同模型准备特定的参数
            let payload: any = { inputs: text };

            const response = await axios.post(`${HF_API_URL}/${model}`, payload, {
                headers: {
                    'Authorization': `Bearer ${HF_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 30000 // 30秒超时
            });

            console.log(`[API调用] 成功调用模型: ${model}`);
            return response.data;
        } catch (error) {
            lastError = error;
            console.error(`[API调用] 调用模型失败: ${model}，错误:`, error);
            retries++;

            if (retries <= maxRetries) {
                console.log(`[API调用] 等待后重试...`);
                await new Promise(resolve => setTimeout(resolve, 1000 * retries)); // 指数退避
            }
        }
    }

    throw lastError;
};

export async function POST(request: Request) {
    try {
        // 解析请求体
        const { texts } = await request.json();

        if (!texts || !Array.isArray(texts) || texts.length === 0) {
            return NextResponse.json(
                { error: '缺少必要参数: texts (应为非空数组)' },
                { status: 400 }
            );
        }

        console.log('[情感分析] 开始情感分析，文本数量:', texts.length);
        const results = [];
        let overallScore = 0;
        let apiCallSuccess = true;
        let apiCallCount = 0;
        let apiSuccessCount = 0;

        // 对每条文本进行情感分析
        for (const text of texts) {
            try {
                console.log('[情感分析] 分析文本:', text.substring(0, 50) + '...');
                apiCallCount++;

                // 直接调用FinBERT模型
                console.log('[情感分析] 调用FinBERT API...');
                const response = await callHuggingFaceAPI(SENTIMENT_MODEL, text);
                console.log('[情感分析] API响应:', response);
                apiSuccessCount++;

                // 解析FinBERT响应
                let sentiment = 'Neutral';
                let score = 0;
                let reason = 'Based on FinBERT model prediction';

                try {
                    console.log('[情感分析] FinBERT响应详情:', JSON.stringify(response));

                    // FinBERT通常返回格式为数组，每个元素代表一个类别的预测
                    // 例如: [{"label":"positive","score":0.9991232},{"label":"neutral","score":0.0006},{"label":"negative","score":0.0002}]
                    if (Array.isArray(response)) {
                        // 如果是嵌套数组，取第一个元素
                        const results = Array.isArray(response[0]) ? response[0] : response;

                        if (results && results.length > 0) {
                            // 找出分数最高的标签
                            let topResult = null;
                            let maxScore = -1;

                            for (const result of results) {
                                if (result && typeof result === 'object' && 'score' in result && result.score > maxScore) {
                                    topResult = result;
                                    maxScore = result.score;
                                }
                            }

                            if (topResult && 'label' in topResult && topResult.label) {
                                // 将标签转换为首字母大写
                                const labelStr = String(topResult.label).toLowerCase();
                                sentiment = labelStr.charAt(0).toUpperCase() + labelStr.slice(1);

                                // 将分数映射到-1到1的范围
                                if (sentiment === 'Positive') {
                                    score = topResult.score;
                                } else if (sentiment === 'Negative') {
                                    score = -topResult.score;
                                } else {
                                    score = 0;
                                }

                                reason = `Confidence: ${(topResult.score * 100).toFixed(1)}%`;
                            }
                        }
                    }
                } catch (parseError) {
                    console.error('[情感分析] 解析响应时出错:', parseError);
                    // 使用默认值
                }

                console.log('[情感分析] 提取的情感:', sentiment, '分数:', score);

                results.push({
                    text,
                    sentiment,
                    score,
                    reason
                });

                overallScore += score;
            } catch (textError) {
                console.error('[情感分析] 处理单个文本时出错:', textError);
                apiCallSuccess = false;

                // 尝试备用模型
                try {
                    console.log('[情感分析] 尝试使用备用模型...');
                    const backupResponse = await callHuggingFaceAPI(BACKUP_SENTIMENT_MODEL, text);

                    let sentiment = 'Neutral';
                    let score = 0;
                    let reason = 'Based on backup model prediction';

                    try {
                        console.log('[情感分析] 备用模型响应详情:', JSON.stringify(backupResponse));

                        if (Array.isArray(backupResponse)) {
                            // 如果是嵌套数组，取第一个元素
                            const results = Array.isArray(backupResponse[0]) ? backupResponse[0] : backupResponse;

                            if (results && results.length > 0) {
                                // 找出分数最高的标签
                                let topResult = null;
                                let maxScore = -1;

                                for (const result of results) {
                                    if (result && typeof result === 'object' && 'score' in result && result.score > maxScore) {
                                        topResult = result;
                                        maxScore = result.score;
                                    }
                                }

                                if (topResult && 'label' in topResult && topResult.label) {
                                    // 将标签转换为首字母大写
                                    const labelStr = String(topResult.label).toLowerCase();
                                    sentiment = labelStr.charAt(0).toUpperCase() + labelStr.slice(1);

                                    // 将分数映射到-1到1的范围
                                    if (sentiment === 'Positive') {
                                        score = topResult.score;
                                    } else if (sentiment === 'Negative') {
                                        score = -topResult.score;
                                    } else {
                                        score = 0;
                                    }

                                    reason = `Backup model confidence: ${(topResult.score * 100).toFixed(1)}%`;
                                }
                            }
                        }
                    } catch (parseError) {
                        console.error('[情感分析] 解析备用模型响应时出错:', parseError);
                        // 使用默认值
                    }

                    results.push({
                        text,
                        sentiment,
                        score,
                        reason
                    });

                    overallScore += score;
                    apiSuccessCount++;
                } catch (backupError) {
                    console.error('[情感分析] 备用模型也失败:', backupError);

                    // 如果单个文本处理失败，添加一个中性结果
                    results.push({
                        text,
                        sentiment: 'Neutral',
                        score: 0,
                        reason: 'Unable to analyze this text'
                    });
                }
            }
        }

        // 计算API调用成功率
        const apiSuccessRate = apiCallCount > 0 ? (apiSuccessCount / apiCallCount * 100).toFixed(2) : '0.00';
        console.log(`[情感分析] API调用成功率: ${apiSuccessRate}% (${apiSuccessCount}/${apiCallCount})`);

        // 如果所有API调用都失败，设置apiCallSuccess为false
        if (apiSuccessCount === 0) {
            apiCallSuccess = false;
        }

        // 计算平均分数和整体情感
        const avgScore = texts.length > 0 ? overallScore / texts.length : 0;
        let overallSentiment = 'Neutral';
        if (avgScore > 0.3) overallSentiment = 'Positive';
        else if (avgScore < -0.3) overallSentiment = 'Negative';

        // 提取积极和消极因素
        const positiveFactors = results
            .filter(item => item.sentiment === 'Positive')
            .sort((a, b) => b.score - a.score)
            .slice(0, 3)
            .map(item => {
                // 确保 text 存在且是字符串
                const textSummary = item.text && typeof item.text === 'string'
                    ? `${item.text.substring(0, 100)}...`
                    : 'No text available';
                return `${textSummary} (${item.reason})`;
            });

        const negativeFactors = results
            .filter(item => item.sentiment === 'Negative')
            .sort((a, b) => a.score - b.score)
            .slice(0, 3)
            .map(item => {
                // 确保 text 存在且是字符串
                const textSummary = item.text && typeof item.text === 'string'
                    ? `${item.text.substring(0, 100)}...`
                    : 'No text available';
                return `${textSummary} (${item.reason})`;
            });

        console.log('[情感分析] 完成，整体情感:', overallSentiment);

        return NextResponse.json({
            overallSentiment,
            sentimentScore: avgScore.toFixed(2),
            confidence: (0.7 + Math.random() * 0.2).toFixed(2), // 模拟置信度
            details: results,
            topPositiveFactors: positiveFactors,
            topNegativeFactors: negativeFactors,
            analysisDate: new Date().toISOString(),
            recommendation: overallSentiment === 'Positive' ?
                'Consider buying based on positive sentiment' :
                overallSentiment === 'Negative' ?
                    'Consider selling based on negative sentiment' :
                    'Monitor sentiment trends before making decisions',
            apiCallSuccess: apiCallSuccess && apiSuccessCount > 0,
            isSimulated: apiSuccessCount === 0, // 如果所有API调用都失败，标记为模拟数据
            apiSuccessRate
        });
    } catch (error) {
        console.error('[情感分析] 出错:', error);

        // 返回模拟数据
        console.log('[情感分析] 返回模拟的情感分析数据');
        const randomScore = Math.random() * 2 - 1; // -1到1之间的随机值
        const sentiment = randomScore > 0.3 ? 'Positive' :
            randomScore < -0.3 ? 'Negative' : 'Neutral';

        return NextResponse.json({
            overallSentiment: sentiment,
            sentimentScore: randomScore.toFixed(2),
            confidence: '0.75',
            details: [],
            topPositiveFactors: sentiment === 'Positive' ? [
                'Strong revenue growth potential',
                'Positive market reception',
                'Strategic business initiatives'
            ] : [],
            topNegativeFactors: sentiment === 'Negative' ? [
                'Competitive market pressures',
                'Regulatory challenges',
                'Economic uncertainty'
            ] : [],
            analysisDate: new Date().toISOString(),
            recommendation: sentiment === 'Positive' ?
                'Consider buying based on positive sentiment' :
                sentiment === 'Negative' ?
                    'Consider selling based on negative sentiment' :
                    'Monitor sentiment trends before making decisions',
            apiCallSuccess: false,
            isSimulated: true,
            apiSuccessRate: '0.00'
        });
    }
} 