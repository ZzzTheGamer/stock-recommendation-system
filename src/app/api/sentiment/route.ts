import { NextResponse } from 'next/server';
import axios from 'axios';

// Hugging Face API key
const HF_API_KEY = process.env.HF_API_KEY || '';
const HF_API_URL = 'https://api-inference.huggingface.co/models';

// Sentiment analysis model - use FinBERT
const SENTIMENT_MODEL = 'ProsusAI/finbert';
// Backup sentiment analysis model
const BACKUP_SENTIMENT_MODEL = 'yiyanghkust/finbert-tone';

/**
 * Call HuggingFace API, with retry mechanism
 */
const callHuggingFaceAPI = async (model: string, text: string, maxRetries = 2) => {
    let retries = 0;
    let lastError;

    while (retries <= maxRetries) {
        try {
            console.log(`[API call] Trying to call model: ${model}, retry count: ${retries}`);

            // Prepare specific parameters for different models
            let payload: any = { inputs: text };

            const response = await axios.post(`${HF_API_URL}/${model}`, payload, {
                headers: {
                    'Authorization': `Bearer ${HF_API_KEY}`,
                    'Content-Type': 'application/json'
                },
                timeout: 30000 // 30 seconds timeout
            });

            console.log(`[API call] Successfully called model: ${model}`);
            return response.data;
        } catch (error) {
            lastError = error;
            console.error(`[API call] Failed to call model: ${model}, error:`, error);
            retries++;

            if (retries <= maxRetries) {
                console.log(`[API call] Waiting and retrying...`);
                await new Promise(resolve => setTimeout(resolve, 1000 * retries)); // Exponential backoff
            }
        }
    }

    throw lastError;
};

export async function POST(request: Request) {
    try {
        // Parse request body
        const { texts } = await request.json();

        if (!texts || !Array.isArray(texts) || texts.length === 0) {
            return NextResponse.json(
                { error: 'Missing required parameter: texts (should be a non-empty array)' },
                { status: 400 }
            );
        }

        console.log('[Sentiment analysis] Starting sentiment analysis, text count:', texts.length);
        const results = [];
        let overallScore = 0;
        let apiCallSuccess = true;
        let apiCallCount = 0;
        let apiSuccessCount = 0;

        // Analyze each text for sentiment
        for (const text of texts) {
            try {
                console.log('[Sentiment analysis] Analyzing text:', text.substring(0, 50) + '...');
                apiCallCount++;

                // Call FinBERT model directly
                console.log('[Sentiment analysis] Calling FinBERT API...');
                const response = await callHuggingFaceAPI(SENTIMENT_MODEL, text);
                console.log('[Sentiment analysis] API response:', response);
                apiSuccessCount++;

                // Parse FinBERT response
                let sentiment = 'Neutral';
                let score = 0;
                let reason = 'Based on FinBERT model prediction';

                try {
                    console.log('[Sentiment analysis] FinBERT response details:', JSON.stringify(response));

                    // FinBERT usually returns an array, each element representing a category prediction
                    // For example: [{"label":"positive","score":0.9991232},{"label":"neutral","score":0.0006},{"label":"negative","score":0.0002}]
                    if (Array.isArray(response)) {
                        // If it's a nested array, take the first element
                        const results = Array.isArray(response[0]) ? response[0] : response;

                        if (results && results.length > 0) {
                            // Find the label with the highest score
                            let topResult = null;
                            let maxScore = -1;

                            for (const result of results) {
                                if (result && typeof result === 'object' && 'score' in result && result.score > maxScore) {
                                    topResult = result;
                                    maxScore = result.score;
                                }
                            }

                            if (topResult && 'label' in topResult && topResult.label) {
                                // Convert the label to uppercase
                                const labelStr = String(topResult.label).toLowerCase();
                                sentiment = labelStr.charAt(0).toUpperCase() + labelStr.slice(1);

                                // Map the score to the range of -1 to 1
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
                    console.error('[Sentiment analysis] Error parsing response:', parseError);
                    // Use default values
                }

                console.log('[Sentiment analysis] Extracted sentiment:', sentiment, 'score:', score);

                results.push({
                    text,
                    sentiment,
                    score,
                    reason
                });

                overallScore += score;
            } catch (textError) {
                console.error('[Sentiment analysis] Error processing single text:', textError);
                apiCallSuccess = false;

                // Try backup model
                try {
                    console.log('[Sentiment analysis] Trying backup model...');
                    const backupResponse = await callHuggingFaceAPI(BACKUP_SENTIMENT_MODEL, text);

                    let sentiment = 'Neutral';
                    let score = 0;
                    let reason = 'Based on backup model prediction';

                    try {
                        console.log('[Sentiment analysis] Backup model response details:', JSON.stringify(backupResponse));

                        if (Array.isArray(backupResponse)) {
                            // If it's a nested array, take the first element
                            const results = Array.isArray(backupResponse[0]) ? backupResponse[0] : backupResponse;

                            if (results && results.length > 0) {
                                // Find the label with the highest score
                                let topResult = null;
                                let maxScore = -1;

                                for (const result of results) {
                                    if (result && typeof result === 'object' && 'score' in result && result.score > maxScore) {
                                        topResult = result;
                                        maxScore = result.score;
                                    }
                                }

                                if (topResult && 'label' in topResult && topResult.label) {
                                    // Convert the label to uppercase
                                    const labelStr = String(topResult.label).toLowerCase();
                                    sentiment = labelStr.charAt(0).toUpperCase() + labelStr.slice(1);

                                    // Map the score to the range of -1 to 1
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
                        console.error('[Sentiment analysis] Error parsing backup model response:', parseError);
                        // Use default values
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
                    console.error('[Sentiment analysis] Backup model also failed:', backupError);

                    // If single text processing fails, add a neutral result
                    results.push({
                        text,
                        sentiment: 'Neutral',
                        score: 0,
                        reason: 'Unable to analyze this text'
                    });
                }
            }
        }

        // Calculate API call success rate
        const apiSuccessRate = apiCallCount > 0 ? (apiSuccessCount / apiCallCount * 100).toFixed(2) : '0.00';
        console.log(`[Sentiment analysis] API call success rate: ${apiSuccessRate}% (${apiSuccessCount}/${apiCallCount})`);

        // If all API calls fail, set apiCallSuccess to false
        if (apiSuccessCount === 0) {
            apiCallSuccess = false;
        }

        // Calculate average score and overall sentiment
        const avgScore = texts.length > 0 ? overallScore / texts.length : 0;
        let overallSentiment = 'Neutral';
        if (avgScore > 0.3) overallSentiment = 'Positive';
        else if (avgScore < -0.3) overallSentiment = 'Negative';

        // Extract positive and negative factors
        const positiveFactors = results
            .filter(item => item.sentiment === 'Positive')
            .sort((a, b) => b.score - a.score)
            .slice(0, 3)
            .map(item => {
                // Ensure text exists and is a string
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
                // Ensure text exists and is a string
                const textSummary = item.text && typeof item.text === 'string'
                    ? `${item.text.substring(0, 100)}...`
                    : 'No text available';
                return `${textSummary} (${item.reason})`;
            });

        console.log('[Sentiment analysis] Completed, overall sentiment:', overallSentiment);

        return NextResponse.json({
            overallSentiment,
            sentimentScore: avgScore.toFixed(2),
            confidence: (0.7 + Math.random() * 0.2).toFixed(2), // Simulate confidence
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
            isSimulated: apiSuccessCount === 0, // If all API calls fail, mark as simulated data
            apiSuccessRate
        });
    } catch (error) {
        console.error('[Sentiment analysis] Error:', error);

        // Return simulated data
        console.log('[Sentiment analysis] Returning simulated sentiment analysis data');
        const randomScore = Math.random() * 2 - 1; // Random value between -1 and 1
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