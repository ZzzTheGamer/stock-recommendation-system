import axios from 'axios';

const HF_API_KEY = process.env.NEXT_PUBLIC_HUGGINGFACE_API_KEY || '';
const HF_API_URL = 'https://api-inference.huggingface.co/models';

// Sentiment analysis model - using FinBERT
const SENTIMENT_MODEL = 'ProsusAI/finbert';
// Backup sentiment analysis model
const BACKUP_SENTIMENT_MODEL = 'yiyanghkust/finbert-tone';
// Financial analysis model
const FALCON_MODEL = 'tiiuae/falcon-7b-instruct';
const MISTRAL_MODEL = 'mistralai/Mistral-7B-Instruct-v0.1';

/**
 * Call HuggingFace API with retry mechanism
 */
const callHuggingFaceAPI = async (model: string, text: string, maxRetries = 2) => {
    let retries = 0;
    let lastError;

    while (retries <= maxRetries) {
        try {
            console.log(`[API Call] Attempting to call model: ${model}, retry count: ${retries}`);

            // Prepare specific parameters for different models
            let payload: any = { inputs: text };
            let timeout = 60000; // Default 60 second timeout

            // If it's a T5 model, use specific parameters
            if (model.includes('t5')) {
                payload = {
                    inputs: text,
                    parameters: {
                        max_new_tokens: 1024,
                        temperature: 0.7,
                        top_p: 0.9,
                        do_sample: true
                    }
                };
            }
            // If it's a causal language model (Fino, Llama, GPT, etc.), use specific parameters
            else if (model.includes('Fino') || model.includes('llama') || model.includes('gpt')) {
                payload = {
                    inputs: text,
                    parameters: {
                        max_new_tokens: 1024,
                        temperature: 0.7,
                        top_p: 0.9,
                        do_sample: true,
                        return_full_text: false
                    }
                };

                // For Fino models, increase timeout as it's a larger model
                if (model.includes('Fino')) {
                    timeout = 120000; // 2 minutes
                }
            }

            const response = await axios.post(
                `${HF_API_URL}/${model}`,
                payload,
                {
                    headers: {
                        'Authorization': `Bearer ${HF_API_KEY}`,
                        'Content-Type': 'application/json'
                    },
                    timeout: timeout
                }
            );

            console.log(`[API Call] Success, status code: ${response.status}`);
            return response.data;
        } catch (error: any) {
            retries++;
            lastError = error;

            console.error(`[API Call] Failed, retry ${retries}/${maxRetries}:`, error.message);

            if (retries <= maxRetries) {
                // Exponential backoff strategy
                const delay = Math.pow(2, retries) * 1000 + Math.random() * 1000;
                console.log(`[API Call] Waiting ${delay}ms before retrying...`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    // All retries failed
    throw lastError;
};

/**
 * Use FinBERT model for sentiment analysis
 */
export const analyzeSentimentWithFinGPT = async (texts: string[]) => {
    try {
        console.log('[Sentiment Analysis] Starting sentiment analysis, text count:', texts.length);

        // Call API route
        console.log('[Sentiment Analysis] Calling API route...');
        const response = await axios.post('/api/sentiment', {
            texts
        });

        console.log('[Sentiment Analysis] API route response successful');
        return response.data;
    } catch (error) {
        console.error('[Sentiment Analysis] Error:', error);

        // Return simulated data
        console.log('[Sentiment Analysis] Returning simulated sentiment analysis data');
        const randomScore = Math.random() * 2 - 1; // Random value between -1 and 1
        const sentiment = randomScore > 0.3 ? 'Positive' :
            randomScore < -0.3 ? 'Negative' : 'Neutral';

        return {
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
        };
    }
};

/**
 * Use Mistral model to generate financial analysis report
 */
export const generateFinancialAnalysisWithMistral = async (
    symbol: string,
    companyName: string,
    industry: string,
    financialData: any
) => {
    try {
        console.log(`[Research Analysis] Starting to generate financial analysis report for ${symbol} (using Mistral-7B-Instruct)`);

        // Improved prompt to get better analysis results
        const prompt = `
Analyze the financial condition of ${companyName} (${symbol}) in the ${industry} industry.

Financial data:
- Revenue Growth: ${financialData.revenueGrowth}%
- Net Income Growth: ${financialData.netIncomeGrowth}%
- Current Ratio: ${financialData.currentRatio}
- Debt to Equity: ${financialData.debtToEquity}
- Return on Equity: ${financialData.returnOnEquity}%

Please provide a DETAILED ANALYSIS with the following sections:
1. FINANCIAL SUMMARY (provide at least 3-4 sentences analyzing the company's overall financial health)
2. STRENGTHS (list at least 2-3 financial strengths)
3. WEAKNESSES (list at least 2-3 financial weaknesses)
4. METRICS ANALYSIS (analyze what the key metrics indicate)
5. INVESTMENT RECOMMENDATION (explicitly state "Buy", "Hold", or "Sell" as your recommendation and explain why)

IMPORTANT: Provide detailed textual analysis, not just numbers. Ensure your final recommendation clearly states either "Buy", "Hold", or "Sell".`;

        // Use Mistral model directly
        const result = await callHuggingFaceAPI(MISTRAL_MODEL, prompt);
        console.log('[Research Analysis] Mistral model response successful');

        // Parse output
        let outputText = '';
        if (Array.isArray(result)) {
            outputText = result[0]?.generated_text || '';
        } else {
            outputText = result.generated_text || '';
        }

        // Process output
        if (outputText) {
            try {
                const processedResult = processFinancialAnalysisOutput(outputText, symbol, companyName, financialData);
                return {
                    ...processedResult,
                    modelType: 'mistral',
                    modelSuccess: true
                };
            } catch (error) {
                console.error('[Research Analysis] Error processing Mistral output:', error);
                // If processing fails, return original text as summary
                return {
                    summary: outputText.substring(0, 1000),  // Take first 1000 characters as summary
                    strengths: '',
                    weaknesses: '',
                    metricsAnalysis: '',
                    recommendation: outputText.toLowerCase().includes('buy') ? 'Buy' :
                        outputText.toLowerCase().includes('sell') ? 'Sell' : 'Hold',
                    rating: outputText.toLowerCase().includes('buy') ? 'Buy' :
                        outputText.toLowerCase().includes('sell') ? 'Sell' : 'Hold',
                    modelType: 'mistral',
                    modelSuccess: true,
                    apiCallSuccess: true,
                    isSimulated: false,
                    keyMetrics: financialData,
                    analysisDate: new Date().toISOString()
                };
            }
        } else {
            throw new Error('Mistral model did not return valid output');
        }

    } catch (error) {
        console.error('[Research Analysis] Mistral analysis error:', error);
        console.log('[Research Analysis] Using fallback simulated data');

        // Return simulated data
        return {
            summary: `${companyName} (${symbol}) shows mixed financial performance with some areas of strength and concern.`,
            strengths: '1. Stable revenue growth\n2. Healthy profit margins\n3. Strong market position',
            weaknesses: '1. Increasing debt levels\n2. Cash flow challenges\n3. Competitive pressure',
            metricsAnalysis: 'Key financial metrics indicate a stable but challenging financial position.',
            recommendation: 'Hold',
            rating: 'Hold',
            modelType: 'mistral',
            modelSuccess: false,
            apiCallSuccess: false,
            isSimulated: true,
            keyMetrics: financialData,
            analysisDate: new Date().toISOString()
        };
    }
};

// For backward compatibility, keep original function name but call the new function
export const generateFinancialAnalysisWithFinGPT = generateFinancialAnalysisWithMistral;

/**
 * Process financial analysis model output
 */
const processFinancialAnalysisOutput = (
    outputText: string,
    symbol: string,
    companyName: string,
    financialData: any
) => {
    // Log original output for debugging
    console.log('[Research Analysis] Model original output:', outputText);

    // Preprocess output text - remove prompt part and overall formatting

    // If output text contains input prompt part, remove it
    const promptEndIndex = outputText.indexOf("IMPORTANT: Provide detailed textual analysis");
    if (promptEndIndex > 0) {
        // Find the next line break after the prompt ends
        const nextLineIndex = outputText.indexOf("\n", promptEndIndex);
        if (nextLineIndex > 0) {
            outputText = outputText.substring(nextLineIndex + 1).trim();
        }
    }

    // Remove all backtick marks
    outputText = outputText.replace(/```/g, '');

    // Define regular expressions for various possible heading formats
    const financialSummaryTitles = [
        /FINANCIAL\s+SUMMARY\s*:/i,
        /FINANCIAL\s+SUMMARY\s*[^:]/i,
        /(\d+\.?\s*)?FINANCIAL\s+SUMMARY\s*:?/i,
        /\d+\.\s*Financial\s+Summary/i,  // Match "1. Financial Summary" format
        /Financial\s+Summary/i           // Match simple "Financial Summary"
    ];

    const strengthsTitles = [
        /STRENGTHS\s*:/i,
        /STRENGTHS\s*[^:]/i,
        /(\d+\.?\s*)?STRENGTHS\s*:?/i,
        /\d+\.\s*Strengths/i,            // Match "2. Strengths" format
        /Strengths/i                      // Match simple "Strengths"
    ];

    const weaknessesTitles = [
        /WEAKNESSES\s*:/i,
        /WEAKNESSES\s*[^:]/i,
        /(\d+\.?\s*)?WEAKNESSES\s*:?/i,
        /\d+\.\s*Weaknesses/i,           // Match "3. Weaknesses" format
        /Weaknesses/i                     // Match simple "Weaknesses"
    ];

    const metricsAnalysisTitles = [
        /METRICS\s+ANALYSIS\s*:/i,
        /METRICS\s+ANALYSIS\s*[^:]/i,
        /(\d+\.?\s*)?METRICS\s+ANALYSIS\s*:?/i,
        /\d+\.\s*Metrics\s+Analysis/i,    // Match "4. Metrics Analysis" format
        /Metrics\s+Analysis/i             // Match simple "Metrics Analysis"
    ];

    const investmentRecommendationTitles = [
        /INVESTMENT\s+RECOMMENDATION\s*:/i,
        /INVESTMENT\s+RECOMMENDATION\s*[^:]/i,
        /(\d+\.?\s*)?INVESTMENT\s+RECOMMENDATION\s*:?/i,
        /\d+\.\s*Investment\s+Recommendation/i, // Match "5. Investment Recommendation" format
        /Investment\s+Recommendation/i,         // Match simple "Investment Recommendation"
        /\d+\.\s*Recommendation/i,              // Match "5. Recommendation" format
        /Recommendation:/i                      // Match "Recommendation:" format
    ];

    // Find position of each section in the text
    const findSectionPosition = (titles: RegExp[], text: string) => {
        for (const pattern of titles) {
            const match = text.match(pattern);
            if (match && match.index !== undefined) {
                return {
                    start: match.index,
                    end: match.index + match[0].length
                };
            }
        }
        return null;
    };

    // Extract each section content
    const financialPosition = findSectionPosition(financialSummaryTitles, outputText);
    const strengthsPosition = findSectionPosition(strengthsTitles, outputText);
    const weaknessesPosition = findSectionPosition(weaknessesTitles, outputText);
    const metricsPosition = findSectionPosition(metricsAnalysisTitles, outputText);
    const recommendationPosition = findSectionPosition(investmentRecommendationTitles, outputText);

    // Debug log - helps determine if sections were correctly identified
    console.log('[Parsing] Found section positions:', {
        financial: financialPosition ? `${financialPosition.start}:${financialPosition.end}` : 'not found',
        strengths: strengthsPosition ? `${strengthsPosition.start}:${strengthsPosition.end}` : 'not found',
        weaknesses: weaknessesPosition ? `${weaknessesPosition.start}:${weaknessesPosition.end}` : 'not found',
        metrics: metricsPosition ? `${metricsPosition.start}:${metricsPosition.end}` : 'not found',
        recommendation: recommendationPosition ? `${recommendationPosition.start}:${recommendationPosition.end}` : 'not found'
    });

    // Initialize section contents
    let summary = '', strengths = '', weaknesses = '', metricsAnalysis = '', investmentRecommendation = '', recommendation = 'Hold';

    // Extract financial summary
    if (financialPosition && strengthsPosition) {
        summary = outputText.substring(
            financialPosition.end,
            strengthsPosition.start
        ).trim();
    } else if (financialPosition) {
        // If next section not found, extract to the end of text
        summary = outputText.substring(financialPosition.end).trim();
    }

    // Extract strengths
    if (strengthsPosition && weaknessesPosition) {
        strengths = outputText.substring(
            strengthsPosition.end,
            weaknessesPosition.start
        ).trim();
    } else if (strengthsPosition) {
        strengths = outputText.substring(strengthsPosition.end).trim();
    }

    // Extract weaknesses
    if (weaknessesPosition && metricsPosition) {
        weaknesses = outputText.substring(
            weaknessesPosition.end,
            metricsPosition.start
        ).trim();
    } else if (weaknessesPosition) {
        weaknesses = outputText.substring(weaknessesPosition.end).trim();
    }

    // Extract metrics analysis
    if (metricsPosition && recommendationPosition) {
        metricsAnalysis = outputText.substring(
            metricsPosition.end,
            recommendationPosition.start
        ).trim();
    } else if (metricsPosition) {
        // If investment recommendation section not found, metrics analysis extends to the end of text
        metricsAnalysis = outputText.substring(metricsPosition.end).trim();
    }

    // Extract investment recommendation
    if (recommendationPosition) {
        investmentRecommendation = outputText.substring(
            recommendationPosition.end
        ).trim();

        // More intelligently extract recommendation rating
        // Modify regex to capture quoted recommendations and various formats
        const explicitRecommendMatch = investmentRecommendation.match(/(?:we|I)\s+recommend\s+(?:"([A-Za-z]+)"|'([A-Za-z]+)'|\s*([A-Za-z]+))\s+(?:for|on)?/i);
        if (explicitRecommendMatch) {
            // Extract matched groups, possibly any of three capture groups
            const extractedRec = (explicitRecommendMatch[1] || explicitRecommendMatch[2] || explicitRecommendMatch[3] || '').toLowerCase();
            if (extractedRec === 'buy') recommendation = 'Buy';
            else if (extractedRec === 'sell') recommendation = 'Sell';
            else if (extractedRec === 'hold') recommendation = 'Hold';
            console.log(`[Parsing] Found explicit recommendation sentence: "${explicitRecommendMatch[0]}", Extracted recommendation: "${recommendation}"`);
        } else {
            // Try to find other forms of explicit recommendation
            const otherExplicitPatterns = [
                /recommendation\s+is\s+(?:"([A-Za-z]+)"|'([A-Za-z]+)'|\s*([A-Za-z]+))/i,
                /recommend(?:ation|ed)?\s*:\s*(?:"([A-Za-z]+)"|'([A-Za-z]+)'|\s*([A-Za-z]+))/i,
                /(?:our|the)\s+recommendation\s+(?:is|would\s+be)\s+(?:"([A-Za-z]+)"|'([A-Za-z]+)'|\s*([A-Za-z]+))/i,
                /based\s+on.*(?:we|I)\s+(?:would\s+)recommend\s+(?:"([A-Za-z]+)"|'([A-Za-z]+)'|\s*([A-Za-z]+))/i,
                /investment\s+recommendation.*(?:"([A-Za-z]+)"|'([A-Za-z]+)'|\s*([A-Za-z]+))\s+(?:for|on)/i
            ];

            let foundExplicit = false;
            for (const pattern of otherExplicitPatterns) {
                const match = investmentRecommendation.match(pattern);
                if (match) {
                    // Extract matched groups, possibly any of three capture groups
                    const extractedRec = (match[1] || match[2] || match[3] || '').toLowerCase();
                    if (extractedRec === 'buy') {
                        recommendation = 'Buy';
                        foundExplicit = true;
                    } else if (extractedRec === 'sell') {
                        recommendation = 'Sell';
                        foundExplicit = true;
                    } else if (extractedRec === 'hold') {
                        recommendation = 'Hold';
                        foundExplicit = true;
                    }
                    if (foundExplicit) {
                        console.log(`[Parsing] Found other form of explicit recommendation: "${match[0]}", Extracted recommendation: "${recommendation}"`);
                        break;
                    }
                }
            }

            // Backup method: If no explicit recommendation found, use keyword matching
            if (!foundExplicit) {
                console.log('[Parsing] No explicit recommendation sentence found, using keyword matching method');
                if (investmentRecommendation.toLowerCase().includes('buy') &&
                    !investmentRecommendation.toLowerCase().includes('do not buy') &&
                    !investmentRecommendation.toLowerCase().includes('don\'t buy')) {
                    recommendation = 'Buy';
                } else if (investmentRecommendation.toLowerCase().includes('sell') &&
                    !investmentRecommendation.toLowerCase().includes('do not sell') &&
                    !investmentRecommendation.toLowerCase().includes('don\'t sell')) {
                    recommendation = 'Sell';
                } else if (investmentRecommendation.toLowerCase().includes('hold') ||
                    investmentRecommendation.toLowerCase().includes('neutral')) {
                    recommendation = 'Hold';
                }
            }
        }
    } else {
        // Try backup method to find investment recommendation
        // 1. Find "recommendation" keyword in metrics analysis
        const recommendationKeywordMatch = metricsAnalysis.match(/recommendation\s*:/i);
        if (recommendationKeywordMatch && recommendationKeywordMatch.index !== undefined) {
            investmentRecommendation = metricsAnalysis.substring(recommendationKeywordMatch.index);
            // Update metrics analysis, remove investment recommendation part
            metricsAnalysis = metricsAnalysis.substring(0, recommendationKeywordMatch.index).trim();
        }
    }

    // Modify cleanContent function, enhance cleaning effect
    const cleanContent = (content: string): string => {
        if (!content) return '';

        // Delete instruction text with parentheses, like (provide at least 3-4 sentences...)
        let cleanedContent = content.replace(/\([^)]*\)/g, '').trim();

        // Delete opening and closing backticks(```)
        cleanedContent = cleanedContent.replace(/^```[\s\S]*?```\s*/g, '').trim();
        cleanedContent = cleanedContent.replace(/^```[\s\S]*/g, '').trim();
        cleanedContent = cleanedContent.replace(/[\s\S]*```$/g, '').trim();

        // Delete paragraph end number format (like "2.", "3.")
        cleanedContent = cleanedContent.replace(/\s+\d+\.\s*$/g, '').trim();

        // Delete all backticks
        cleanedContent = cleanedContent.replace(/```/g, '').trim();

        // Delete "Explain this" type prompts
        cleanedContent = cleanedContent.replace(/Explain\s+this/i, '').trim();

        // Delete every line end number mark
        cleanedContent = cleanedContent.replace(/\s+\d+\.$/gm, '').trim();

        // Delete every part start possible number mark
        cleanedContent = cleanedContent.replace(/^\d+\.\s*/gm, '').trim();

        // Ensure list items keep previous dash or point, but delete unnecessary spaces
        cleanedContent = cleanedContent.replace(/^-\s+/gm, '- ').trim();
        cleanedContent = cleanedContent.replace(/^\*\s+/gm, '* ').trim();

        // Process extra line breaks, replace 3 or more consecutive line breaks with 2
        cleanedContent = cleanedContent.replace(/\n{3,}/g, '\n\n');

        // Remove possible number marks at the end (including number marks on a separate line)
        cleanedContent = cleanedContent.replace(/\s*\n+\s*\d+\.\s*$/g, '');

        // Remove HTML tags from the text (if any)
        cleanedContent = cleanedContent.replace(/<[^>]*>/g, '');

        return cleanedContent;
    };

    // Final processing of each section content, clean up formatting issues
    if (summary) summary = cleanContent(summary);
    if (strengths) strengths = cleanContent(strengths);
    if (weaknesses) weaknesses = cleanContent(weaknesses);
    if (metricsAnalysis) metricsAnalysis = cleanContent(metricsAnalysis);
    if (investmentRecommendation) investmentRecommendation = cleanContent(investmentRecommendation);

    // Debug log - show extracted content
    console.log('[Parsing] Extracted content:', {
        summary: summary.substring(0, 50) + (summary.length > 50 ? '...' : ''),
        strengths: strengths.substring(0, 50) + (strengths.length > 50 ? '...' : ''),
        weaknesses: weaknesses.substring(0, 50) + (weaknesses.length > 50 ? '...' : ''),
        metricsAnalysis: metricsAnalysis.substring(0, 50) + (metricsAnalysis.length > 50 ? '...' : ''),
        investmentRecommendation: investmentRecommendation.substring(0, 50) + (investmentRecommendation.length > 50 ? '...' : ''),
        recommendation
    });

    // Investment recommendation final extraction result detailed log
    if (investmentRecommendation) {
        console.log('[Parsing-Detail] Investment recommendation full content:');
        console.log(investmentRecommendation);
        console.log(`[Parsing-Result] Final extracted recommendation: "${recommendation}"`);
    } else {
        console.log('[Parsing-Warning] Investment recommendation section not found');
    }

    // If we've extracted at least the financial summary, consider processing successful
    if (summary) {
        console.log('[Research Analysis] Successfully extracted model analysis results for each section');

        // Format final content
        const formatContent = (content: string): string => {
            if (!content) return '';

            // Process lists in strengths and weaknesses, ensure list items are correctly formatted
            if (content === strengths || content === weaknesses) {
                // If content doesn't have list item formatting (dash or number), try splitting by line and adding dashes
                if (!content.includes('- ') && !content.match(/\d+\.\s+/)) {
                    return content.split(/\n+/)
                        .filter(line => line.trim())
                        .map(line => '- ' + line.trim())
                        .join('\n');
                }
            }

            return content;
        };

        return {
            summary: formatContent(summary),
            strengths: formatContent(strengths),
            weaknesses: formatContent(weaknesses),
            metricsAnalysis: formatContent(metricsAnalysis),
            investmentRecommendation: formatContent(investmentRecommendation),
            recommendation,
            rating: recommendation,
            apiCallSuccess: true,
            isSimulated: false,
            keyMetrics: financialData,
            analysisDate: new Date().toISOString()
        };
    } else {
        // If unable to extract valid content, try fallback extraction method
        console.log('[Research Analysis] Unable to extract valid content, trying fallback extraction method');

        // Look for numbered paragraphs
        const textSections = outputText.split(/\d+\.\s+/).filter(s => s.trim().length > 0);

        if (textSections.length >= 5) {
            // Assume sections are in order: Financial Summary, Strengths, Weaknesses, Metrics Analysis, Investment Recommendation
            summary = cleanContent(textSections[0]);
            strengths = cleanContent(textSections[1]);
            weaknesses = cleanContent(textSections[2]);
            metricsAnalysis = cleanContent(textSections[3]);
            investmentRecommendation = cleanContent(textSections[4]);

            const recText = textSections[4];
            if (recText.toLowerCase().includes('buy')) recommendation = 'Buy';
            else if (recText.toLowerCase().includes('sell')) recommendation = 'Sell';
            else recommendation = 'Hold';

            console.log('[Research Analysis] Fallback method successfully extracted content');
        } else {
            console.log('[Research Analysis] Fallback extraction method also failed, using default content');
            summary = `${companyName} (${symbol}) shows mixed financial performance with some areas of strength and concern.`;
            strengths = '1. Stable revenue growth\n2. Healthy profit margins';
            weaknesses = '1. Increasing debt levels\n2. Cash flow challenges';
            metricsAnalysis = 'Key financial metrics indicate a stable but challenging financial position.';
            investmentRecommendation = 'Based on the analysis, we recommend to Hold this stock. While there are some positive indicators, the risks balance out the potential rewards.';
        }

        // Ensure proper formatting of section headings in the final response
        return {
            summary,
            strengths,
            weaknesses,
            metricsAnalysis,
            investmentRecommendation,
            recommendation,
            rating: recommendation,
            apiCallSuccess: true,
            isSimulated: false,
            keyMetrics: financialData,
            analysisDate: new Date().toISOString()
        };
    }
};

// Add type definitions
interface ExplainabilityData {
    tokens: string[] | null;
    importanceValues: number[] | null;
    attentionMatrix: number[][] | null;
    importanceError: string | null;
}

interface ExplainabilityResponse {
    sentiment: string;
    sentimentScore: number;
    explainability: ExplainabilityData;
    methodInfo?: {
        method: string;
        description: string;
    };
    processingTimes?: {
        total: number;
        sentiment: number;
        attention: number;
        importance: number;
    };
    ignored?: boolean;
    error?: boolean;
    message?: string;
}

// Use request ID to track latest request instead of canceling old requests
let latestRequestId = 0;
// Track request status
let requestInProgress = false;
// Request queue to prevent multiple simultaneous requests
const requestQueue: (() => Promise<any>)[] = [];

// Process next request in queue
const processNextRequest = async () => {
    if (requestQueue.length === 0 || requestInProgress) {
        return;
    }

    requestInProgress = true;
    try {
        await requestQueue[0]();
    } finally {
        requestInProgress = false;
        requestQueue.shift(); // Remove processed request
        if (requestQueue.length > 0) {
            // Process next request in queue
            processNextRequest();
        }
    }
};

export const explainSingleText = async (text: string): Promise<ExplainabilityResponse> => {
    // Generate unique request ID
    const requestId = ++latestRequestId;

    // Create a wrapper function to encapsulate the actual request logic
    const doRequest = async () => {
        try {
            // Ensure text is a string and remove possible undefined prefix
            if (typeof text !== 'string') {
                console.error('[Explainability Analysis] Text format error:', text);
                throw new Error('Text format error, please provide a valid string');
            }

            // Remove possible undefined prefix
            if (text.startsWith('undefined:')) {
                console.log('[Explainability Analysis] Fixing text format, removing undefined: prefix');
                text = text.substring('undefined:'.length).trim();
            }

            // Limit text length - keep consistent with backend
            const MAX_TEXT_LENGTH = 1000;
            if (text.length > MAX_TEXT_LENGTH) {
                console.log(`[Explainability Analysis-${requestId}] Text too long (${text.length} characters), truncating to ${MAX_TEXT_LENGTH} characters`);
                text = text.substring(0, MAX_TEXT_LENGTH);
            }

            console.log(`[Explainability Analysis-${requestId}] Starting single text explanation analysis:`, text.substring(0, 50));

            // Call Flask backend API directly, skip Next.js API route
            console.log(`[Explainability Analysis-${requestId}] Sending request to http://localhost:5000/api/explain-single-text`);

            try {
                const response = await axios.post('http://localhost:5000/api/explain-single-text', {
                    text
                }, {
                    timeout: 300000, // 5 minute timeout
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    }
                });

                // Remove old request check, process all responses
                console.log(`[Explainability Analysis-${requestId}] Flask backend API response successful`);

                // Add detailed response data check and logging
                const responseData = response.data;

                // Print the original structure of the response for debugging
                console.log(`[Explainability Analysis-${requestId}] Original response structure:`, {
                    hasData: !!responseData,
                    hasSentiment: 'sentiment' in responseData,
                    sentimentType: responseData.sentiment ? typeof responseData.sentiment : 'missing',
                    hasSentimentScore: 'sentimentScore' in responseData,
                    hasExplainability: 'explainability' in responseData,
                    explainabilityKeys: responseData.explainability ? Object.keys(responseData.explainability) : 'missing',
                    hasMethodInfo: 'methodInfo' in responseData,
                    hasProcessingTimes: 'processingTimes' in responseData
                });

                // Relax validation of response structure requirements
                if (!responseData || typeof responseData !== 'object') {
                    console.error(`[Explainability Analysis-${requestId}] Response data is not a valid object:`, responseData);
                    throw new Error('Invalid data format returned from backend');
                }

                // Ensure return data has basic structure
                if (!responseData.sentiment) {
                    console.warn(`[Explainability Analysis-${requestId}] Response missing sentiment field, setting default value`);
                    responseData.sentiment = 'Unknown';
                }

                if (typeof responseData.sentimentScore !== 'number') {
                    console.warn(`[Explainability Analysis-${requestId}] Response missing sentimentScore field, setting to 0`);
                    responseData.sentimentScore = 0;
                }

                // Ensure explainability field exists and is an object
                if (!responseData.explainability || typeof responseData.explainability !== 'object') {
                    console.warn(`[Explainability Analysis-${requestId}] Response missing explainability object, creating default structure`);
                    responseData.explainability = {
                        tokens: [],
                        importanceValues: [],
                        attentionMatrix: null,
                        importanceError: "No explainability data returned"
                    };
                } else {
                    // If explainability exists but structure is incomplete, ensure necessary fields exist
                    if (!responseData.explainability.tokens) {
                        console.warn(`[Explainability Analysis-${requestId}] explainability object missing tokens field`);
                        responseData.explainability.tokens = [];
                    }

                    if (!responseData.explainability.importanceValues) {
                        console.warn(`[Explainability Analysis-${requestId}] explainability object missing importanceValues field`);
                        responseData.explainability.importanceValues = [];
                    }

                    if (!responseData.explainability.attentionMatrix) {
                        console.warn(`[Explainability Analysis-${requestId}] explainability object missing attentionMatrix field`);
                        responseData.explainability.attentionMatrix = null;
                    }
                }

                // Log the method used
                if (response.data.methodInfo && response.data.methodInfo.method) {
                    console.log(`[Explainability Analysis-${requestId}] Used ${response.data.methodInfo.method} method`);
                }

                // Log processing time
                if (response.data.processingTimes) {
                    const times = response.data.processingTimes;
                    console.log(`[Explainability Analysis-${requestId}] Total processing time: ${times.total?.toFixed(2)} seconds, Feature importance calculation: ${times.importance?.toFixed(2)} seconds`);
                }

                return response.data;
            } catch (error: any) {
                // Remove old request check, handle all errors

                // Add detailed error logging
                console.error(`[Explainability Analysis-${requestId}] Request error details:`, {
                    message: error.message,
                    status: error.response?.status,
                    statusText: error.response?.statusText,
                    data: error.response?.data,
                    code: error.code
                });

                // Special handling for CORS errors
                if (error.message && (
                    error.message.includes('blocked by CORS policy') ||
                    error.message.includes('Network Error'))
                ) {
                    console.error(`[Explainability Analysis-${requestId}] CORS or network error, may need to refresh page or restart backend service`);
                    throw new Error('CORS or network error: Please check if Flask backend service is running and CORS is configured correctly');
                }

                // Throw error up
                throw error;
            }
        } catch (error: any) {
            // Remove old request check, handle all errors

            console.error(`[Explainability Analysis-${requestId}] Single text explanation error:`, error);
            if (error.code === 'ECONNABORTED') {
                throw new Error('Analysis timeout, feature importance calculation is taking too long. Please try using shorter text or check server status.');
            } else if (error.response?.status === 502) {
                throw new Error('Backend service response timeout, calculating feature importance may require more time');
            } else if (error.message.includes('Network Error')) {
                throw new Error('Network error, please ensure backend service is started and running on port 5000');
            } else if (error.message.includes('CORS') || error.message.includes('Access-Control-Allow-Origin')) {
                throw new Error('CORS cross-origin error: Browser blocked the request, please check Flask backend CORS configuration or try refreshing the page');
            }
            throw new Error('Unable to get explainability analysis data, please ensure backend service is running');
        }
    };

    // Add current request to queue
    return new Promise((resolve, reject) => {
        const queuedRequest = async () => {
            try {
                const result = await doRequest();
                resolve(result);
            } catch (error) {
                reject(error);
            }
        };

        requestQueue.push(queuedRequest);

        // If no request is currently being processed, start processing
        if (!requestInProgress) {
            processNextRequest();
        }
    });
};