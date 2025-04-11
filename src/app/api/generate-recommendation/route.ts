import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import axios from 'axios';
import { getVIXData, getMarketTrend } from '@/services/yahooFinanceService';

// Initialize the OpenAI client
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

// Alpha Vantage API Key
const ALPHA_VANTAGE_API_KEY = process.env.ALPHA_VANTAGE_API_KEY || 'demo';

// Access to market data (including VIX, market trends, etc.)
async function fetchMarketData() {
    try {
        // First try to use Yahoo Finance API to get VIX data
        console.log('Getting VIX data through Yahoo Finance API...');
        const vixData = await getVIXData();

        // Get market trend data
        console.log('Getting market trend data...');
        const trendData = await getMarketTrend(10); // Use 10-day period to determine trend

        return {
            volatility: vixData.volatility,
            marketTrend: trendData.marketTrend,
            date: vixData.date,
            source: 'yahoo' // Mark the data source
        };

    } catch (error) {
        console.error('Failed to get market data from Yahoo Finance:', error);

        // If Yahoo Finance fails, try using Alpha Vantage as a backup
        try {
            console.log('Trying to use Alpha Vantage API...');

            // Get VIX data from Alpha Vantage
            const vixResponse = await axios.get(
                `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=VIX&apikey=${ALPHA_VANTAGE_API_KEY}`
            );

            // Parse VIX data
            const timeSeries = vixResponse.data['Time Series (Daily)'];
            if (!timeSeries) {
                throw new Error('Failed to get VIX data');
            }

            // Get the latest VIX closing price
            const latestDate = Object.keys(timeSeries)[0];
            const vixValue = parseFloat(timeSeries[latestDate]['4. close']);
            const isHighVolatility = vixValue > 25; // VIX > 25 is usually considered high volatility

            // Try to get the market index (S&P 500) to determine market trend
            const spyResponse = await axios.get(
                `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&apikey=${ALPHA_VANTAGE_API_KEY}`
            );

            // Parse S&P 500 data
            const spyTimeSeries = spyResponse.data['Time Series (Daily)'];
            let marketTrend = 'Neutral';

            if (spyTimeSeries) {
                const dates = Object.keys(spyTimeSeries).sort((a, b) => new Date(b).getTime() - new Date(a).getTime());

                if (dates.length >= 10) {
                    const latest = parseFloat(spyTimeSeries[dates[0]]['4. close']);
                    const tenDaysAgo = parseFloat(spyTimeSeries[dates[9]]['4. close']);

                    // Simple market trend determination (10-day涨跌幅)
                    const changePercent = ((latest - tenDaysAgo) / tenDaysAgo) * 100;

                    if (changePercent > 3) {
                        marketTrend = 'Bullish';
                    } else if (changePercent < -3) {
                        marketTrend = 'Bearish';
                    }
                }
            }

            return {
                volatility: {
                    vix: vixValue,
                    isHigh: isHighVolatility
                },
                marketTrend: marketTrend,
                date: latestDate,
                source: 'alphavantage'
            };
        } catch (alphaVantageError) {
            console.error('Alpha Vantage API also failed:', alphaVantageError);
            // Both methods failed, return default values
            return {
                volatility: {
                    vix: 15,  // Default medium volatility
                    isHigh: false
                },
                marketTrend: 'Neutral',
                date: new Date().toISOString().split('T')[0],
                source: 'default' // Mark as default data
            };
        }
    }
}

// Process requests
export async function POST(req: NextRequest) {
    try {
        // Parse request data
        const requestData = await req.json();
        const { stockSymbol, sentimentData, financialData, strategyData } = requestData;

        // Get market data
        const marketData = await fetchMarketData();
        console.log("Market data obtained:", marketData);

        // Build OpenAI Prompt
        const prompt = buildPrompt(stockSymbol, sentimentData, financialData, strategyData, marketData);

        console.log("Sending prompt to OpenAI:", prompt);

        // Call OpenAI API
        const completion = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
                {
                    role: "system",
                    content: "You are a professional investment advisor with expertise in financial analysis, stock market trends, and investment strategies. You are tasked with providing investment recommendations based on various data points."
                },
                {
                    role: "user",
                    content: prompt
                }
            ],
            temperature: 0, // Set to 0 to ensure consistency
        });

        // Parse OpenAI response
        const responseContent = completion.choices[0].message.content || '';
        console.log("OpenAI response:", responseContent);

        // Try to extract investment score and recommendation reasoning
        let score = 0;
        let reasoning = '';
        let featureImportance = [];

        try {
            // Find the score (format: Score: X.XX)
            const scoreMatch = responseContent.match(/Score:\s*(-?\d+\.?\d*)/i);
            if (scoreMatch && scoreMatch[1]) {
                score = parseFloat(scoreMatch[1]);
                // Ensure the score is within the range of -1 to 1
                score = Math.max(-1, Math.min(1, score));
            }

            // Extract recommendation reasoning - only keep the part after "Reasoning:" including all paragraphs
            const reasoningMatch = responseContent.match(/Reasoning:\s*([\s\S]*?)(?=Feature Weights:|$)/i);
            if (reasoningMatch && reasoningMatch[1]) {
                reasoning = reasoningMatch[1].trim();
            } else {
                // If there is no clear "Reasoning:" section, use the entire response as reasoning
                reasoning = responseContent.trim();
            }

            // Extract feature weights
            const featureWeightsSection = responseContent.match(/Feature Weights:([\s\S]*?)(?=$)/i);
            if (featureWeightsSection && featureWeightsSection[1]) {
                const weightsText = featureWeightsSection[1].trim();
                const weightPattern = /- ([^:]+):\s*(\d+\.?\d*)/g;
                let match;

                while ((match = weightPattern.exec(weightsText)) !== null) {
                    featureImportance.push({
                        name: match[1].trim(),
                        importance: parseFloat(match[2])
                    });
                }
            }
        } catch (parseError) {
            console.error("Error parsing OpenAI response:", parseError);
            reasoning = responseContent;
        }

        // If feature weights are not successfully extracted, use the alternative calculation method
        if (featureImportance.length === 0) {
            featureImportance = generateFeatureImportance(sentimentData, financialData, strategyData, marketData);
        }

        // Construct return data
        const result = {
            score,
            reasoning,
            featureImportance,
            marketData,
            rawResponse: responseContent,
        };

        return NextResponse.json(result);
    } catch (error) {
        console.error("Error generating recommendation:", error);
        return NextResponse.json(
            { error: "Failed to generate recommendation" },
            { status: 500 }
        );
    }
}

// Build the prompt sent to OpenAI
function buildPrompt(
    stockSymbol: string,
    sentimentData: any,
    financialData: any,
    strategyData: any,
    marketData: any
) {
    return `
Analyze the following data for ${stockSymbol} and provide an investment recommendation:

1. Sentiment Analysis:
   - Overall Sentiment: ${sentimentData.overallSentiment}
   - Sentiment Score: ${sentimentData.sentimentScore}

2. Financial Metrics:
${Object.entries(financialData)
            .map(([key, value]) => `   - ${key}: ${value}`)
            .join('\n')}

3. Strategy Analysis:
   - Strategy Type: ${strategyData.type}
   - Total Return: ${strategyData.totalReturn}%
   - Strategy Recommendation: ${strategyData.recommendation}

4. Current Market Conditions (as of ${marketData.date}):
   - Market Volatility (VIX): ${marketData.volatility.vix.toFixed(2)} (${marketData.volatility.isHigh ? 'High' : 'Normal'})
   - Market Trend: ${marketData.marketTrend}

Weight Assignment Rules:
- You should calculate weights for the following three core factors ONLY:
  1. Sentiment Analysis
  2. Financial Health
  3. Strategy Performance
- Consider market conditions when assigning weights, but DO NOT include "Market Conditions" as a separate weighted factor
- If market volatility is high (VIX > 25), increase the weight of Strategy Performance to at least 0.35
- If the stock has strong financials but poor sentiment, prioritize Financial Health (weight > 0.40)
- If the stock has poor financials but strong strategy performance, give Strategy Performance more weight (> 0.35)
- In bullish markets, increase weight of Sentiment Analysis slightly
- In bearish markets, increase weight of Financial Health slightly
- Always ensure all weights sum to exactly 1.0 (or 100%)

Based on this data, calculate an investment score between -1 (strong sell) and 1 (strong buy).

Explain how market conditions influenced your weight assignment in your reasoning, but DO NOT explicitly list the exact weight values in your reasoning. Instead, describe the relative importance you placed on each factor (e.g., "Financial Health was given higher priority than Sentiment Analysis due to strong metrics").

Your response should be structured as follows:
Score: [number between -1 and 1]
Reasoning: [detailed explanation of your recommendation, including how market conditions influenced weight assignment, but WITHOUT listing explicit weight values]
Feature Weights:
- Sentiment Analysis: [weight as decimal between 0 and 1]
- Financial Health: [weight as decimal between 0 and 1]
- Strategy Performance: [weight as decimal between 0 and 1]
`;
}

// Generate feature importance data based on rules (alternative method, used when GPT does not provide effective weights)
function generateFeatureImportance(sentimentData: any, financialData: any, strategyData: any, marketData: any) {
    // Base weights
    let sentimentWeight = 0.30;
    let financialWeight = 0.35;
    let strategyWeight = 0.35;

    // Adjust strategy weight based on market volatility
    if (marketData.volatility.isHigh) {
        // When high volatility, increase strategy weight
        strategyWeight = 0.40;
        // Reduce other weights to keep the total at 1
        sentimentWeight = 0.25;
        financialWeight = 0.35;
    }

    // Adjust weights based on market trend
    if (marketData.marketTrend === 'Bullish') {
        // In bullish markets, increase sentiment analysis weight
        sentimentWeight += 0.05;
        financialWeight -= 0.05;
    } else if (marketData.marketTrend === 'Bearish') {
        // 熊市增加财务健康权重
        financialWeight += 0.05;
        sentimentWeight -= 0.05;
    }

    // Ensure the total is 1
    const total = sentimentWeight + financialWeight + strategyWeight;
    if (total !== 1) {
        const adjustmentFactor = 1 / total;
        sentimentWeight *= adjustmentFactor;
        financialWeight *= adjustmentFactor;
        strategyWeight *= adjustmentFactor;
    }

    return [
        { name: 'Sentiment Analysis', importance: sentimentWeight },
        { name: 'Financial Health', importance: financialWeight },
        { name: 'Strategy Performance', importance: strategyWeight }
    ];
} 