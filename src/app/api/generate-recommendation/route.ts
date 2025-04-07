import { NextRequest, NextResponse } from 'next/server';
import OpenAI from 'openai';
import axios from 'axios';

// 初始化OpenAI客户端
const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY
});

// Alpha Vantage API Key
const ALPHA_VANTAGE_API_KEY = process.env.ALPHA_VANTAGE_API_KEY || 'demo';

// 获取市场数据（包括VIX指数、市场趋势等）
async function fetchMarketData() {
    try {
        // 通过Alpha Vantage获取VIX数据
        const vixResponse = await axios.get(
            `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=VIX&apikey=${ALPHA_VANTAGE_API_KEY}`
        );

        // 解析VIX数据
        const timeSeries = vixResponse.data['Time Series (Daily)'];
        if (!timeSeries) {
            throw new Error('无法获取VIX数据');
        }

        // 获取最近的VIX收盘价
        const latestDate = Object.keys(timeSeries)[0];
        const vixValue = parseFloat(timeSeries[latestDate]['4. close']);
        const isHighVolatility = vixValue > 25; // VIX > 25 通常被视为高波动

        // 尝试获取市场指数（S&P 500）判断市场趋势
        const spyResponse = await axios.get(
            `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&apikey=${ALPHA_VANTAGE_API_KEY}`
        );

        // 解析S&P 500数据
        const spyTimeSeries = spyResponse.data['Time Series (Daily)'];
        let marketTrend = 'Neutral';

        if (spyTimeSeries) {
            const dates = Object.keys(spyTimeSeries).sort((a, b) => new Date(b).getTime() - new Date(a).getTime());

            if (dates.length >= 10) {
                const latest = parseFloat(spyTimeSeries[dates[0]]['4. close']);
                const tenDaysAgo = parseFloat(spyTimeSeries[dates[9]]['4. close']);

                // 简单判断市场趋势（10天涨跌幅）
                const changePercent = ((latest - tenDaysAgo) / tenDaysAgo) * 100;

                if (changePercent > 3) {
                    marketTrend = 'Bullish';  // 10天涨幅超过3%视为牛市
                } else if (changePercent < -3) {
                    marketTrend = 'Bearish';  // 10天跌幅超过3%视为熊市
                }
            }
        }

        return {
            volatility: {
                vix: vixValue,
                isHigh: isHighVolatility
            },
            marketTrend: marketTrend,
            date: new Date().toISOString().split('T')[0]
        };
    } catch (error) {
        console.error('获取市场数据失败:', error);

        // 如果Alpha Vantage失败，尝试使用Yahoo Finance API
        try {
            console.log('尝试使用Yahoo Finance API...');
            // 这里可以实现Yahoo Finance的备选方案
            // 但这需要额外的包，简单起见，这里返回一个默认值
            return {
                volatility: {
                    vix: 15,  // 默认中等波动率
                    isHigh: false
                },
                marketTrend: 'Neutral',
                date: new Date().toISOString().split('T')[0],
                source: 'default' // 标记为默认数据
            };
        } catch (yahooError) {
            console.error('获取Yahoo Finance数据也失败:', yahooError);
            // 两种方法都失败，返回默认值
            return {
                volatility: {
                    vix: 15,
                    isHigh: false
                },
                marketTrend: 'Neutral',
                date: new Date().toISOString().split('T')[0],
                source: 'default'
            };
        }
    }
}

// 处理请求
export async function POST(req: NextRequest) {
    try {
        // 解析请求数据
        const requestData = await req.json();
        const { stockSymbol, sentimentData, financialData, strategyData } = requestData;

        // 获取市场数据
        const marketData = await fetchMarketData();
        console.log("获取到的市场数据:", marketData);

        // 构建OpenAI Prompt
        const prompt = buildPrompt(stockSymbol, sentimentData, financialData, strategyData, marketData);

        console.log("Sending prompt to OpenAI:", prompt);

        // 调用OpenAI API
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
            temperature: 0, // 设为0以确保一致性
        });

        // 解析OpenAI响应
        const responseContent = completion.choices[0].message.content || '';
        console.log("OpenAI response:", responseContent);

        // 尝试提取投资分数和推荐理由
        let score = 0;
        let reasoning = '';
        let featureImportance = [];

        try {
            // 查找分数（格式：Score: X.XX）
            const scoreMatch = responseContent.match(/Score:\s*(-?\d+\.?\d*)/i);
            if (scoreMatch && scoreMatch[1]) {
                score = parseFloat(scoreMatch[1]);
                // 确保分数在-1到1的范围内
                score = Math.max(-1, Math.min(1, score));
            }

            // 提取推荐理由 - 只保留"Reasoning:"后面的部分，包括所有段落
            const reasoningMatch = responseContent.match(/Reasoning:\s*([\s\S]*?)(?=Feature Weights:|$)/i);
            if (reasoningMatch && reasoningMatch[1]) {
                reasoning = reasoningMatch[1].trim();
            } else {
                // 如果没有明确的"Reasoning:"部分，使用全部回复作为推理
                reasoning = responseContent.trim();
            }

            // 提取特征权重
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

        // 如果没有成功提取特征权重，使用备选的计算方法
        if (featureImportance.length === 0) {
            featureImportance = generateFeatureImportance(sentimentData, financialData, strategyData, marketData);
        }

        // 构造返回数据
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

// 构建发送给OpenAI的提示词
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

// 生成基于规则的特征重要性数据（备选方法，当GPT未能提供有效权重时使用）
function generateFeatureImportance(sentimentData: any, financialData: any, strategyData: any, marketData: any) {
    // 基础权重
    let sentimentWeight = 0.30;
    let financialWeight = 0.35;
    let strategyWeight = 0.35;

    // 根据市场波动率调整策略权重
    if (marketData.volatility.isHigh) {
        // 高波动率时增加策略权重
        strategyWeight = 0.40;
        // 相应减少其他权重以保持总和为1
        sentimentWeight = 0.25;
        financialWeight = 0.35;
    }

    // 根据市场趋势调整权重
    if (marketData.marketTrend === 'Bullish') {
        // 牛市增加情感分析权重
        sentimentWeight += 0.05;
        financialWeight -= 0.05;
    } else if (marketData.marketTrend === 'Bearish') {
        // 熊市增加财务健康权重
        financialWeight += 0.05;
        sentimentWeight -= 0.05;
    }

    // 确保总和为1
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