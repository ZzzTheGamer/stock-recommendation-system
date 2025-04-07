// import axios from 'axios';

// // OpenAI API配置
// const OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions';

// // 默认模型
// const GPT4O_MODEL = 'gpt-4o';
// const GPT35_TURBO_MODEL = 'gpt-3.5-turbo';

// /**
//  * 调用OpenAI API，带重试机制
//  */
// export const callOpenAIAPI = async (
//     prompt: string,
//     model: string = GPT4O_MODEL,
//     maxTokens: number = 2048,
//     temperature: number = 0.7,
//     maxRetries: number = 2
// ) => {
//     let retries = 0;
//     let lastError;

//     while (retries <= maxRetries) {
//         try {
//             console.log(`[OpenAI调用] 尝试调用模型: ${model}，重试次数: ${retries}`);

//             // 构建消息
//             const messages = [
//                 {
//                     role: 'system',
//                     content: 'You are a financial analyst who provides concise, direct analysis without any markdown formatting or symbols like * or **. Keep your responses brief and to the point.'
//                 },
//                 {
//                     role: 'user',
//                     content: prompt
//                 }
//             ];

//             const response = await axios.post(
//                 OPENAI_API_URL,
//                 {
//                     model: model,
//                     messages: messages,
//                     max_tokens: maxTokens,
//                     temperature: temperature
//                 },
//                 {
//                     headers: {
//                         'Authorization': `Bearer ${OPENAI_API_KEY}`,
//                         'Content-Type': 'application/json'
//                     },
//                     timeout: 60000 // 60秒超时
//                 }
//             );

//             console.log(`[OpenAI调用] 成功，状态码: ${response.status}`);
//             return response.data.choices[0].message.content;
//         } catch (error: any) {
//             retries++;
//             lastError = error;

//             console.error(`[OpenAI调用] 失败，重试 ${retries}/${maxRetries}:`, error.message);

//             if (retries <= maxRetries) {
//                 // 指数退避策略
//                 const delay = Math.pow(2, retries) * 1000 + Math.random() * 1000;
//                 console.log(`[OpenAI调用] 等待 ${delay}ms 后重试...`);
//                 await new Promise(resolve => setTimeout(resolve, delay));
//             }
//         }
//     }

//     // 所有重试都失败
//     throw lastError;
// };

// /**
//  * 使用GPT-4o生成财务分析报告
//  */
// export const generateFinancialAnalysisWithGPT4o = async (
//     symbol: string,
//     companyName: string,
//     industry: string,
//     financialData: any
// ) => {
//     try {
//         console.log(`[FinGPT分析] 开始使用GPT-4o生成${symbol}的财务分析报告`);

//         // 构建金融专家提示词 - 精简版，无markdown格式
//         const prompt = `You are a senior financial analyst. Provide a concise financial analysis for ${companyName} (${symbol}) in the ${industry} industry based on these metrics:

// Revenue Growth: ${financialData.revenueGrowth}%
// Net Income Growth: ${financialData.netIncomeGrowth}%
// Current Ratio: ${financialData.currentRatio}
// Debt-to-Equity Ratio: ${financialData.debtToEquity}
// Return on Equity (ROE): ${financialData.returnOnEquity}%
// Price-to-Earnings Ratio (P/E): ${financialData.peRatio}
// Analyst Target Price: $${financialData.targetPrice}

// Your analysis should be brief and direct with no formatting symbols (like * or **). Include:

// Financial Summary: Brief overview of financial health (2-3 sentences)
// Key Strengths: 2-3 key strengths with minimal explanation
// Potential Risks: 2-3 key risks with minimal explanation
// Investment Recommendation: Clear rating (Strong Buy/Buy/Hold/Sell/Strong Sell) with brief rationale

// Keep the entire response under 400 words with simple paragraph headings (no markdown). Use plain, direct language.`;

//         // 调用GPT-4o模型
//         console.log('[FinGPT分析] 调用GPT-4o...');
//         const analysisText = await callOpenAIAPI(prompt, GPT4O_MODEL, 2048, 0.7);
//         console.log('[FinGPT分析] GPT-4o响应成功');

//         // 尝试提取各部分内容 - 简化的正则表达式，适用于精简格式
//         const summaryMatch = analysisText.match(/(?:Financial Summary|Summary)[:\s]+([\s\S]*?)(?=(?:Key Strengths|Strengths)[:\s]+|$)/i);
//         const strengthsMatch = analysisText.match(/(?:Key Strengths|Strengths)[:\s]+([\s\S]*?)(?=(?:Potential Risks|Risks|Weaknesses)[:\s]+|$)/i);
//         const weaknessesMatch = analysisText.match(/(?:Potential Risks|Risks|Weaknesses)[:\s]+([\s\S]*?)(?=(?:Investment Recommendation|Recommendation)[:\s]+|$)/i);
//         const recommendationMatch = analysisText.match(/(?:Investment Recommendation|Recommendation)[:\s]+([\s\S]*?)$/i);

//         // 提取各部分内容，确保有默认值
//         const summary = summaryMatch && summaryMatch[1] ? summaryMatch[1].trim() :
//             `${companyName} (${symbol}) shows stable revenue growth of ${financialData.revenueGrowth}% with net income growth of ${financialData.netIncomeGrowth}%. Current ratio is ${financialData.currentRatio} and debt-to-equity ratio is ${financialData.debtToEquity}.`;

//         const strengths = strengthsMatch && strengthsMatch[1] ? strengthsMatch[1].trim() :
//             `1. Stable revenue growth of ${financialData.revenueGrowth}%\n2. Strong return on equity of ${financialData.returnOnEquity}%`;

//         const weaknesses = weaknessesMatch && weaknessesMatch[1] ? weaknessesMatch[1].trim() :
//             `1. Net income decline of ${financialData.netIncomeGrowth}%\n2. Current ratio below 1 at ${financialData.currentRatio}`;

//         // We don't need a detailed analysis section for this concise format
//         const metricsAnalysis = '';

//         // 提取推荐评级 - 简化的评级提取逻辑
//         let recommendation = 'Hold';
//         if (recommendationMatch && recommendationMatch[1]) {
//             const recText = recommendationMatch[1].toLowerCase();

//             // 更全面的匹配模式，处理各种表述方式
//             if (recText.match(/strong(\s+)buy|strong\s+purchase|highly\s+recommended\s+buy|aggressive\s+buy/i)) {
//                 recommendation = 'Strong Buy';
//             } else if (recText.match(/buy|purchase|accumulate|add/i) && !recText.match(/do\s+not\s+buy|don't\s+buy|avoid\s+buying/i)) {
//                 recommendation = 'Buy';
//             } else if (recText.match(/hold|maintain|neutral|wait|observe/i)) {
//                 recommendation = 'Hold';
//             } else if (recText.match(/strong(\s+)sell|aggressive\s+sell|immediate\s+sell/i)) {
//                 recommendation = 'Strong Sell';
//             } else if (recText.match(/sell|reduce|decrease|divest/i) && !recText.match(/do\s+not\s+sell|don't\s+sell|avoid\s+selling/i)) {
//                 recommendation = 'Sell';
//             }

//             // 检查是否有明确的评级声明
//             const ratingMatch = recText.match(/rating[:\s]+\s*(strong\s*buy|buy|hold|sell|strong\s*sell)/i);
//             if (ratingMatch) {
//                 const rating = ratingMatch[1].toLowerCase();
//                 if (rating.match(/strong\s*buy/)) recommendation = 'Strong Buy';
//                 else if (rating.match(/buy/)) recommendation = 'Buy';
//                 else if (rating.match(/hold/)) recommendation = 'Hold';
//                 else if (rating.match(/strong\s*sell/)) recommendation = 'Strong Sell';
//                 else if (rating.match(/sell/)) recommendation = 'Sell';
//             }
//         }

//         console.log(`[FinGPT分析] 分析完成，推荐: ${recommendation}`);

//         return {
//             summary,
//             strengths,
//             weaknesses,
//             // No detailed analysis in concise format
//             metricsAnalysis: "",
//             recommendation,
//             rating: recommendation,
//             apiCallSuccess: true,
//             isSimulated: false,
//             keyMetrics: financialData,
//             analysisDate: new Date().toISOString()
//         };
//     } catch (error) {
//         console.error('[FinGPT分析] 出错:', error);

//         // 返回模拟数据
//         console.log('[FinGPT分析] 返回模拟的财务分析数据');
//         return {
//             summary: `${companyName} (${symbol}) has revenue growth of ${financialData.revenueGrowth}% and net income growth of ${financialData.netIncomeGrowth}%. The financial position shows mixed performance with some strengths and risks.`,
//             strengths: `1. Revenue growth of ${financialData.revenueGrowth}%\n2. Return on equity of ${financialData.returnOnEquity}%`,
//             weaknesses: `1. Net income growth of ${financialData.netIncomeGrowth}%\n2. Current ratio of ${financialData.currentRatio}`,
//             metricsAnalysis: "",
//             recommendation: 'Hold',
//             rating: 'Hold',
//             apiCallSuccess: false,
//             isSimulated: true,
//             keyMetrics: financialData,
//             analysisDate: new Date().toISOString()
//         };
//     }
// }; 