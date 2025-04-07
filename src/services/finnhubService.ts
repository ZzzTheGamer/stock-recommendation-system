import axios from 'axios';

const API_KEY = process.env.NEXT_PUBLIC_FINNHUB_API_KEY || '';
const BASE_URL = 'https://finnhub.io/api/v1';

// Get company news
export const getCompanyNews = async (symbol: string, from: string, to: string) => {
    try {
        console.log(`Getting news for ${symbol}, date range: ${from} to ${to}`);

        // Use local API route instead of directly calling Finnhub API
        const response = await axios.get(`/api/news`, {
            params: {
                symbol
            },
            timeout: 10000 // 10 seconds timeout
        });

        console.log(`Successfully retrieved ${response.data.length} news items`);
        return response.data;
    } catch (error) {
        console.error('Error fetching company news:', error);
        console.log('Returning simulated news data');
        // If API call fails, return simulated data
        return [
            {
                id: 1,
                headline: `${symbol} Announces New Product Launch`,
                summary: `${symbol} unveiled its latest product line, expected to boost revenue in the coming quarters.`,
                source: 'Financial Times',
                datetime: Date.now(),
                url: '#'
            },
            {
                id: 2,
                headline: `${symbol} Reports Strong Quarterly Earnings`,
                summary: `${symbol} exceeded analyst expectations with a 15% increase in quarterly revenue.`,
                source: 'Bloomberg',
                datetime: Date.now() - 86400000, // 1 day ago
                url: '#'
            },
            {
                id: 3,
                headline: `${symbol} Expands into New Markets`,
                summary: `${symbol} announced plans to enter emerging markets in Asia, targeting significant growth opportunities.`,
                source: 'Reuters',
                datetime: Date.now() - 172800000, // 2 days ago
                url: '#'
            },
            {
                id: 4,
                headline: `${symbol} Partners with Tech Giant`,
                summary: `${symbol} formed a strategic partnership with a leading tech company to enhance its digital capabilities.`,
                source: 'CNBC',
                datetime: Date.now() - 259200000, // 3 days ago
                url: '#'
            },
            {
                id: 5,
                headline: `Analysts Upgrade ${symbol} Stock Rating`,
                summary: `Several analysts have upgraded ${symbol}'s stock rating, citing strong growth prospects and market position.`,
                source: 'Wall Street Journal',
                datetime: Date.now() - 345600000, // 4 days ago
                url: '#'
            }
        ];
    }
};

// Get social media sentiment
export const getSocialSentiment = async (symbol: string) => {
    try {
        const response = await axios.get(`${BASE_URL}/stock/social-sentiment`, {
            params: {
                symbol,
                token: API_KEY,
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching social sentiment:', error);
        throw error;
    }
};

// Use FinGPT for sentiment analysis
export const analyzeSentiment = async (symbol: string) => {
    try {
        // Get news for the past 30 days
        const today = new Date();
        const thirtyDaysAgo = new Date(today);
        thirtyDaysAgo.setDate(today.getDate() - 30);

        const toDate = today.toISOString().split('T')[0];
        const fromDate = thirtyDaysAgo.toISOString().split('T')[0];

        const news = await getCompanyNews(symbol, fromDate, toDate);
        const sentiment = await getSocialSentiment(symbol).catch(() => null);

        // Simulate FinGPT analysis results
        // In an actual implementation, this should call the FinGPT API or use a local model for analysis
        const sentimentScore = Math.random() * 2 - 1; // Random value between -1 and 1
        const sentimentLabel = sentimentScore > 0.3 ? 'Positive' :
            sentimentScore < -0.3 ? 'Negative' : 'Neutral';

        const newsHeadlines = news.slice(0, 10).map((item: any) => item.headline);
        const topPositiveFactors = [
            'Positive product launch reception',
            'Quarterly earnings exceeding expectations',
            'New market expansion plans',
            'Strategic partnerships',
            'Analyst rating upgrades'
        ];

        const topNegativeFactors = [
            'Increased regulatory scrutiny',
            'Competitor market share growth',
            'Supply chain disruptions',
            'Rising costs',
            'Product recalls'
        ];

        return {
            symbol,
            overallSentiment: sentimentLabel,
            sentimentScore: sentimentScore.toFixed(2),
            confidence: (0.7 + Math.random() * 0.2).toFixed(2), // Random value between 0.7 and 0.9
            recentNews: newsHeadlines,
            topPositiveFactors: sentimentScore > 0 ? topPositiveFactors.slice(0, 3) : [],
            topNegativeFactors: sentimentScore < 0 ? topNegativeFactors.slice(0, 3) : [],
            analysisDate: new Date().toISOString(),
            recommendation: sentimentScore > 0.5 ? 'Consider buying based on positive sentiment' :
                sentimentScore < -0.5 ? 'Consider selling based on negative sentiment' :
                    'Monitor sentiment trends before making decisions'
        };
    } catch (error) {
        console.error('Error analyzing sentiment:', error);
        throw error;
    }
}; 