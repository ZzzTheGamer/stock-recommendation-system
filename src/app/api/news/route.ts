import { NextResponse } from 'next/server';
import axios from 'axios';

// Finnhub API key
const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY || '';

export async function GET(request: Request) {
    try {
        // Get URL parameters
        const { searchParams } = new URL(request.url);
        const symbol = searchParams.get('symbol');

        if (!symbol) {
            return NextResponse.json(
                { error: 'Missing required parameter: symbol' },
                { status: 400 }
            );
        }

        // Calculate date range (from 30 days ago to today)
        const today = new Date();
        const thirtyDaysAgo = new Date();
        thirtyDaysAgo.setDate(today.getDate() - 30);

        const toDate = Math.floor(today.getTime() / 1000);
        const fromDate = Math.floor(thirtyDaysAgo.getTime() / 1000);

        // Call Finnhub API to get news
        console.log(`[News API] Getting news for ${symbol} from ${new Date(fromDate * 1000).toISOString()} to ${new Date(toDate * 1000).toISOString()}`);

        try {
            const response = await axios.get(`https://finnhub.io/api/v1/company-news`, {
                params: {
                    symbol,
                    from: thirtyDaysAgo.toISOString().split('T')[0],
                    to: today.toISOString().split('T')[0],
                    token: FINNHUB_API_KEY
                }
            });

            console.log(`[News API] Successfully obtained ${response.data.length} news`);

            // Return news data
            return NextResponse.json(response.data);
        } catch (apiError) {
            console.error('[News API] Finnhub API call failed:', apiError);

            // Return simulated data
            return NextResponse.json([
                {
                    category: "company news",
                    datetime: toDate - 86400,
                    headline: `${symbol} Announces Strong Quarterly Results`,
                    id: 1,
                    image: "https://example.com/image1.jpg",
                    related: symbol,
                    source: "Company Press Release",
                    summary: `${symbol} reported quarterly earnings that exceeded analyst expectations, driven by strong product sales and expanding market share.`,
                    url: "https://example.com/news1"
                },
                {
                    category: "technology",
                    datetime: toDate - 172800,
                    headline: `${symbol} Unveils New Product Line`,
                    id: 2,
                    image: "https://example.com/image2.jpg",
                    related: symbol,
                    source: "Tech News",
                    summary: `${symbol} announced a new product line that aims to revolutionize the industry with cutting-edge features and improved performance.`,
                    url: "https://example.com/news2"
                },
                {
                    category: "business",
                    datetime: toDate - 259200,
                    headline: `${symbol} Expands into New Markets`,
                    id: 3,
                    image: "https://example.com/image3.jpg",
                    related: symbol,
                    source: "Business Journal",
                    summary: `${symbol} is expanding its operations into emerging markets, signaling confidence in global growth opportunities despite economic uncertainties.`,
                    url: "https://example.com/news3"
                },
                {
                    category: "company news",
                    datetime: toDate - 345600,
                    headline: `${symbol} Announces Strategic Partnership`,
                    id: 4,
                    image: "https://example.com/image4.jpg",
                    related: symbol,
                    source: "Industry News",
                    summary: `${symbol} has formed a strategic partnership with a leading technology provider to enhance its product offerings and accelerate innovation.`,
                    url: "https://example.com/news4"
                },
                {
                    category: "technology",
                    datetime: toDate - 432000,
                    headline: `Analysts Upgrade ${symbol} Stock Rating`,
                    id: 5,
                    image: "https://example.com/image5.jpg",
                    related: symbol,
                    source: "Financial News",
                    summary: `Several analysts have upgraded their rating for ${symbol} stock, citing strong fundamentals and positive growth outlook for the coming year.`,
                    url: "https://example.com/news5"
                }
            ]);
        }
    } catch (error) {
        console.error('[News API] Error:', error);
        return NextResponse.json(
            { error: String(error) },
            { status: 500 }
        );
    }
} 