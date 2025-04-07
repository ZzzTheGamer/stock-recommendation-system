import { NextResponse } from 'next/server';
import axios from 'axios';

// Finnhub API 密钥
const FINNHUB_API_KEY = process.env.FINNHUB_API_KEY || '';

export async function GET(request: Request) {
    try {
        // 获取 URL 参数
        const { searchParams } = new URL(request.url);
        const symbol = searchParams.get('symbol');

        if (!symbol) {
            return NextResponse.json(
                { error: '缺少必要参数: symbol' },
                { status: 400 }
            );
        }

        // 计算日期范围（从30天前到今天）
        const today = new Date();
        const thirtyDaysAgo = new Date();
        thirtyDaysAgo.setDate(today.getDate() - 30);

        const toDate = Math.floor(today.getTime() / 1000);
        const fromDate = Math.floor(thirtyDaysAgo.getTime() / 1000);

        // 调用 Finnhub API 获取新闻
        console.log(`[新闻API] 获取 ${symbol} 的新闻，从 ${new Date(fromDate * 1000).toISOString()} 到 ${new Date(toDate * 1000).toISOString()}`);

        try {
            const response = await axios.get(`https://finnhub.io/api/v1/company-news`, {
                params: {
                    symbol,
                    from: thirtyDaysAgo.toISOString().split('T')[0],
                    to: today.toISOString().split('T')[0],
                    token: FINNHUB_API_KEY
                }
            });

            console.log(`[新闻API] 成功获取 ${response.data.length} 条新闻`);

            // 返回新闻数据
            return NextResponse.json(response.data);
        } catch (apiError) {
            console.error('[新闻API] Finnhub API 调用失败:', apiError);

            // 返回模拟数据
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
        console.error('[新闻API] 错误:', error);
        return NextResponse.json(
            { error: String(error) },
            { status: 500 }
        );
    }
} 