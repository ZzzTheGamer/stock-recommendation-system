'use client';

import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import StockHeader from '@/components/StockHeader';
import NewsSection from '@/components/NewsSection';
import AnalysisSection from '@/components/AnalysisSection';
import FinalRecommendation from '@/components/FinalRecommendation';
import StockSearch from '@/components/StockSearch';
import LoadingState from '@/components/LoadingState';
import ErrorState from '@/components/ErrorState';
import { processStockData, processNewsData } from '@/services/alphaVantageService';

export default function Home() {
    const searchParams = useSearchParams();
    const symbolParam = searchParams?.get('symbol');
    const [symbol, setSymbol] = useState(symbolParam || 'AAPL');

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // 股票数据状态
    const [stockData, setStockData] = useState({
        symbol: 'AAPL',
        name: 'Apple Inc.',
        price: 182.63,
        change: 1.25,
        changePercent: 0.69,
        open: 181.27,
        high: 183.12,
        low: 180.63,
        volume: 58432156,
        week52High: 198.23,
        week52Low: 124.17,
        marketCap: '2.87T',
        beta: 1.28,
        eps: 6.14,
        dividendYield: 0.5,
    });

    // 新闻数据状态
    const [newsData, setNewsData] = useState([
        {
            id: 1,
            title: 'Apple Announces New iPhone Features',
            summary: 'Apple unveils groundbreaking features for the next iPhone generation, focusing on AI capabilities.',
            source: 'TechCrunch',
            date: '2023-06-15T14:30:00Z',
            url: '#',
        },
        {
            id: 2,
            title: "Apple's Q2 Earnings Beat Expectations",
            summary: 'The tech giant reported stronger than expected earnings, with services revenue hitting an all-time high.',
            source: 'Bloomberg',
            date: '2023-05-02T18:45:00Z',
            url: '#',
        },
        {
            id: 3,
            title: 'Apple Expands Manufacturing in India',
            summary: 'In a strategic move to diversify production, Apple increases its manufacturing capacity in India.',
            source: 'Reuters',
            date: '2023-04-28T09:15:00Z',
            url: '#',
        },
        {
            id: 4,
            title: 'New MacBook Pro Models Expected This Fall',
            summary: 'Industry insiders suggest Apple will release updated MacBook Pro models with enhanced performance.',
            source: 'MacRumors',
            date: '2023-04-15T11:20:00Z',
            url: '#',
        },
    ]);

    // 当URL参数变化时更新symbol
    useEffect(() => {
        if (symbolParam) {
            setSymbol(symbolParam);
        }
    }, [symbolParam]);

    // 获取股票数据和新闻
    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);

            try {
                // Fetch stock data and news in parallel
                const [stockDataResult, newsDataResult] = await Promise.all([
                    processStockData(symbol),
                    processNewsData(symbol)
                ]);

                setStockData(stockDataResult);
                setNewsData(newsDataResult);
            } catch (err) {
                console.error('Error fetching data:', err);
                setError('Failed to fetch data. Please try again later or check if the stock symbol is correct.');
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [symbol]);

    return (
        <div className="space-y-8 relative">
            <h1 className="text-3xl font-bold text-center mb-8">Stock Recommendation System</h1>

            {/* 股票搜索框 */}
            <StockSearch />

            {loading ? (
                <LoadingState message={`Loading data for ${symbol}...`} />
            ) : error ? (
                <ErrorState message={error} />
            ) : (
                <>
                    {/* 股票基本信息 */}
                    <StockHeader stockData={stockData} />

                    {/* 新闻列表 */}
                    <NewsSection news={newsData} />

                    {/* 分析模块 */}
                    <AnalysisSection stockSymbol={stockData.symbol} />

                    {/* 最终推荐 */}
                    <FinalRecommendation stockSymbol={stockData.symbol} />
                </>
            )}
        </div>
    );
} 