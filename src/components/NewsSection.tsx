'use client';

import React, { useState } from 'react';
import { format, parseISO } from 'date-fns';
import { FaChevronUp } from 'react-icons/fa';

interface NewsItem {
    id: number;
    title: string;
    summary: string;
    source: string;
    date: string;
    url: string;
}

const NewsSection: React.FC<{ news: NewsItem[] }> = ({ news }) => {
    const [visibleNews, setVisibleNews] = useState(4);
    const initialNewsCount = 4;

    const handleLoadMore = () => {
        setVisibleNews(prev => prev + 4);
    };

    const handleCollapse = () => {
        setVisibleNews(initialNewsCount);
    };

    // Format date
    const formatDate = (dateString: string) => {
        try {
            // AlphaVantage API returns date in format yyyyMMddTHHmmss
            if (dateString.includes('T') && !dateString.includes('-')) {
                const year = dateString.substring(0, 4);
                const month = dateString.substring(4, 6);
                const day = dateString.substring(6, 8);
                const time = dateString.substring(9, 11) + ':' + dateString.substring(11, 13);
                return `${year}-${month}-${day} ${time}`;
            }
            // Standard ISO format
            return format(parseISO(dateString), 'yyyy-MM-dd HH:mm');
        } catch (error) {
            return dateString;
        }
    };

    return (
        <div className="card">
            <h2 className="text-xl font-bold mb-4">Related News</h2>

            {news.length === 0 ? (
                <p className="text-gray-500">No related news available</p>
            ) : (
                <>
                    <div className="space-y-4">
                        {news.slice(0, visibleNews).map((item) => (
                            <div key={item.id} className="border-b border-gray-200 pb-4 last:border-0">
                                <a
                                    href={item.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="block hover:text-primary transition-colors"
                                >
                                    <h3 className="text-lg font-semibold mb-1">{item.title}</h3>
                                    <p className="text-gray-600 text-sm mb-2">{item.summary}</p>
                                    <div className="flex items-center text-xs text-gray-500">
                                        <span className="mr-2">{item.source}</span>
                                        <span>{formatDate(item.date)}</span>
                                    </div>
                                </a>
                            </div>
                        ))}
                    </div>

                    <div className="mt-4 flex justify-between">
                        {visibleNews < news.length && (
                            <button
                                onClick={handleLoadMore}
                                className="btn btn-secondary text-sm"
                            >
                                Load More
                            </button>
                        )}

                        {visibleNews > initialNewsCount && (
                            <button
                                onClick={handleCollapse}
                                className="btn btn-secondary text-sm flex items-center"
                            >
                                <FaChevronUp className="mr-1" />
                                Collapse
                            </button>
                        )}
                    </div>
                </>
            )}
        </div>
    );
};

export default NewsSection; 