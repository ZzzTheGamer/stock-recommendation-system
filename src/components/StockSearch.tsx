'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { FaSearch } from 'react-icons/fa';

const StockSearch: React.FC = () => {
    const [symbol, setSymbol] = useState('');
    const router = useRouter();

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (symbol.trim()) {
            router.push(`/?symbol=${symbol.trim().toUpperCase()}`);
        }
    };

    return (
        <div className="absolute top-4 right-4 z-10">
            <form onSubmit={handleSubmit} className="flex items-center">
                <input
                    type="text"
                    value={symbol}
                    onChange={(e) => setSymbol(e.target.value)}
                    placeholder="Enter stock symbol (e.g. AAPL)"
                    className="px-3 py-2 border border-gray-300 rounded-l-md focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                />
                <button
                    type="submit"
                    className="bg-primary text-white px-3 py-2 rounded-r-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
                >
                    <FaSearch />
                </button>
            </form>
        </div>
    );
};

export default StockSearch; 