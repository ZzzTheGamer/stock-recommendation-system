'use client';

import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import { FaArrowUp, FaArrowDown } from 'react-icons/fa';

// Register Chart.js components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

interface StockDataProps {
    symbol: string;
    name: string;
    price: number;
    change: number;
    changePercent: number;
    open: number;
    high: number;
    low: number;
    volume: number;
    week52High: number;
    week52Low: number;
    marketCap: string;
    beta: number;
    eps: number;
    dividendYield: number;
    peRatio?: number;
    chartData?: {
        labels: string[];
        prices: number[];
    };
}

const StockHeader: React.FC<{ stockData: StockDataProps }> = ({ stockData }) => {
    const isPositive = stockData.change >= 0;

    // Prepare chart data using actual daily data for the past year
    const chartData = {
        labels: stockData.chartData?.labels || [],
        datasets: [
            {
                label: 'Stock Price',
                data: stockData.chartData?.prices || [],
                fill: true,
                backgroundColor: isPositive ? 'rgba(0, 177, 106, 0.1)' : 'rgba(255, 68, 68, 0.1)',
                borderColor: isPositive ? 'rgba(0, 177, 106, 1)' : 'rgba(255, 68, 68, 1)',
                tension: 0.4,
            },
        ],
    };

    return (
        <div className="card">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Stock Basic Information - Real-time Trading Data (Left Side) */}
                <div className="col-span-1">
                    <div className="flex items-baseline mb-2">
                        <h2 className="text-2xl font-bold mr-2">{stockData.name}</h2>
                        <span className="text-gray-500">({stockData.symbol})</span>
                    </div>

                    <div className="flex items-center mb-4">
                        <span className="text-3xl font-bold mr-3">${stockData.price.toFixed(2)}</span>
                        <div className={`flex items-center ${isPositive ? 'text-success' : 'text-danger'}`}>
                            {isPositive ? <FaArrowUp className="mr-1" /> : <FaArrowDown className="mr-1" />}
                            <span className="font-medium">{stockData.change.toFixed(2)} ({stockData.changePercent.toFixed(2)}%)</span>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                        {/* Real-time trading information */}
                        <div>
                            <span className="text-gray-500">Open:</span>
                            <span className="ml-2 font-medium">${stockData.open.toFixed(2)}</span>
                        </div>
                        <div>
                            <span className="text-gray-500">High:</span>
                            <span className="ml-2 font-medium">${stockData.high.toFixed(2)}</span>
                        </div>
                        <div>
                            <span className="text-gray-500">Low:</span>
                            <span className="ml-2 font-medium">${stockData.low.toFixed(2)}</span>
                        </div>
                        <div>
                            <span className="text-gray-500">Volume:</span>
                            <span className="ml-2 font-medium">{stockData.volume.toLocaleString()}</span>
                        </div>
                        <div>
                            <span className="text-gray-500">52W High:</span>
                            <span className="ml-2 font-medium">${stockData.week52High.toFixed(2)}</span>
                        </div>
                        <div>
                            <span className="text-gray-500">52W Low:</span>
                            <span className="ml-2 font-medium">${stockData.week52Low.toFixed(2)}</span>
                        </div>
                    </div>
                </div>

                {/* Stock Chart (Right Side) */}
                <div className="col-span-1 md:col-span-2 h-64">
                    <div className="mb-2 text-sm font-medium">Price Chart (Past Year)</div>
                    <Line
                        data={chartData}
                        options={{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    display: false,
                                },
                                tooltip: {
                                    mode: 'index',
                                    intersect: false,
                                    callbacks: {
                                        label: function (context) {
                                            return `$${context.raw}`;
                                        }
                                    }
                                },
                            },
                            scales: {
                                y: {
                                    grid: {
                                        display: true,
                                        color: 'rgba(0, 0, 0, 0.05)',
                                    },
                                    ticks: {
                                        callback: function (value) {
                                            return '$' + value;
                                        }
                                    }
                                },
                                x: {
                                    grid: {
                                        display: false,
                                    },
                                    ticks: {
                                        maxTicksLimit: 12, // Show approximately one month per tick
                                        maxRotation: 0,
                                    }
                                },
                            },
                        }}
                    />
                </div>
            </div>

            {/* Additional Metrics - Calculated Indicators (Bottom) */}
            <div className="mt-6 pt-4 border-t border-gray-200">
                <h3 className="text-lg font-medium mb-3">Financial Metrics</h3>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                    <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-gray-500 mb-1">Market Cap</div>
                        <div className="font-medium">{stockData.marketCap}</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-gray-500 mb-1">Beta</div>
                        <div className="font-medium">{stockData.beta.toFixed(2)}</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-gray-500 mb-1">EPS</div>
                        <div className="font-medium">${stockData.eps.toFixed(2)}</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                        <div className="text-gray-500 mb-1">Dividend Yield</div>
                        <div className="font-medium">{stockData.dividendYield.toFixed(2)}%</div>
                    </div>
                    {stockData.peRatio && (
                        <div className="p-3 bg-gray-50 rounded-lg">
                            <div className="text-gray-500 mb-1">P/E Ratio</div>
                            <div className="font-medium">{stockData.peRatio.toFixed(2)}</div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default StockHeader; 