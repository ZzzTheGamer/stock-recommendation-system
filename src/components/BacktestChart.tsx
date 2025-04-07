import React from 'react';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ChartOptions
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

interface BacktestChartProps {
    dates: string[];
    strategyReturns: number[];
    activePeriods?: Array<{ startIndex: number, endIndex: number }>;
    totalDays?: number;
    activeDays?: number;
    height?: number;
    title?: string;
}

const BacktestChart: React.FC<BacktestChartProps> = ({
    dates,
    strategyReturns,
    activePeriods = [],
    totalDays = 0,
    activeDays = 0,
    height = 400,
    title = 'Strategy Returns'
}) => {
    // Only display data during active trading periods
    let filteredDates: string[] = [];
    let filteredReturns: number[] = [];

    if (activePeriods && activePeriods.length > 0) {
        // When active trading periods exist, only show data from these periods
        activePeriods.forEach(period => {
            for (let i = period.startIndex; i <= period.endIndex; i++) {
                if (i < dates.length) {
                    filteredDates.push(dates[i]);
                    filteredReturns.push(strategyReturns[i]);
                }
            }
        });
    } else {
        // When no active trading periods exist, show all data
        filteredDates = [...dates];
        filteredReturns = [...strategyReturns];
    }

    // Format date display
    const formattedDates = filteredDates.map(date => {
        const d = new Date(date);
        return d.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    });

    // If too many data points, filter to improve readability
    const filterStep = formattedDates.length > 120 ? Math.floor(formattedDates.length / 120) : 1;
    const displayDates = formattedDates.filter((_, i) => i % filterStep === 0);
    const displayReturns = filteredReturns.filter((_, i) => i % filterStep === 0);

    const data = {
        labels: displayDates,
        datasets: [
            {
                label: 'Strategy Returns',
                data: displayReturns,
                borderColor: 'rgb(53, 162, 235)',
                backgroundColor: 'rgba(53, 162, 235, 0.5)',
                borderWidth: 2,
                tension: 0.2,
                pointRadius: 1,
                pointHoverRadius: 5,
            }
        ],
    };

    const options: ChartOptions<'line'> = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top' as const,
                labels: {
                    font: {
                        size: 14
                    }
                }
            },
            title: {
                display: Boolean(title),
                text: title,
                font: {
                    size: 16,
                    weight: 'bold'
                }
            },
            tooltip: {
                callbacks: {
                    label: (context) => {
                        return `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`;
                    }
                },
                titleFont: {
                    size: 14
                },
                bodyFont: {
                    size: 14
                }
            }
        },
        scales: {
            x: {
                ticks: {
                    maxRotation: 60,
                    minRotation: 60,
                    font: {
                        size: 11
                    },
                    callback: function (val, index) {
                        return index % Math.ceil(displayDates.length / 15) === 0 ? this.getLabelForValue(val as number) : '';
                    }
                },
                grid: {
                    display: false,
                }
            },
            y: {
                ticks: {
                    callback: (value) => `${value}%`,
                    font: {
                        size: 12
                    }
                },
                grid: {
                    color: 'rgba(200, 200, 200, 0.2)',
                }
            },
        },
    };

    return (
        <div>
            <div style={{ height: `${height}px`, position: 'relative', width: '100%' }}>
                <Line data={data} options={options} />
            </div>
            {activePeriods && activePeriods.length > 0 && totalDays > 0 && (
                <div className="text-xs text-gray-500 mt-2 italic">
                    Note: The chart only displays data during active trading periods (from buy signal to sell signal), totaling {activeDays} days, which represents {Math.round((activeDays / totalDays) * 100)}% of the total backtest period of {totalDays} days.
                </div>
            )}
        </div>
    );
};

export default BacktestChart; 