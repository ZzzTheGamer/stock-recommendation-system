import axios from 'axios';

// Yahoo Finance API URL
const BASE_URL = 'https://query1.finance.yahoo.com/v8/finance/chart/';

// Get historical data
export const getHistoricalData = async (
    symbol: string,
    range: string = '1y',  // Default to 1 year data
    interval: string = '1d' // Default to daily data
) => {
    try {
        const url = `${BASE_URL}${symbol}`;
        const response = await axios.get(url, {
            params: {
                range,
                interval,
                includePrePost: false,
                events: 'div,split',
            }
        });

        const result = response.data.chart.result[0];

        // If there is no data, return an empty array
        if (!result) {
            return { dates: [], prices: [], volumes: [] };
        }

        const timestamps = result.timestamp || [];
        const quotes = result.indicators.quote[0] || {};
        const { open, high, low, close, volume } = quotes;

        // Convert timestamp to date string
        const dates = timestamps.map((timestamp: number) => {
            const date = new Date(timestamp * 1000);
            return date.toISOString().split('T')[0];
        });

        return {
            dates,
            prices: {
                open: open || [],
                high: high || [],
                low: low || [],
                close: close || [],
            },
            volumes: volume || [],
            symbol,
            meta: result.meta
        };
    } catch (error) {
        console.error('Error fetching historical data from Yahoo Finance:', error);
        throw error;
    }
};

// Calculate moving average
export const calculateMA = (prices: number[], period: number) => {
    const result = [];

    for (let i = 0; i < prices.length; i++) {
        if (i < period - 1) {
            // The part that is not enough for the calculation period is filled with null
            result.push(null);
        } else {
            // Calculate moving average
            let sum = 0;
            for (let j = 0; j < period; j++) {
                sum += prices[i - j];
            }
            result.push(sum / period);
        }
    }

    return result;
};

// Calculate RSI
export const calculateRSI = (prices: number[], period: number = 14) => {
    const result = [];
    let gains = 0;
    let losses = 0;

    // First calculate the initial average gain and loss
    for (let i = 1; i < period + 1; i++) {
        const change = prices[i] - prices[i - 1];
        if (change >= 0) {
            gains += change;
        } else {
            losses -= change;
        }
    }

    // Calculate the initial relative strength
    let avgGain = gains / period;
    let avgLoss = losses / period;

    // Fill the first period values with null
    for (let i = 0; i < period; i++) {
        result.push(null);
    }

    // Calculate the first RSI value
    let rs = avgGain / (avgLoss === 0 ? 1 : avgLoss);
    let rsi = 100 - (100 / (1 + rs));
    result.push(rsi);

    // Calculate the remaining RSI values
    for (let i = period + 1; i < prices.length; i++) {
        const change = prices[i] - prices[i - 1];
        let currentGain = 0;
        let currentLoss = 0;

        if (change >= 0) {
            currentGain = change;
        } else {
            currentLoss = -change;
        }

        // Use Wilder's Smoothing Method
        avgGain = ((avgGain * (period - 1)) + currentGain) / period;
        avgLoss = ((avgLoss * (period - 1)) + currentLoss) / period;

        rs = avgGain / (avgLoss === 0 ? 1 : avgLoss);
        rsi = 100 - (100 / (1 + rs));

        result.push(rsi);
    }

    return result;
};

// Calculate Bollinger Bands
export const calculateBollingerBands = (prices: number[], period: number = 20, stdDev: number = 2) => {
    const middleBand = calculateMA(prices, period);
    const upperBand = [];
    const lowerBand = [];

    for (let i = 0; i < prices.length; i++) {
        if (i < period - 1) {
            upperBand.push(null);
            lowerBand.push(null);
        } else {
            // Calculate standard deviation
            let sum = 0;
            for (let j = 0; j < period; j++) {
                // Ensure middleBand[i] exists
                const middleValue = middleBand[i] || 0;
                sum += Math.pow(prices[i - j] - middleValue, 2);
            }
            const std = Math.sqrt(sum / period);

            // Ensure middleBand[i] exists
            const middleValue = middleBand[i] || 0;
            upperBand.push(middleValue + (stdDev * std));
            lowerBand.push(middleValue - (stdDev * std));
        }
    }

    return {
        middleBand,
        upperBand,
        lowerBand
    };
};

// Calculate active trading periods
export const calculateActiveTradingPeriods = (
    signals: Array<{ index: number, type: 'buy' | 'sell' }>,
    totalDays: number
): Array<{ startIndex: number, endIndex: number }> => {
    // Sort signals by date
    signals.sort((a, b) => a.index - b.index);

    const activePeriods: Array<{ startIndex: number, endIndex: number }> = [];

    // Find each pair of buy-sell signals
    let currentBuyIndex: number | null = null;

    for (const signal of signals) {
        if (signal.type === 'buy' && currentBuyIndex === null) {
            // Find the buy signal to start a new active period
            currentBuyIndex = signal.index;
        } else if (signal.type === 'sell' && currentBuyIndex !== null) {
            // Find the sell signal to end the current active period
            activePeriods.push({
                startIndex: currentBuyIndex,
                endIndex: signal.index
            });
            currentBuyIndex = null;
        }
    }

    // If there is a buy signal at the end but no corresponding sell signal, use the last day as the end
    if (currentBuyIndex !== null) {
        activePeriods.push({
            startIndex: currentBuyIndex,
            endIndex: totalDays - 1
        });
    }

    return activePeriods;
};

// Calculate backtest results
export const calculateBacktest = (
    prices: number[],
    signals: Array<{ index: number, type: 'buy' | 'sell' }>,
    initialCapital: number = 10000
) => {
    let capital = initialCapital;
    let shares = 0;
    const equity = [initialCapital];

    // Sort signals by date
    signals.sort((a, b) => a.index - b.index);

    // Calculate active trading periods
    const activePeriods = calculateActiveTradingPeriods(signals, prices.length);

    // Initialize daily equity
    for (let i = 1; i < prices.length; i++) {
        // Find the trade signal for the current date
        const todaySignal = signals.find(signal => signal.index === i);

        if (todaySignal) {
            if (todaySignal.type === 'buy' && capital > 0) {
                // Buy signal, buy all shares
                shares = capital / prices[i];
                capital = 0;
            } else if (todaySignal.type === 'sell' && shares > 0) {
                // Sell signal, sell all shares
                capital = shares * prices[i];
                shares = 0;
            }
        }

        // Calculate current equity
        const currentEquity = capital + (shares * prices[i]);
        equity.push(currentEquity);
    }

    // Calculate returns
    const returns = [];
    for (let i = 1; i < equity.length; i++) {
        returns.push((equity[i] / equity[i - 1]) - 1);
    }

    // Calculate cumulative returns (expressed as a percentage)
    const cumulativeReturns = [];
    let cumReturn = 0;

    for (let i = 0; i < returns.length; i++) {
        cumReturn = (1 + cumReturn) * (1 + returns[i]) - 1;
        cumulativeReturns.push(cumReturn * 100);
    }

    return {
        equity,
        returns,
        cumulativeReturns,
        activePeriods  // Add active trading periods information
    };
};

// Get current price
export const getCurrentPrice = async (symbol: string) => {
    try {
        const url = `${BASE_URL}${symbol}`;
        const response = await axios.get(url, {
            params: {
                range: '1d',
                interval: '1m'
            }
        });

        const result = response.data.chart.result[0];
        if (!result) {
            throw new Error('No data returned from Yahoo Finance');
        }

        return result.meta.regularMarketPrice;
    } catch (error) {
        console.error('Error fetching current price from Yahoo Finance:', error);
        throw error;
    }
};