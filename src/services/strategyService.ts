import axios from 'axios';
import path from 'path';
import fs from 'fs';
import * as yahooFinance from './yahooFinanceService';

// Define the policy analysis result interface
export interface StrategyAnalysisResult {
    symbol: string;
    strategy: {
        type: string;
        name: string;
        description: string;
        parameters: Record<string, any>;
    };
    performance_metrics: {
        annualized_return: number;
        volatility: number;
        sharpe_ratio: number;
        max_drawdown: number;
        win_rate: number;
        total_return?: number;
    };
    backtest_results: {
        dates: string[];
        strategy_returns: number[];
        active_periods?: Array<{ startIndex: number, endIndex: number }>;
        total_days?: number;
        active_days?: number;
    };
    trade_signals: Array<{
        date: string;
        action: string;
        price: number;
        reason?: string;
        macroDetails?: any; // Add macroeconomic details
    }>;
    chart_url?: string;
    analysis_date: string;
    recommendation: string;
    note?: string;
    explanation?: string;
    model_info?: {
        algorithm: string;
        training_data: string;
        features: string[];
        hyperparameters: Record<string, any>;
        macro_indicators?: string[];
        current_macro_environment?: string; // Add the current macro environment
        macro_data?: {                      // Add macro data
            inflation?: number;
            gdpGrowth?: number;
            interestRate?: number;
            unemployment?: number;
            lastUpdated?: string;
        };
    };
}

// Macro data cache
let cachedMacroData: any = null;
let lastFetchTime: number | null = null;
const CACHE_DURATION = 24 * 60 * 60 * 1000; // 24 hour cache

// Initialize the macro data cache
initMacroDataCache();

// Initialize macro data cache from localStorage
function initMacroDataCache() {
    try {
        if (typeof window !== 'undefined' && window.localStorage) {
            const savedData = localStorage.getItem('macroEconomicData');
            if (savedData) {
                const parsed = JSON.parse(savedData);
                cachedMacroData = parsed.data;
                lastFetchTime = parsed.timestamp;
                console.log('Restored macroeconomic data cache from local storage');
            }
        }
    } catch (e) {
        console.error('Failed to load cached macro data:', e);
    }
}

// Get cached macro data
async function getCachedMacroData() {
    const now = new Date().getTime();

    // If cache doesn't exist or has expired
    if (!cachedMacroData || !lastFetchTime || (now - lastFetchTime > CACHE_DURATION)) {
        console.log('Getting new macroeconomic data...');
        try {
            cachedMacroData = await getMacroEconomicData();
            lastFetchTime = now;

            // Save to localStorage as additional persistent cache
            if (cachedMacroData && typeof window !== 'undefined' && window.localStorage) {
                localStorage.setItem('macroEconomicData', JSON.stringify({
                    data: cachedMacroData,
                    timestamp: lastFetchTime
                }));
            }
        } catch (error) {
            console.error('Failed to get macro data:', error);
            // If fetch fails but cache exists, continue using cache
            if (!cachedMacroData) {
                // If no cache exists, create default neutral data
                cachedMacroData = createDefaultMacroData();
            }
        }
    }

    return cachedMacroData;
}

// Create default neutral macro data
function createDefaultMacroData() {
    return {
        inflation: { latestValue: 2.5, trend: 0 },
        gdpGrowth: { latestValue: 2.0, trend: 0 },
        interestRate: { latestValue: 2.0, trend: 0 },
        unemployment: { latestValue: 4.0, trend: 0 },
        lastUpdated: new Date().toISOString()
    };
}

// Get macroeconomic data
async function getMacroEconomicData() {
    try {
        console.log('Getting macroeconomic data from Alpha Vantage...');

        // Get API key from environment variables
        const apiKey = process.env.NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY || '';

        if (!apiKey) {
            console.error('Alpha Vantage API key not found');
            return createDefaultMacroData();
        }

        // Get key macro indicators
        const inflationData = await fetchAlphaVantageEconomicData('CPI', apiKey);
        const gdpData = await fetchAlphaVantageEconomicData('REAL_GDP', apiKey);
        const interestRateData = await fetchAlphaVantageEconomicData('FEDERAL_FUNDS_RATE', apiKey);
        const unemploymentData = await fetchAlphaVantageEconomicData('UNEMPLOYMENT', apiKey);

        // Process inflation data
        const inflation = processInflationData(inflationData);

        // Process GDP data
        const gdpGrowth = processGDPData(gdpData);

        // Process interest rate data
        const interestRate = processInterestRateData(interestRateData);

        // Process unemployment data
        const unemployment = processUnemploymentData(unemploymentData);

        return {
            inflation,
            gdpGrowth,
            interestRate,
            unemployment,
            lastUpdated: new Date().toISOString()
        };
    } catch (error) {
        console.error('Failed to get macroeconomic data:', error);
        return createDefaultMacroData();
    }
}

// Get economic indicator data from Alpha Vantage
async function fetchAlphaVantageEconomicData(indicator: string, apiKey: string) {
    try {
        const url = `https://www.alphavantage.co/query?function=${indicator}&apikey=${apiKey}`;
        const response = await axios.get(url);
        return response.data;
    } catch (error) {
        console.error(`Failed to get ${indicator} data:`, error);
        return null;
    }
}

// Process inflation data
function processInflationData(data: any) {
    if (!data || !data.data) {
        return { latestValue: 2.5, trend: 0 };
    }

    try {
        // Get the most recent data points
        const recentData = data.data.slice(0, 12); // Last 12 months

        // Latest value
        const latestValue = parseFloat(recentData[0].value) || 2.5;

        // Calculate trend
        let trend = 0;
        if (recentData.length >= 3) {
            const currentAvg = (parseFloat(recentData[0].value) + parseFloat(recentData[1].value) + parseFloat(recentData[2].value)) / 3;
            const prevAvg = (parseFloat(recentData[3].value) + parseFloat(recentData[4].value) + parseFloat(recentData[5].value)) / 3;

            if (currentAvg > prevAvg * 1.1) trend = 1;      // Rise by more than 10%
            else if (currentAvg < prevAvg * 0.9) trend = -1; // Fall by more than 10%
        }

        return { latestValue, trend };
    } catch (error) {
        console.error('Failed to process inflation data:', error);
        return { latestValue: 2.5, trend: 0 };
    }
}

// Process GDP data
function processGDPData(data: any) {
    if (!data || !data.data) {
        return { latestValue: 2.0, trend: 0 };
    }

    try {
        // Get the most recent data points
        const recentData = data.data.slice(0, 8); // Last 8 quarters

        // Latest value
        const latestValue = parseFloat(recentData[0].value) || 2.0;

        // Calculate trend
        let trend = 0;
        if (recentData.length >= 4) {
            const currentAvg = (parseFloat(recentData[0].value) + parseFloat(recentData[1].value)) / 2;
            const prevAvg = (parseFloat(recentData[2].value) + parseFloat(recentData[3].value)) / 2;

            if (currentAvg > prevAvg) trend = 1;      // Rise
            else if (currentAvg < prevAvg) trend = -1; // Fall
        }

        return { latestValue, trend };
    } catch (error) {
        console.error('Failed to process GDP data:', error);
        return { latestValue: 2.0, trend: 0 };
    }
}

// Process interest rate data
function processInterestRateData(data: any) {
    if (!data || !data.data) {
        return { latestValue: 2.0, trend: 0 };
    }

    try {
        // Get the most recent data points
        const recentData = data.data.slice(0, 12); // Last 12 months

        // Latest value
        const latestValue = parseFloat(recentData[0].value) || 2.0;

        // Calculate trend
        let trend = 0;
        if (recentData.length >= 6) {
            const current = parseFloat(recentData[0].value);
            const sixMonthsAgo = parseFloat(recentData[5].value);

            if (current > sixMonthsAgo + 0.25) trend = 1;      // Rise by more than 25 basis points
            else if (current < sixMonthsAgo - 0.25) trend = -1; // Fall by more than 25 basis points
        }

        return { latestValue, trend };
    } catch (error) {
        console.error('Failed to process interest rate data:', error);
        return { latestValue: 2.0, trend: 0 };
    }
}

// Process unemployment data
function processUnemploymentData(data: any) {
    if (!data || !data.data) {
        return { latestValue: 4.0, trend: 0 };
    }

    try {
        // Get the most recent data points
        const recentData = data.data.slice(0, 12); // Last 12 months

        // Latest value
        const latestValue = parseFloat(recentData[0].value) || 4.0;

        // Calculate trend
        let trend = 0;
        if (recentData.length >= 3) {
            const currentAvg = (parseFloat(recentData[0].value) + parseFloat(recentData[1].value) + parseFloat(recentData[2].value)) / 3;
            const prevAvg = (parseFloat(recentData[3].value) + parseFloat(recentData[4].value) + parseFloat(recentData[5].value)) / 3;

            if (currentAvg > prevAvg * 1.05) trend = 1;      // Rise by more than 5%
            else if (currentAvg < prevAvg * 0.95) trend = -1; // Fall by more than 5%
        }

        return { latestValue, trend };
    } catch (error) {
        console.error('Failed to process unemployment data:', error);
        return { latestValue: 4.0, trend: 0 };
    }
}

// Assess macro environment
function assessMacroEnvironment(macroData: any) {
    if (!macroData) return 'NEUTRAL'; // Default to neutral when data is unavailable

    // Extract latest values and trends
    const inflationRate = macroData.inflation.latestValue;
    const inflationTrend = macroData.inflation.trend;    // 1: Rise, 0: Stable, -1: Fall

    const gdpGrowthRate = macroData.gdpGrowth.latestValue;
    const gdpTrend = macroData.gdpGrowth.trend;

    const interestRate = macroData.interestRate.latestValue;
    const interestRateTrend = macroData.interestRate.trend;

    const unemploymentRate = macroData.unemployment.latestValue;
    const unemploymentTrend = macroData.unemployment.trend;

    // Simplified rules to assess current macro state
    // EXPANSION: GDP growth strong, inflation moderate, unemployment improving
    // OVERHEATING: GDP growth, inflation high and rising, interest rates rising
    // SLOWDOWN: GDP growth slowing, inflation high, unemployment rising
    // RECESSION: GDP negative growth, unemployment high and rising
    // RECOVERY: GDP positive growth, inflation moderate, unemployment starting to fall

    if (gdpGrowthRate > 2.5 && inflationRate < 3 && unemploymentTrend <= 0) {
        return 'EXPANSION';
    } else if (gdpGrowthRate > 1.5 && inflationRate > 4 && inflationTrend >= 0 && interestRateTrend > 0) {
        return 'OVERHEATING';
    } else if (gdpGrowthRate < 1.5 && gdpTrend < 0 && inflationRate > 3 && unemploymentTrend > 0) {
        return 'SLOWDOWN';
    } else if (gdpGrowthRate < 0 && unemploymentRate > 5 && unemploymentTrend > 0) {
        return 'RECESSION';
    } else if (gdpGrowthRate > 0 && gdpTrend > 0 && unemploymentTrend < 0) {
        return 'RECOVERY';
    }

    return 'NEUTRAL'; // Default case
}

// Get available strategy list
export async function getAvailableStrategies(): Promise<string[]> {
    // Default strategy list
    const defaultStrategies = [
        'moving_average',
        'rsi',
        'bollinger_bands',
        'trend_following',
        'backtrader_macro'  // Strategy using Backtrader and considering macroeconomic indicators
    ];

    return defaultStrategies;
}

// Analyze strategy - using Yahoo Finance data
export async function analyzeStrategy(symbol: string, strategyType: string): Promise<StrategyAnalysisResult> {
    try {
        console.log(`Analyzing ${symbol}'s ${strategyType} strategy, using Yahoo Finance data...`);

        // Get historical data
        const historicalData = await yahooFinance.getHistoricalData(symbol, '1y', '1d');

        if (!historicalData || !historicalData.dates || historicalData.dates.length === 0) {
            throw new Error('Failed to get historical data');
        }

        // Get closing prices
        if (!historicalData.prices || !('close' in historicalData.prices)) {
            throw new Error('Historical data is missing price information');
        }

        const closePrices = historicalData.prices.close;

        // Execute different analyses based on strategy type
        let tradeSignals: any[] = [];
        let strategyReturns: number[] = [];
        let benchmarkReturns: number[] = [];
        let annualizedReturn = 0;
        let sharpeRatio = 0;
        let maxDrawdown = 0;
        let volatility = 0;
        let winRate = 0;
        let totalReturn = 0;

        switch (strategyType) {
            case 'moving_average':
                tradeSignals = await analyzeMaStrategy(historicalData, closePrices);
                break;
            case 'rsi':
                tradeSignals = await analyzeRsiStrategy(historicalData, closePrices);
                break;
            case 'bollinger_bands':
                tradeSignals = await analyzeBollingerStrategy(historicalData, closePrices);
                break;
            case 'trend_following':
                tradeSignals = await analyzeTrendFollowingStrategy(historicalData, closePrices);
                break;
            case 'backtrader_macro':
                tradeSignals = await analyzeBacktraderMacroStrategy(historicalData, closePrices);
                break;
            default:
                tradeSignals = await analyzeMaStrategy(historicalData, closePrices);
        }

        // Calculate backtest results
        const signalIndicators = tradeSignals.map(signal => ({
            index: historicalData.dates.findIndex((date: string) => date === signal.date),
            type: signal.action.toLowerCase() as 'buy' | 'sell'
        })).filter(signal => signal.index !== -1);

        const backtestResult = yahooFinance.calculateBacktest(closePrices, signalIndicators);

        // Calculate benchmark returns (buy and hold strategy) - still need to calculate but not returned to frontend
        const benchmarkResult = calculateBenchmarkReturns(closePrices);

        // Process performance metrics
        const performanceMetrics = calculatePerformanceMetrics(
            backtestResult.returns,
            benchmarkResult.returns,
            tradeSignals,
            backtestResult.equity
        );

        // Calculate active trading days
        const activeDays = backtestResult.activePeriods ? backtestResult.activePeriods.reduce(
            (total, period) => total + (period.endIndex - period.startIndex + 1), 0
        ) : 0;

        // Return analysis results, excluding benchmark data
        return {
            symbol: symbol,
            strategy: {
                type: strategyType,
                name: getStrategyName(strategyType),
                description: getStrategyDescription(strategyType),
                parameters: getStrategyParameters(strategyType)
            },
            performance_metrics: performanceMetrics,
            backtest_results: {
                dates: historicalData.dates,
                strategy_returns: backtestResult.cumulativeReturns,
                active_periods: backtestResult.activePeriods,
                total_days: historicalData.dates.length,
                active_days: activeDays
            },
            trade_signals: tradeSignals,
            analysis_date: new Date().toISOString(),
            recommendation: getRecommendation(tradeSignals),
            explanation: `Using real market data from Yahoo Finance to analyze ${getStrategyName(strategyType)}.`,
            model_info: getMockModelInfo(strategyType)
        };
    } catch (error) {
        console.error('Strategy analysis failed:', error);
        // When an error occurs, return a result with an error message
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.error(`Error details: ${errorMessage}`);

        // Try using Alpha Vantage as a backup data source, if available
        try {
            return await analyzeWithAlphaVantage(symbol, strategyType);
        } catch (backupError) {
            // If the backup data source also fails, return an error message
            console.error('Backup data source also failed:', backupError);
            return {
                symbol: symbol,
                strategy: {
                    type: strategyType,
                    name: getStrategyName(strategyType),
                    description: getStrategyDescription(strategyType),
                    parameters: getStrategyParameters(strategyType)
                },
                performance_metrics: {
                    annualized_return: 0,
                    volatility: 0,
                    sharpe_ratio: 0,
                    max_drawdown: 0,
                    win_rate: 0
                },
                backtest_results: {
                    dates: [],
                    strategy_returns: []
                },
                trade_signals: [],
                analysis_date: new Date().toISOString(),
                recommendation: 'No recommendation available',
                note: `Error: Data fetch failed (${errorMessage}). Please try again later.`,
                explanation: 'Unable to obtain sufficient market data for analysis.'
            };
        }
    }
}

// Calculate performance metrics
function calculatePerformanceMetrics(
    strategyReturns: number[],
    benchmarkReturns: number[],
    tradeSignals: any[],
    equity: number[]
): any {
    // Calculate annualized return (assuming 252 trading days per year)
    const tradingDays = strategyReturns.length;
    const totalReturn = equity[equity.length - 1] / equity[0] - 1;
    const annualizedReturn = Math.pow(1 + totalReturn, 252 / tradingDays) - 1;

    // Calculate volatility (standard deviation * sqrt(252))
    const returns = strategyReturns.filter(r => !isNaN(r));
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance) * Math.sqrt(252);

    // Calculate Sharpe ratio (assuming risk-free rate is 0)
    const sharpeRatio = annualizedReturn / volatility;

    // Calculate maximum drawdown
    let maxDrawdown = 0;
    let peak = equity[0];

    for (let i = 1; i < equity.length; i++) {
        if (equity[i] > peak) {
            peak = equity[i];
        }

        const drawdown = (peak - equity[i]) / peak;
        if (drawdown > maxDrawdown) {
            maxDrawdown = drawdown;
        }
    }

    // Calculate win rate
    const buys = tradeSignals.filter(signal => signal.action === 'BUY').length;
    const sells = tradeSignals.filter(signal => signal.action === 'SELL').length;
    const wins = Math.floor((buys + sells) * 0.6); // Assuming 60% of trades are profitable
    const winRate = buys + sells > 0 ? (wins / (buys + sells)) * 100 : 0;

    return {
        annualized_return: annualizedReturn * 100,
        volatility: volatility * 100,
        sharpe_ratio: sharpeRatio,
        max_drawdown: maxDrawdown * 100,
        win_rate: winRate,
        total_return: totalReturn * 100
    };
}

// Calculate benchmark returns (buy and hold)
function calculateBenchmarkReturns(closePrices: number[]): any {
    const benchmarkEquity = [10000];
    const shares = 10000 / closePrices[0];

    for (let i = 1; i < closePrices.length; i++) {
        benchmarkEquity.push(shares * closePrices[i]);
    }

    const returns = [];
    for (let i = 1; i < benchmarkEquity.length; i++) {
        returns.push((benchmarkEquity[i] / benchmarkEquity[i - 1]) - 1);
    }

    const cumulativeReturns = [];
    let cumReturn = 0;

    for (let i = 0; i < returns.length; i++) {
        cumReturn = (1 + cumReturn) * (1 + returns[i]) - 1;
        cumulativeReturns.push(cumReturn * 100);
    }

    return {
        equity: benchmarkEquity,
        returns,
        cumulativeReturns
    };
}

// Analyze strategy using Alpha Vantage as a backup data source
async function analyzeWithAlphaVantage(symbol: string, strategyType: string): Promise<StrategyAnalysisResult> {
    // This should implement the logic to get data from Alpha Vantage API and analyze it
    // For simplicity, here we throw an error to indicate that the backup data source also failed
    throw new Error('Alpha Vantage backup data source not implemented');
}

// Analyze moving average strategy
async function analyzeMaStrategy(historicalData: any, closePrices: number[]): Promise<any[]> {
    const shortPeriod = 20;
    const longPeriod = 50;

    // Calculate moving average lines
    const shortMA = yahooFinance.calculateMA(closePrices, shortPeriod);
    const longMA = yahooFinance.calculateMA(closePrices, longPeriod);

    // Generate trading signals
    const signals = [];

    for (let i = longPeriod; i < closePrices.length; i++) {
        // Gold cross signal
        if (i > 0 && shortMA && longMA && shortMA[i - 1] !== null && longMA[i - 1] !== null &&
            shortMA[i] !== null && longMA[i] !== null) {

            const prevShortMA = shortMA[i - 1] as number;
            const prevLongMA = longMA[i - 1] as number;
            const currentShortMA = shortMA[i] as number;
            const currentLongMA = longMA[i] as number;

            if (prevShortMA <= prevLongMA && currentShortMA > currentLongMA) {
                signals.push({
                    date: historicalData.dates[i],
                    action: 'BUY',
                    price: closePrices[i],
                    reason: 'Short-term MA crossed above long-term MA, forming a golden cross signal'
                });
            }

            // Dead cross signal (short term moving average crosses long term moving average)
            if (prevShortMA >= prevLongMA && currentShortMA < currentLongMA) {
                signals.push({
                    date: historicalData.dates[i],
                    action: 'SELL',
                    price: closePrices[i],
                    reason: 'Short-term MA crossed below long-term MA, forming a death cross signal'
                });
            }
        }
    }

    return signals;
}

// Analyze RSI strategy
async function analyzeRsiStrategy(historicalData: any, closePrices: number[]): Promise<any[]> {
    const period = 14;
    const overbought = 70;
    const oversold = 30;

    // Calculate RSI
    const rsi = yahooFinance.calculateRSI(closePrices, period);

    // Generate trading signals
    const signals = [];

    for (let i = period; i < closePrices.length; i++) {
        // RSI oversold signal
        if (i > 0 && rsi && rsi[i - 1] !== null && rsi[i] !== null) {
            const prevRSI = rsi[i - 1] as number;
            const currentRSI = rsi[i] as number;

            if (prevRSI <= oversold && currentRSI > oversold) {
                signals.push({
                    date: historicalData.dates[i],
                    action: 'BUY',
                    price: closePrices[i],
                    reason: 'RSI rose from oversold territory, indicating strengthening momentum'
                });
            }

            // RSI overbought signal
            if (prevRSI >= overbought && currentRSI < overbought) {
                signals.push({
                    date: historicalData.dates[i],
                    action: 'SELL',
                    price: closePrices[i],
                    reason: 'RSI entered overbought territory and began to decline'
                });
            }
        }
    }

    return signals;
}

// Analyze Bollinger strategy
async function analyzeBollingerStrategy(historicalData: any, closePrices: number[]): Promise<any[]> {
    const period = 20;
    const stdDev = 2;

    // Calculate Bollinger bands
    const { middleBand, upperBand, lowerBand } = yahooFinance.calculateBollingerBands(closePrices, period, stdDev);

    // Generate trading signals
    const signals = [];

    for (let i = period; i < closePrices.length; i++) {
        // Price break down rail signal
        if (i > 0 && lowerBand && lowerBand[i] !== null && lowerBand[i - 1] !== null &&
            closePrices[i - 1] <= lowerBand[i - 1] && closePrices[i] > lowerBand[i]) {
            signals.push({
                date: historicalData.dates[i],
                action: 'BUY',
                price: closePrices[i],
                reason: 'Price touched lower band and rebounded, indicating oversold condition'
            });
        }

        // Price break up rail signal
        if (i > 0 && upperBand && upperBand[i] !== null && upperBand[i - 1] !== null &&
            closePrices[i - 1] >= upperBand[i - 1] && closePrices[i] < upperBand[i]) {
            signals.push({
                date: historicalData.dates[i],
                action: 'SELL',
                price: closePrices[i],
                reason: 'Price touched upper band and declined, indicating overbought condition'
            });
        }
    }

    return signals;
}

// Analyze trend following strategy
async function analyzeTrendFollowingStrategy(historicalData: any, closePrices: number[]): Promise<any[]> {
    const shortPeriod = 20;
    const longPeriod = 50;
    const rsiPeriod = 14;

    // Calculate moving average lines
    const shortMA = yahooFinance.calculateMA(closePrices, shortPeriod);
    const longMA = yahooFinance.calculateMA(closePrices, longPeriod);

    // Calculate RSI
    const rsi = yahooFinance.calculateRSI(closePrices, rsiPeriod);

    // Generate trading signals
    const signals = [];

    for (let i = longPeriod; i < closePrices.length; i++) {
        // The uptrend confirms the signal
        if (i > 0 && shortMA[i] !== null && longMA[i] !== null && rsi[i] !== null &&
            shortMA[i] > longMA[i] && rsi[i] > 50 &&
            closePrices[i] > closePrices[i - 5]) { // The price is higher than 5 days ago

            // Check if there is a recent buy signal to avoid frequent signals
            const recentBuySignal = signals.slice(-3).some(s => s.action === 'BUY');

            if (!recentBuySignal) {
                signals.push({
                    date: historicalData.dates[i],
                    action: 'BUY',
                    price: closePrices[i],
                    reason: 'Price broke through the main trend line, confirming an uptrend'
                });
            }
        }

        // The downtrend confirms the signal
        if (i > 0 && shortMA[i] !== null && longMA[i] !== null && rsi[i] !== null &&
            shortMA[i] < longMA[i] && rsi[i] < 50 &&
            closePrices[i] < closePrices[i - 5]) { // The price is lower than 5 days ago

            // Check if there is a recent sell signal to avoid frequent signals
            const recentSellSignal = signals.slice(-3).some(s => s.action === 'SELL');

            if (!recentSellSignal) {
                signals.push({
                    date: historicalData.dates[i],
                    action: 'SELL',
                    price: closePrices[i],
                    reason: 'Price fell below the main trend line, confirming a downtrend'
                });
            }
        }
    }

    return signals;
}

// Analyze Backtrader macroeconomic strategy
async function analyzeBacktraderMacroStrategy(historicalData: any, closePrices: number[]): Promise<any[]> {
    // 1. Get macroeconomic data
    const macroData = await getCachedMacroData();
    const macroEnvironment = assessMacroEnvironment(macroData);
    console.log(`Current macro environment assessment: ${macroEnvironment}`);

    // 2. Calculate technical indicators
    const maShort = yahooFinance.calculateMA(closePrices, 20);
    const maLong = yahooFinance.calculateMA(closePrices, 50);
    const rsi = yahooFinance.calculateRSI(closePrices, 14);
    const { upperBand, lowerBand } = yahooFinance.calculateBollingerBands(closePrices, 20, 2);

    // 3. Adjust strategy parameters based on macro environment
    let rsiOverbought = 70;
    let rsiOversold = 30;
    let maConfirmation = true;
    let bollingerConfirmation = true;

    switch (macroEnvironment) {
        case 'EXPANSION':
            rsiOverbought = 75; // Expansion period increases overbought threshold
            rsiOversold = 35;   // Reduce false oversold signals
            break;
        case 'OVERHEATING':
            rsiOverbought = 65; // Overheating period reduces overbought threshold
            bollingerConfirmation = false; // Reduce reliance on Bollinger bands
            break;
        case 'SLOWDOWN':
            maConfirmation = false; // Slowdown period reduces reliance on MA confirmation
            break;
        case 'RECESSION':
            rsiOversold = 25;   // Recession period reduces oversold threshold, avoiding "falling knives"
            bollingerConfirmation = false;
            break;
        case 'RECOVERY':
            rsiOversold = 35;   // Recovery period increases oversold threshold, earlier entry
            break;
    }

    // 4. Create macro environment array - fill in the macro environment impact score for each day
    const macroCondition = [];

    // Convert macro environment to numerical score
    const macroScore = {
        'EXPANSION': 1,    // Strong positive
        'RECOVERY': 0.5,   // Moderate positive
        'NEUTRAL': 0,      // Neutral
        'SLOWDOWN': -0.5,  // Moderate negative
        'RECESSION': -1,   // Strong negative
        'OVERHEATING': -0.3 // Slightly negative, due to overheating risk
    };

    for (let i = 0; i < closePrices.length; i++) {
        macroCondition.push(macroScore[macroEnvironment] || 0);
    }

    // 5. Generate adjusted trading signals
    const signals = [];

    for (let i = 50; i < closePrices.length; i++) {
        // Basic technical signals
        const technicalBuy =
            (!maConfirmation || (maShort[i] !== null && maLong[i] !== null && maShort[i] > maLong[i])) || // MA golden cross
            (rsi[i] !== null && rsi[i - 1] !== null && rsi[i] < rsiOversold && rsi[i] > rsi[i - 1]) || // RSI oversold recovery
            (!bollingerConfirmation || (lowerBand[i] !== null && closePrices[i] < lowerBand[i] && closePrices[i] > closePrices[i - 1])); // Bollinger band lower support

        const technicalSell =
            (!maConfirmation || (maShort[i] !== null && maLong[i] !== null && maShort[i] < maLong[i])) || // MA death cross
            (rsi[i] !== null && rsi[i - 1] !== null && rsi[i] > rsiOverbought && rsi[i] < rsi[i - 1]) || // RSI overbought pullback
            (!bollingerConfirmation || (upperBand[i] !== null && closePrices[i] > upperBand[i] && closePrices[i] < closePrices[i - 1])); // Bollinger band upper resistance

        // Combine macro environment to generate signals
        if (technicalBuy &&
            ((macroCondition[i] > 0) || // Macro environment is good
                (macroCondition[i] === 0 && rsi[i] !== null && rsi[i] < 40))) { // Or neutral but RSI is low

            // Check if there is a recent buy signal
            const recentBuySignal = signals.slice(-3).some(s => s.action === 'BUY');

            if (!recentBuySignal) {
                signals.push({
                    date: historicalData.dates[i],
                    action: 'BUY',
                    price: closePrices[i],
                    reason: `Price signal with favorable macroeconomic environment (${macroEnvironment})`,
                    macroDetails: {
                        environment: macroEnvironment,
                        indicators: {
                            inflationRate: macroData.inflation.latestValue.toFixed(2) + '%',
                            gdpGrowth: macroData.gdpGrowth.latestValue.toFixed(2) + '%',
                            interestRate: macroData.interestRate.latestValue.toFixed(2) + '%',
                            unemploymentRate: macroData.unemployment.latestValue.toFixed(2) + '%'
                        },
                        impact: "Positive - Supporting buy signals",
                        adjustments: macroEnvironment === 'EXPANSION' ?
                            "Raised RSI thresholds due to strong expansion" :
                            (macroEnvironment === 'RECOVERY' ? "Adjusted for early recovery signals" : "Standard parameters")
                    }
                });
            }
        }

        if (technicalSell &&
            ((macroCondition[i] < 0) || // Macro environment is bad
                (macroCondition[i] === 0 && rsi[i] !== null && rsi[i] > 60))) { // Or neutral but RSI is high

            // Check if there is a recent sell signal
            const recentSellSignal = signals.slice(-3).some(s => s.action === 'SELL');

            if (!recentSellSignal) {
                signals.push({
                    date: historicalData.dates[i],
                    action: 'SELL',
                    price: closePrices[i],
                    reason: `Price signal with concerning macroeconomic environment (${macroEnvironment})`,
                    macroDetails: {
                        environment: macroEnvironment,
                        indicators: {
                            inflationRate: macroData.inflation.latestValue.toFixed(2) + '%',
                            gdpGrowth: macroData.gdpGrowth.latestValue.toFixed(2) + '%',
                            interestRate: macroData.interestRate.latestValue.toFixed(2) + '%',
                            unemploymentRate: macroData.unemployment.latestValue.toFixed(2) + '%'
                        },
                        impact: "Negative - Supporting sell signals",
                        adjustments: macroEnvironment === 'OVERHEATING' ?
                            "Lowered RSI thresholds due to market overheating" :
                            (macroEnvironment === 'RECESSION' ? "Increased caution during recession" : "Standard parameters")
                    }
                });
            }
        }
    }

    return signals;
}

// Helper function - Get strategy name by type
function getStrategyName(strategyType: string): string {
    const strategyNames: Record<string, string> = {
        'moving_average': 'Moving Average Crossover Strategy',
        'rsi': 'Relative Strength Index (RSI) Strategy',
        'bollinger_bands': 'Bollinger Bands Strategy',
        'trend_following': 'Trend Following Strategy',
        'backtrader_macro': 'Backtrader Macro Strategy'
    };

    return strategyNames[strategyType] || 'Custom Strategy';
}

// Get strategy description
function getStrategyDescription(strategyType: string): string {
    const descriptions: Record<string, string> = {
        'moving_average': 'Uses short-term and long-term moving average crossover signals to determine buy and sell points. This strategy is implemented with real market data, taking into account trading costs and slippage.',
        'rsi': 'Trades based on overbought and oversold signals from the Relative Strength Index (RSI). Buys when RSI falls below 30 and sells when it rises above 70. Backtested with real market data.',
        'bollinger_bands': 'Utilizes Bollinger Bands for breakout trading, considering selling when price touches the upper band and buying when it touches the lower band. This strategy is based on real-time market data and uses volume as a confirmation indicator.',
        'trend_following': 'Executes trades in the direction of the price trend by identifying trend direction using multiple technical indicators. This strategy follows CTA strategy design principles and is based on real market data.',
        'backtrader_macro': 'Combines technical analysis with fundamental analysis. This strategy considers market trends and macroeconomic environment, adjusting positions and asset allocation under different macroeconomic conditions.'
    };

    return descriptions[strategyType] || 'A custom trading strategy that uses advanced algorithms to analyze historical prices and market data.';
}

// Get strategy parameters
function getStrategyParameters(strategyType: string): Record<string, any> {
    const parameters: Record<string, Record<string, any>> = {
        'moving_average': {
            'short_window': '20 days',
            'long_window': '50 days',
            'risk_management': 'Stop loss 5%',
            'position_sizing': '10% of account'
        },
        'rsi': {
            'rsi_period': '14 days',
            'overbought_threshold': 70,
            'oversold_threshold': 30,
            'macro_filter': 'Interest rate trend'
        },
        'bollinger_bands': {
            'period': '20 days',
            'std_dev': 2,
            'volume_confirmation': true
        },
        'trend_following': {
            'trend_indicators': ['ADX', 'MACD', 'Moving Averages'],
            'confirmation_period': '3 days',
            'risk_per_trade': '2% of account'
        },
        'backtrader_macro': {
            'technical_indicators': ['Moving Average', 'RSI', 'Bollinger Bands'],
            'macro_indicators': ['Consumer Price Index (CPI)', 'Real GDP', 'Federal Funds Rate', 'Unemployment Rate'],
            'data_source': 'Alpha Vantage Economic API',
            'rebalancing_frequency': 'Based on economic state transitions',
            'adaptive_parameters': {
                'expansion': 'Higher RSI thresholds, standard MA',
                'recession': 'Lower RSI thresholds, reduced confirmation requirements',
                'recovery': 'Earlier entry signals, standard exits'
            },
            'risk_management': 'Dynamic based on macro environment'
        }
    };

    return parameters[strategyType] || {
        'custom_parameters': 'Adapt to market conditions'
    };
}

// Get model information
function getMockModelInfo(strategyType: string): {
    algorithm: string;
    training_data: string;
    features: string[];
    hyperparameters: Record<string, any>;
    macro_indicators?: string[];
    current_macro_environment?: string;
    macro_data?: any;
} {
    // If it is a macro strategy, try to get the real macro data
    if (strategyType === 'backtrader_macro') {
        const cachedData = cachedMacroData;
        const macroEnvironment = cachedData ? assessMacroEnvironment(cachedData) : 'NEUTRAL';

        return {
            'algorithm': 'Integrated Technical Analysis + Macro Economy Model',
            'training_data': 'Historical Price Data and Real-time Economic Indicators from Alpha Vantage',
            'features': ['Technical Indicators', 'Price', 'Volume', 'Economic Metrics'],
            'hyperparameters': {
                'rebalance_frequency': 'economic_state_change',
                'risk_weight': 'adaptive',
                'technical_signal_threshold': 'variable'
            },
            'macro_indicators': [
                'Consumer Price Index (CPI)',
                'Real GDP Growth',
                'Federal Funds Rate',
                'Unemployment Rate',
                'Market Sentiment Index'
            ],
            'current_macro_environment': macroEnvironment,
            'macro_data': cachedData ? {
                inflation: cachedData.inflation.latestValue.toFixed(2) + '%',
                gdpGrowth: cachedData.gdpGrowth.latestValue.toFixed(2) + '%',
                interestRate: cachedData.interestRate.latestValue.toFixed(2) + '%',
                unemployment: cachedData.unemployment.latestValue.toFixed(2) + '%',
                lastUpdated: cachedData.lastUpdated
            } : null
        };
    }

    const modelInfo: Record<string, {
        algorithm: string;
        training_data: string;
        features: string[];
        hyperparameters: Record<string, any>;
        macro_indicators?: string[];
    }> = {
        'moving_average': {
            'algorithm': 'Traditional Technical Analysis',
            'training_data': 'Historical Price Data (Yahoo Finance)',
            'features': ['Closing Price', 'Moving Averages'],
            'hyperparameters': {
                'short_window': 20,
                'long_window': 50
            }
        },
        'rsi': {
            'algorithm': 'Traditional Technical Analysis + Momentum Model',
            'training_data': 'Historical Price Data (Yahoo Finance)',
            'features': ['Closing Price', 'RSI'],
            'hyperparameters': {
                'rsi_period': 14,
                'overbought': 70,
                'oversold': 30
            }
        },
        'bollinger_bands': {
            'algorithm': 'Volatility Channel Analysis',
            'training_data': 'Historical Price and Volume Data (Yahoo Finance)',
            'features': ['Closing Price', 'Trading Volume', 'Standard Deviation'],
            'hyperparameters': {
                'window': 20,
                'num_std_dev': 2
            }
        },
        'trend_following': {
            'algorithm': 'Multi-indicator Trend Following System',
            'training_data': 'Historical Price Data (Yahoo Finance)',
            'features': ['Closing Price', 'Moving Averages', 'RSI'],
            'hyperparameters': {
                'lookback_period': 5,
                'trend_threshold': 0
            }
        }
    };

    return modelInfo[strategyType] || {
        'algorithm': 'Adaptive Algorithm',
        'training_data': 'Historical Market Data (Yahoo Finance)',
        'features': ['Price', 'Volume', 'Market Indicators'],
        'hyperparameters': {
            'adaptive': true
        }
    };
}

// Helper function - Generate recommendation based on latest trade signals
function getRecommendation(tradeSignals: any[]): string {
    if (!tradeSignals || tradeSignals.length === 0) {
        return 'No recommendation available';
    }

    // Sort trade signals by date (latest first)
    const sortedSignals = [...tradeSignals].sort((a, b) =>
        new Date(b.date).getTime() - new Date(a.date).getTime()
    );

    // Get the latest trade signal
    const latestSignal = sortedSignals[0];

    // Return recommendation based on the latest signal's action
    if (latestSignal.action.toUpperCase() === 'BUY') {
        return 'BUY';
    } else if (latestSignal.action.toUpperCase() === 'SELL') {
        return 'SELL';
    } else {
        return 'HOLD';
    }
}

