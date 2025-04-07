'use client';

import React, { useState, useEffect, useContext } from 'react';
import { FaChartPie, FaQuestionCircle, FaCheckCircle, FaTimesCircle, FaExclamationTriangle, FaInfoCircle } from 'react-icons/fa';
import ExplanationModal from './ExplanationModal';
import axios from 'axios';

// Rating types
type RatingType = 'STRONG BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG SELL' | null;

// Rating color mapping
const ratingColorMap = {
    'STRONG BUY': 'bg-green-600',
    'BUY': 'bg-green-500',
    'HOLD': 'bg-yellow-500',
    'SELL': 'bg-red-500',
    'STRONG SELL': 'bg-red-600',
};

// Rating icon mapping
const ratingIconMap = {
    'STRONG BUY': <FaCheckCircle className="mr-2" />,
    'BUY': <FaCheckCircle className="mr-2" />,
    'HOLD': <FaExclamationTriangle className="mr-2" />,
    'SELL': <FaTimesCircle className="mr-2" />,
    'STRONG SELL': <FaTimesCircle className="mr-2" />,
};

// Explanation details modal type
interface ExplanationDetails {
    featureImportance: { name: string, importance: number }[];
    modelScore: number;
    reasoning: string;
    marketData?: {
        volatility: {
            vix: number;
            isHigh: boolean;
        };
        marketTrend: string;
        date: string;
    };
}

// Add analysis data interface definition
interface SentimentAnalysisData {
    overallSentiment: string;
    sentimentScore: string;
    confidence?: string;
    details?: any[];
    topPositiveFactors?: string[];
    topNegativeFactors?: string[];
    analysisDate?: string;
    recommendation?: string;
    apiCallSuccess?: boolean;
    isSimulated?: boolean;
    apiSuccessRate?: string;
}

interface FinancialAnalysisData {
    financialMetrics?: Record<string, any>;
    keyMetrics?: {
        revenueGrowth: number;
        netIncomeGrowth: number;
        currentRatio: number;
        debtToEquity: number;
        returnOnEquity: number;
        [key: string]: any;
    };
    recommendation?: string;
    analysis?: string;
    summary?: string;
    strengths?: string;
    weaknesses?: string;
    // Other financial analysis related fields
}

interface StrategyAnalysisData {
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
    backtest_results?: {
        dates: string[];
        strategy_returns: number[];
        benchmark_returns: number[];
    };
    trade_signals?: any[];
    analysis_date?: string;
    recommendation: string;
    // Other strategy analysis related fields
}

interface FinalRecommendationProps {
    stockSymbol: string;
}

const FinalRecommendation: React.FC<FinalRecommendationProps> = ({ stockSymbol }) => {
    const [rating, setRating] = useState<RatingType>(null);
    const [loading, setLoading] = useState(false);
    const [showExplanation, setShowExplanation] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [showDetails, setShowDetails] = useState(false);
    const [explanationDetails, setExplanationDetails] = useState<ExplanationDetails | null>(null);
    const [loadingDetails, setLoadingDetails] = useState(false);
    const [dataComplete, setDataComplete] = useState({
        sentiment: false,
        financial: false,
        strategy: false
    });

    // Modify analysis data state types
    const [analysisData, setAnalysisData] = useState<{
        sentiment: SentimentAnalysisData | null;
        financial: FinancialAnalysisData | null;
        strategy: StrategyAnalysisData | null;
    }>({
        sentiment: null,
        financial: null,
        strategy: null
    });

    // Check if there's saved analysis data in localStorage
    useEffect(() => {
        // Reset data completion status
        setDataComplete({
            sentiment: false,
            financial: false,
            strategy: false
        });

        // Reset analysis data
        setAnalysisData({
            sentiment: null,
            financial: null,
            strategy: null
        });

        // Check if there's sentiment analysis data for the current stock
        const sentimentData = localStorage.getItem(`sentimentAnalysis_${stockSymbol}`);
        if (sentimentData) {
            try {
                const parsedData = JSON.parse(sentimentData);
                if (parsedData && parsedData.overallSentiment) {
                    setAnalysisData(prev => ({ ...prev, sentiment: parsedData }));
                    setDataComplete(prev => ({ ...prev, sentiment: true }));
                }
            } catch (error) {
                console.error('Error parsing sentiment data from localStorage', error);
            }
        }

        // Check if there's research analysis data for the current stock
        const researchData = localStorage.getItem(`researchAnalysis_${stockSymbol}`);
        if (researchData) {
            try {
                const parsedData = JSON.parse(researchData);
                if (parsedData && (parsedData.keyMetrics || parsedData.financialMetrics)) {
                    setAnalysisData(prev => ({ ...prev, financial: parsedData }));
                    setDataComplete(prev => ({ ...prev, financial: true }));
                }
            } catch (error) {
                console.error('Error parsing research data from localStorage', error);
            }
        }

        // Check if there's strategy analysis data for the current stock
        const strategyData = localStorage.getItem(`strategyAnalysis_${stockSymbol}`);
        if (strategyData) {
            try {
                const parsedData = JSON.parse(strategyData);
                if (parsedData && parsedData.performance_metrics) {
                    setAnalysisData(prev => ({ ...prev, strategy: parsedData }));
                    setDataComplete(prev => ({ ...prev, strategy: true }));
                }
            } catch (error) {
                console.error('Error parsing strategy data from localStorage', error);
            }
        }

        // Add event listeners to ensure we only process analysis events for the current stock
        const handleSentimentAnalysisComplete = (e: Event) => {
            const customEvent = e as CustomEvent;
            const data = customEvent.detail;

            // Accept data if it has no symbol property or if the symbol matches the current stock
            if (!data.symbol || data.symbol === stockSymbol) {
                // Add symbol property
                const dataWithSymbol = { ...data, symbol: stockSymbol };

                setAnalysisData(prev => ({ ...prev, sentiment: data }));
                setDataComplete(prev => ({ ...prev, sentiment: true }));

                // Add stock identifier when saving to localStorage
                localStorage.setItem(`sentimentAnalysis_${stockSymbol}`, JSON.stringify(dataWithSymbol));
            }
        };

        const handleResearchAnalysisComplete = (e: Event) => {
            const customEvent = e as CustomEvent;
            const data = customEvent.detail;

            // Accept data if it has no symbol property or if the symbol matches the current stock
            if (!data.symbol || data.symbol === stockSymbol) {
                // Add symbol property
                const dataWithSymbol = { ...data, symbol: stockSymbol };

                setAnalysisData(prev => ({ ...prev, financial: data }));
                setDataComplete(prev => ({ ...prev, financial: true }));

                // Add stock identifier when saving to localStorage
                localStorage.setItem(`researchAnalysis_${stockSymbol}`, JSON.stringify(dataWithSymbol));
            }
        };

        const handleStrategyAnalysisComplete = (e: Event) => {
            const customEvent = e as CustomEvent;
            const data = customEvent.detail;

            // Accept data if it has no symbol property or if the symbol matches the current stock
            if (!data.symbol || data.symbol === stockSymbol) {
                // Add symbol property
                const dataWithSymbol = { ...data, symbol: stockSymbol };

                setAnalysisData(prev => ({ ...prev, strategy: data }));
                setDataComplete(prev => ({ ...prev, strategy: true }));

                // Add stock identifier when saving to localStorage
                localStorage.setItem(`strategyAnalysis_${stockSymbol}`, JSON.stringify(dataWithSymbol));
            }
        };

        // Register event listeners
        window.addEventListener('sentimentAnalysisComplete', handleSentimentAnalysisComplete);
        window.addEventListener('researchAnalysisComplete', handleResearchAnalysisComplete);
        window.addEventListener('strategyAnalysisComplete', handleStrategyAnalysisComplete);

        // Remove event listeners on component unmount
        return () => {
            window.removeEventListener('sentimentAnalysisComplete', handleSentimentAnalysisComplete);
            window.removeEventListener('researchAnalysisComplete', handleResearchAnalysisComplete);
            window.removeEventListener('strategyAnalysisComplete', handleStrategyAnalysisComplete);
        };
    }, [stockSymbol]); // Add stockSymbol as a dependency so this runs when the stock changes

    // Generate investment recommendation
    const generateRecommendation = async () => {
        // Check if all necessary data has been obtained
        if (!dataComplete.sentiment || !dataComplete.financial || !dataComplete.strategy) {
            setErrorMessage('Please complete sentiment analysis, financial analysis, and strategy analysis first');
            return;
        }

        setLoading(true);
        setErrorMessage(null);

        try {
            // Build prompt to send to GPT
            const { sentiment, financial, strategy } = analysisData;

            // Get overall sentiment from sentiment analysis
            const overallSentiment = sentiment?.overallSentiment || 'Neutral';

            // Get core financial metrics from financial analysis - corrected to proper property path
            const financialMetrics = financial?.keyMetrics ? {
                revenueGrowth: financial.keyMetrics.revenueGrowth,
                netIncomeGrowth: financial.keyMetrics.netIncomeGrowth,
                currentRatio: financial.keyMetrics.currentRatio,
                debtToEquity: financial.keyMetrics.debtToEquity,
                returnOnEquity: financial.keyMetrics.returnOnEquity
            } : {};

            // Get strategy performance data from strategy analysis
            const strategyData = strategy;
            const strategyType = strategyData?.strategy?.type || 'unknown';
            const totalReturn = strategyData?.performance_metrics?.total_return || 0;
            const strategyRecommendation = strategyData?.recommendation || 'No recommendation';

            // Call backend API to generate final recommendation
            const response = await axios.post('/api/generate-recommendation', {
                stockSymbol,
                sentimentData: {
                    overallSentiment,
                    sentimentScore: sentiment?.sentimentScore || '0.00'
                },
                financialData: financialMetrics,
                strategyData: {
                    type: strategyType,
                    totalReturn,
                    recommendation: strategyRecommendation
                }
            });

            // Process response
            const result = response.data;

            // Map score from -1 to 1 to rating
            setRating(mapScoreToRating(result.score));

            // Store explanatory data for detailed display
            setExplanationDetails({
                featureImportance: result.featureImportance || [],
                modelScore: result.score,
                reasoning: result.reasoning || '',
                marketData: result.marketData  // Include market data
            });
        } catch (error) {
            console.error('Error generating recommendation:', error);
            setErrorMessage('Failed to generate recommendation, please try again later');
        } finally {
            setLoading(false);
        }
    };

    // Map score to rating
    const mapScoreToRating = (score: number): RatingType => {
        if (score > 0.7) return 'STRONG BUY';
        if (score > 0.3) return 'BUY';
        if (score >= -0.3) return 'HOLD';
        if (score >= -0.7) return 'SELL';
        return 'STRONG SELL';
    };

    // View recommendation details
    const viewRecommendationDetails = async () => {
        if (!rating) return;

        setLoadingDetails(true);

        try {
            // This should display the existing explanationDetails
            // In a real project, more detailed explanation data might be fetched from the backend
            setShowDetails(true);
        } catch (error) {
            console.error('Error getting recommendation details:', error);
        } finally {
            setLoadingDetails(false);
        }
    };

    // Provide different colors for feature importance
    const getBarColor = (featureName: string) => {
        switch (featureName) {
            case 'Sentiment Analysis':
            case 'Sentiment':
                return 'bg-blue-600';
            case 'Financial Health':
                return 'bg-green-600';
            case 'Strategy Performance':
                return 'bg-purple-600';
            default:
                return 'bg-gray-600';
        }
    };

    return (
        <div className="card">
            <h2 className="text-xl font-bold mb-6 flex items-center">
                <FaChartPie className="mr-2 text-primary" />
                Final Recommendation
                <button
                    onClick={() => setShowExplanation(true)}
                    className="ml-2 text-gray-500 hover:text-primary"
                    aria-label="View recommendation explanation"
                >
                    <FaQuestionCircle size={16} />
                </button>
            </h2>

            {/* Data completeness indicator */}
            <div className="mb-4 p-3 bg-gray-50 rounded-md">
                <h3 className="text-sm font-medium text-gray-600 mb-2">Required Analysis:</h3>
                <div className="grid grid-cols-3 gap-2">
                    <div className={`p-2 rounded-md ${dataComplete.sentiment ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                        <div className="flex items-center">
                            {dataComplete.sentiment ? <FaCheckCircle className="mr-1" /> : <FaTimesCircle className="mr-1" />}
                            <span>Sentiment</span>
                        </div>
                    </div>
                    <div className={`p-2 rounded-md ${dataComplete.financial ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                        <div className="flex items-center">
                            {dataComplete.financial ? <FaCheckCircle className="mr-1" /> : <FaTimesCircle className="mr-1" />}
                            <span>Financial</span>
                        </div>
                    </div>
                    <div className={`p-2 rounded-md ${dataComplete.strategy ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                        <div className="flex items-center">
                            {dataComplete.strategy ? <FaCheckCircle className="mr-1" /> : <FaTimesCircle className="mr-1" />}
                            <span>Strategy</span>
                        </div>
                    </div>
                </div>
            </div>

            <div className="text-center py-4">
                {rating ? (
                    <div className="mb-6">
                        <div className={`inline-flex items-center px-4 py-2 rounded-full text-white font-bold ${ratingColorMap[rating]}`}>
                            {ratingIconMap[rating]}
                            {rating}
                        </div>
                        <p className="mt-4 text-gray-600">
                            Based on our comprehensive analysis, we give {stockSymbol} the above rating.
                        </p>

                        {/* View details button */}
                        <button
                            onClick={viewRecommendationDetails}
                            className="mt-4 px-4 py-2 text-primary border border-primary rounded hover:bg-primary hover:text-white transition-colors"
                            disabled={loadingDetails}
                        >
                            {loadingDetails ? 'Loading...' : 'View Details'}
                        </button>
                    </div>
                ) : (
                    <p className="text-gray-600 mb-6">
                        Click the button below to generate an investment recommendation based on comprehensive analysis.
                    </p>
                )}

                {errorMessage && (
                    <div className="mb-4 p-3 bg-red-100 text-red-700 rounded">
                        <div className="flex items-center">
                            <FaExclamationTriangle className="mr-2" />
                            <span>{errorMessage}</span>
                        </div>
                    </div>
                )}

                <button
                    onClick={generateRecommendation}
                    disabled={loading || !dataComplete.sentiment || !dataComplete.financial || !dataComplete.strategy}
                    className={`btn btn-primary px-6 ${(loading || !dataComplete.sentiment || !dataComplete.financial || !dataComplete.strategy) ? 'opacity-70 cursor-not-allowed' : ''}`}
                >
                    {loading ? 'Generating...' : 'Generate Recommendation'}
                </button>
            </div>

            {/* Details modal */}
            {showDetails && explanationDetails && (
                <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
                    <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-xl font-bold">Recommendation Explanation</h3>
                            <button
                                onClick={() => setShowDetails(false)}
                                className="text-gray-500 hover:text-gray-700"
                            >
                                <FaTimesCircle size={20} />
                            </button>
                        </div>

                        {/* Market data area - moved to top and redesigned for emphasis */}
                        {explanationDetails.marketData && (
                            <div className="mb-6 border border-gray-200 rounded-lg overflow-hidden">
                                <div className="bg-gray-100 p-3 border-b border-gray-200">
                                    <h4 className="font-semibold text-lg flex items-center">
                                        <FaInfoCircle className="mr-2 text-blue-500" />
                                        Market Context
                                    </h4>
                                    <p className="text-sm text-gray-600 mt-1">
                                        Market conditions influence how different factors are weighted in our analysis
                                    </p>
                                </div>
                                <div className="p-4">
                                    <div className="grid grid-cols-2 gap-4">
                                        <div className="bg-white p-3 rounded-lg border border-gray-200">
                                            <div className="text-sm text-gray-500 mb-1">Market Date</div>
                                            <div className="font-medium">{explanationDetails.marketData.date}</div>
                                        </div>
                                        <div className="bg-white p-3 rounded-lg border border-gray-200">
                                            <div className="text-sm text-gray-500 mb-1">Market Trend</div>
                                            <div className={`font-medium ${explanationDetails.marketData.marketTrend === 'Bullish'
                                                ? 'text-green-600'
                                                : explanationDetails.marketData.marketTrend === 'Bearish'
                                                    ? 'text-red-600'
                                                    : 'text-yellow-600'
                                                }`}>
                                                {explanationDetails.marketData.marketTrend}
                                            </div>
                                        </div>
                                        <div className="bg-white p-3 rounded-lg border border-gray-200">
                                            <div className="text-sm text-gray-500 mb-1">VIX Index</div>
                                            <div className="font-medium">{explanationDetails.marketData.volatility.vix.toFixed(2)}</div>
                                        </div>
                                        <div className="bg-white p-3 rounded-lg border border-gray-200">
                                            <div className="text-sm text-gray-500 mb-1">Volatility</div>
                                            <div className={`font-medium ${explanationDetails.marketData.volatility.isHigh
                                                ? 'text-red-600'
                                                : 'text-green-600'
                                                }`}>
                                                {explanationDetails.marketData.volatility.isHigh ? 'High' : 'Normal'}
                                            </div>
                                        </div>
                                    </div>
                                    <div className="mt-3 text-sm text-gray-600 bg-blue-50 p-3 rounded border border-blue-100">
                                        <div className="font-medium text-blue-700 mb-1">How market conditions affected this analysis:</div>
                                        <ul className="list-disc pl-5 space-y-1">
                                            {explanationDetails.marketData.volatility.isHigh && (
                                                <li>High volatility increased the weight of Strategy Performance</li>
                                            )}
                                            {explanationDetails.marketData.marketTrend === 'Bullish' && (
                                                <li>Bullish market trend increased the weight of Sentiment Analysis</li>
                                            )}
                                            {explanationDetails.marketData.marketTrend === 'Bearish' && (
                                                <li>Bearish market trend increased the weight of Financial Health</li>
                                            )}
                                            {/* Default text for neutral/normal market conditions */}
                                            {(!explanationDetails.marketData.volatility.isHigh && explanationDetails.marketData.marketTrend === 'Neutral') && (
                                                <li>Normal market volatility and neutral trend resulted in balanced weight distribution</li>
                                            )}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        )}

                        <div className="mb-4">
                            <h4 className="font-semibold text-lg mb-2">Model Score</h4>
                            <div className="p-3 bg-gray-50 rounded">
                                <div className="flex items-center">
                                    <div className="font-mono text-xl">{explanationDetails.modelScore.toFixed(2)}</div>
                                    <div className="ml-2 text-sm text-gray-500">(-1 to 1 scale)</div>
                                </div>
                            </div>
                        </div>

                        <div className="mb-6">
                            <h4 className="font-semibold text-lg mb-2">Feature Weights</h4>
                            <div className="p-4 bg-gray-50 rounded">
                                {explanationDetails.featureImportance.length > 0 ? (
                                    <div>
                                        {explanationDetails.featureImportance.map((feature, index) => (
                                            <div key={index} className="mb-3">
                                                <div className="flex justify-between mb-1">
                                                    <span className="font-medium">{feature.name}</span>
                                                    <span className="font-mono">{(feature.importance * 100).toFixed(1)}%</span>
                                                </div>
                                                <div className="w-full bg-gray-200 rounded-full h-3">
                                                    <div
                                                        className={`h-3 rounded-full ${getBarColor(feature.name)}`}
                                                        style={{ width: `${feature.importance * 100}%` }}
                                                    ></div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <p className="text-gray-500">No feature importance data available</p>
                                )}
                            </div>
                        </div>

                        <div className="mb-6">
                            <h4 className="font-semibold text-lg mb-2">Reasoning</h4>
                            <div className="p-4 bg-gray-50 rounded whitespace-pre-line">
                                {explanationDetails.reasoning}
                            </div>
                        </div>

                        <div className="mt-4 flex justify-end">
                            <button
                                onClick={() => setShowDetails(false)}
                                className="px-4 py-2 bg-primary text-white rounded hover:bg-primary-dark"
                            >
                                Close
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Explanation Modal */}
            <ExplanationModal
                isOpen={showExplanation}
                onClose={() => setShowExplanation(false)}
                title="Investment Recommendation Explanation"
                content={
                    <div>
                        <p className="mb-4">
                            Our investment recommendation is based on a comprehensive analysis of multiple factors, including:
                        </p>
                        <ul className="list-disc pl-5 space-y-2 mb-4">
                            <li>
                                <strong>Sentiment Analysis:</strong> Evaluating the overall market sentiment towards the stock, including news and social media data.
                            </li>
                            <li>
                                <strong>Fundamental Analysis:</strong> Assessing the company's financial health, profitability, and growth potential.
                            </li>
                            <li>
                                <strong>Technical Analysis:</strong> Analyzing price trends and trading volumes and other technical indicators.
                            </li>
                            <li>
                                <strong>Industry Analysis:</strong> Evaluating the company's competitive position and market share in the industry.
                            </li>
                            <li>
                                <strong>Risk Assessment:</strong> Identifying key risk factors that may affect the company's performance.
                            </li>
                        </ul>
                        <p className="mb-4">
                            <strong>Rating Explanation:</strong>
                        </p>
                        <ul className="space-y-2">
                            <li className="flex items-center">
                                <span className="inline-block w-24 font-medium">Strong Buy:</span>
                                <span>Expected stock price increase of more than 20% in the next 12 months (Score {'>'}  0.7)</span>
                            </li>
                            <li className="flex items-center">
                                <span className="inline-block w-24 font-medium">Buy:</span>
                                <span>Expected stock price increase of 10-20% in the next 12 months (Score 0.3 to 0.7)</span>
                            </li>
                            <li className="flex items-center">
                                <span className="inline-block w-24 font-medium">Hold:</span>
                                <span>Expected stock price fluctuation between -10% and +10% in the next 12 months (Score -0.3 to 0.3)</span>
                            </li>
                            <li className="flex items-center">
                                <span className="inline-block w-24 font-medium">Sell:</span>
                                <span>Expected stock price decrease of 10-20% in the next 12 months (Score -0.7 to -0.3)</span>
                            </li>
                            <li className="flex items-center">
                                <span className="inline-block w-24 font-medium">Strong Sell:</span>
                                <span>Expected stock price decrease of more than 20% in the next 12 months (Score {'<'} -0.7)</span>
                            </li>
                        </ul>
                    </div>
                }
            />
        </div>
    );
};

export default FinalRecommendation; 