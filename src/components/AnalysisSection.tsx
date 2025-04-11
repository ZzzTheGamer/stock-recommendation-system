'use client';

import React, { useState, useEffect } from 'react';
import { FaChartLine, FaFileAlt, FaChartBar, FaQuestionCircle, FaArrowUp, FaArrowDown, FaLightbulb, FaNewspaper, FaRegFileAlt, FaSearch } from 'react-icons/fa';
import ExplanationModal from '@/components/ExplanationModal';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import { getCompanyNews } from '@/services/finnhubService';
import { generateFinancialAnalysisReport } from '@/services/researchService';
import { analyzeSentimentWithFinGPT, generateFinancialAnalysisWithFinGPT } from '@/services/huggingfaceService';
import { getFullMistralExplanation, getFullMistralExplanationAsync } from '@/services/explainabilityService';
import StrategyAnalysis from '@/components/StrategyAnalysis';
import TextExplainabilityModal from '@/components/TextExplainabilityModal';
import MistralExplanationModal from '@/components/MistralExplanationModal';

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

// Mock SHAP image URL
const shapImageUrl = 'https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_header.png';

interface AnalysisSectionProps {
    stockSymbol: string;
}

interface TradeSignal {
    date: string;
    action: string;
    price: number;
    reason?: string;
}

const AnalysisSection: React.FC<AnalysisSectionProps> = ({ stockSymbol }) => {
    const [activeModal, setActiveModal] = useState<string | null>(null);
    const [loading, setLoading] = useState<{ [key: string]: boolean }>({
        sentiment: false,
        research: false,
        strategy: false
    });

    // Analysis result states
    const [sentimentAnalysis, setSentimentAnalysis] = useState<any>(null);
    const [researchAnalysis, setResearchAnalysis] = useState<any>(null);
    const [strategyAnalysis, setStrategyAnalysis] = useState<any>(null);

    // Single text explanation state
    const [selectedTextForExplain, setSelectedTextForExplain] = useState<string>('');
    const [showExplainModal, setShowExplainModal] = useState<boolean>(false);

    // Debug information
    const [debugInfo, setDebugInfo] = useState<string>('');

    // Add Mistral explainability analysis related states
    const [mistralExplanation, setMistralExplanation] = useState<any>(null);
    const [showMistralExplanationModal, setShowMistralExplanationModal] = useState<boolean>(false);
    const [loadingMistralExplanation, setLoadingMistralExplanation] = useState<boolean>(false);
    const [analysisProgress, setAnalysisProgress] = useState<{ progress: number, message: string }>({
        progress: 0,
        message: 'Preparing analysis...'
    });

    const openModal = (modalId: string) => {
        setActiveModal(modalId);
    };

    const closeModal = () => {
        setActiveModal(null);
    };

    // Debug log function
    const addDebugLog = (message: string) => {
        if (process.env.NEXT_PUBLIC_DEBUG_MODE === 'true') {
            console.log(`[AnalysisSection] ${message}`);
        }
    };

    // Open single text explanation modal
    const handleExplainText = (text: string) => {
        setSelectedTextForExplain(text);
        setShowExplainModal(true);
    };

    // Close single text explanation modal
    const handleCloseExplainModal = () => {
        setShowExplainModal(false);
    };

    // Run sentiment analysis
    const runSentimentAnalysis = async () => {
        try {
            setLoading(prev => ({ ...prev, sentiment: true }));
            setDebugInfo('');

            // Add debug log
            addDebugLog(`[Sentiment Analysis] Starting analysis for ${stockSymbol}`);

            // Get news from the past 30 days
            const thirtyDaysAgo = new Date();
            thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
            const today = new Date();

            addDebugLog(`[Sentiment Analysis] Getting news, date range: ${thirtyDaysAgo.toISOString().split('T')[0]} to ${today.toISOString().split('T')[0]}`);

            // Call API to get news
            const newsResponse = await fetch(`/api/news?symbol=${stockSymbol}`);
            const newsData = await newsResponse.json();

            addDebugLog(`[Sentiment Analysis] Retrieved ${newsData.length} news items`);

            // Extract the latest 5 news texts
            const newsTexts = newsData
                .slice(0, 5)
                .map((news: any) => {
                    // Try to extract title - handle more invalid cases
                    const title = news.title &&
                        news.title !== 'undefined' &&
                        news.title !== 'null' &&
                        typeof news.title === 'string' ?
                        news.title.trim() : '';

                    // Try to extract summary
                    const summary = (news.summary || news.description || '').trim();

                    // Consider it valid news if either title or summary has content
                    if ((title || summary) && (title + summary).trim()) {
                        // Use "title: summary" format if title exists, otherwise just use summary
                        return title ? `${title}: ${summary}` : summary;
                    }
                    return '';
                })
                .filter((text: string) => text && text.trim() !== '' && text !== 'undefined:'); // Filter empty texts

            addDebugLog(`[Sentiment Analysis] Extracted valid news texts: ${newsTexts.length}`);

            // Ensure there are valid news texts to analyze
            if (newsTexts.length === 0) {
                addDebugLog(`[Sentiment Analysis] No valid news texts found, cannot perform analysis`);
                setSentimentAnalysis({
                    overallSentiment: 'Neutral',
                    sentimentScore: '0.00',
                    confidence: 'N/A',
                    details: [],
                    topPositiveFactors: [],
                    topNegativeFactors: [],
                    analysisDate: new Date().toISOString(),
                    recommendation: 'Not enough news data for analysis',
                    apiCallSuccess: false,
                    isSimulated: true,
                    apiSuccessRate: '0.00'
                });
                setLoading(prev => ({ ...prev, sentiment: false }));
                return;
            }

            // Call sentiment analysis API
            addDebugLog(`[Sentiment Analysis] Calling sentiment analysis model...`);

            const result = await analyzeSentimentWithFinGPT(newsTexts);

            // Check API call status
            if (result.isSimulated) {
                addDebugLog(`[Sentiment Analysis] Using simulated data (API call failed)`);
            } else if (result.apiCallSuccess) {
                addDebugLog(`[Sentiment Analysis] Using real API data (success rate: ${result.apiSuccessRate}%)`);
            } else {
                addDebugLog(`[Sentiment Analysis] Using partial real API data`);
            }

            // Record sentiment analysis results
            addDebugLog(`[Sentiment Analysis] Result: ${result.overallSentiment}, Score: ${result.sentimentScore}, Confidence: ${result.confidence}`);

            // Record positive and negative factors
            if (result.topPositiveFactors && result.topPositiveFactors.length > 0) {
                addDebugLog(`[Sentiment Analysis] Positive factors: ${result.topPositiveFactors.join(', ')}`);
            }

            if (result.topNegativeFactors && result.topNegativeFactors.length > 0) {
                addDebugLog(`[Sentiment Analysis] Negative factors: ${result.topNegativeFactors.join(', ')}`);
            }

            setSentimentAnalysis(result);

            // Save to localStorage for use by other components
            localStorage.setItem('sentimentAnalysis', JSON.stringify(result));

            // Trigger custom event to notify other components that data has been updated
            const event = new CustomEvent('sentimentAnalysisComplete', { detail: result });
            window.dispatchEvent(event);

        } catch (error) {
            console.error('Error in sentiment analysis:', error);
            addDebugLog(`[Sentiment Analysis] Error: ${error}`);
        } finally {
            setLoading(prev => ({ ...prev, sentiment: false }));
        }
    };

    // Generate financial analysis report
    const generateResearchReport = async () => {
        setLoading({ ...loading, research: true });
        addDebugLog(`Starting financial analysis report generation, stock symbol: ${stockSymbol}`);

        try {
            // Use real API to get financial analysis report
            addDebugLog('Calling generateFinancialAnalysisReport to get real financial data and analysis...');
            const result = await generateFinancialAnalysisReport(stockSymbol);

            // Check if API call was successful
            if (result.isSimulated) {
                addDebugLog(`Financial analysis complete, recommendation: ${result.recommendation}, using simulated data (API call failed)`);
            } else if (typeof result.apiCallSuccess === 'object') {
                // Get model type and success status
                const modelType = result.apiCallSuccess.modelType || 'unknown';
                const modelSuccess = result.apiCallSuccess.modelSuccess;
                const financialSuccess = result.apiCallSuccess.financialData;

                // Build more accurate log message
                addDebugLog(
                    `Financial analysis complete, recommendation: ${result.recommendation}, ` +
                    `financial data API call ${financialSuccess ? 'successful' : 'failed'}, ` +
                    `${modelType} model call ${modelSuccess ? 'successful' : 'failed'}`
                );
            } else {
                addDebugLog(`Financial analysis complete, recommendation: ${result.recommendation}, API call partially successful`);
            }

            setResearchAnalysis(result);

            // Save research analysis results to localStorage
            localStorage.setItem('researchAnalysis', JSON.stringify(result));

            // Trigger custom event to notify other components that data has been updated
            const event = new CustomEvent('researchAnalysisComplete', { detail: result });
            window.dispatchEvent(event);

        } catch (error) {
            console.error('Error generating research report:', error);
            addDebugLog(`Error generating financial analysis report: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setLoading({ ...loading, research: false });
        }
    };

    // Generate Mistral model explanation analysis
    const generateMistralExplanation = async () => {
        if (!researchAnalysis) {
            addDebugLog('[Mistral Explanation Analysis] Error: No research analysis results available');
            return;
        }

        setLoadingMistralExplanation(true);
        addDebugLog(`[Mistral Explanation Analysis] Starting explanation analysis, using TreeSHAP method...`);

        try {
            // Progress update callback function
            const handleProgress = (progress: { progress: number, message: string }) => {
                addDebugLog(`[Mistral Explanation Analysis] Progress: ${progress.progress}%, ${progress.message}`);
                setAnalysisProgress(progress);
            };

            // Use asynchronous method for analysis
            // Note: Investment recommendation text prioritizes the investmentRecommendation field, falling back to metricsAnalysis if not available
            const investmentRecommendationText = researchAnalysis.investmentRecommendation || researchAnalysis.metricsAnalysis;

            // Add log recording the investment recommendation text passed
            if (researchAnalysis.investmentRecommendation) {
                addDebugLog('[Mistral Explanation Analysis] Using dedicated investmentRecommendation field as investment advice text');
            } else {
                addDebugLog('[Mistral Explanation Analysis] Dedicated investment recommendation field not found, using metricsAnalysis as fallback');
            }

            const result = await getFullMistralExplanationAsync(
                researchAnalysis.keyMetrics,
                researchAnalysis.recommendation,
                researchAnalysis.summary,
                investmentRecommendationText,
                'TreeSHAP',  // Always use TreeSHAP method
                handleProgress  // Pass progress callback
            );

            if (result.success) {
                setMistralExplanation(result.explanation);
                setShowMistralExplanationModal(true);
                addDebugLog(`[Mistral Explanation Analysis] TreeSHAP explanation analysis generated successfully`);
            } else {
                addDebugLog(`[Mistral Explanation Analysis] Generation failed: ${result.error}`);
            }
        } catch (error) {
            console.error('Error generating Mistral explanation:', error);
            addDebugLog(`[Mistral Explanation Analysis] Error: ${error instanceof Error ? error.message : String(error)}`);
        } finally {
            setLoadingMistralExplanation(false);
        }
    };

    // Close Mistral explanation modal
    const handleCloseMistralExplanationModal = () => {
        setShowMistralExplanationModal(false);
    };

    return (
        <div className="card">
            <h2 className="text-xl font-bold mb-6">Analysis Modules</h2>

            {/* Debug information */}
            {debugInfo && (
                <div className="mb-4 p-3 bg-gray-100 rounded text-xs font-mono whitespace-pre-wrap max-h-40 overflow-auto">
                    <h3 className="font-bold mb-1">Debug Info:</h3>
                    {debugInfo}
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Sentiment Analysis Module */}
                <div className="bg-gray-50 rounded-lg p-5">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-lg font-semibold flex items-center">
                            <FaChartLine className="mr-2 text-primary" />
                            Sentiment Analysis
                        </h3>
                        <button
                            onClick={() => openModal('sentiment')}
                            className="text-gray-500 hover:text-primary"
                            aria-label="View sentiment analysis explanation"
                        >
                            <FaQuestionCircle size={18} />
                        </button>
                    </div>

                    <p className="text-sm text-gray-600 mb-4">
                        Analysis of news sentiment and social media mentions for {stockSymbol} to gauge market perception and potential price impact.
                    </p>

                    <button
                        className={`btn btn-primary w-full ${loading.sentiment ? 'opacity-50 cursor-not-allowed' : ''}`}
                        onClick={runSentimentAnalysis}
                        disabled={loading.sentiment}
                    >
                        {loading.sentiment ? 'Analyzing...' : 'Run Sentiment Analysis'}
                    </button>

                    {sentimentAnalysis && (
                        <div className="mt-4">
                            <div className="mb-3">
                                <h4 className="font-semibold">Overall Sentiment:</h4>
                                <span className={`font-medium ${sentimentAnalysis.overallSentiment === 'Positive' ? 'text-success' :
                                    sentimentAnalysis.overallSentiment === 'Negative' ? 'text-danger' :
                                        'text-primary'}`}>
                                    {sentimentAnalysis.overallSentiment}
                                </span>
                            </div>

                            {sentimentAnalysis.isSimulated && (
                                <div className="alert alert-info mb-3">
                                    <small>Note: This is simulated data. Real API call failed.</small>
                                </div>
                            )}

                            <div className="mb-3">
                                <span className="text-sm text-gray-600">Sentiment Score: </span>
                                <span className="font-medium">{sentimentAnalysis.sentimentScore}</span>
                            </div>

                            <div className="mb-3">
                                <span className="text-sm text-gray-600">Confidence: </span>
                                <span className="font-medium">{sentimentAnalysis.confidence}</span>
                            </div>

                            {/* Display list of analyzed texts and their details */}
                            {sentimentAnalysis.details && sentimentAnalysis.details.length > 0 && (
                                <div className="mt-4">
                                    <h4 className="font-semibold mb-2">Analyzed Texts:</h4>
                                    <div className="space-y-3">
                                        {sentimentAnalysis.details
                                            .filter((detail: any) =>
                                                detail.text &&
                                                detail.text !== 'undefined:' &&
                                                detail.text.trim() !== ''
                                            )
                                            .map((detail: any, index: number) => (
                                                <div key={index} className="p-3 bg-gray-50 rounded-md">
                                                    <div className="flex justify-between items-start">
                                                        <div className="flex-1">
                                                            <p className="text-sm">{detail.text.length > 100 ? `${detail.text.substring(0, 100)}...` : detail.text}</p>
                                                            <div className="flex items-center mt-1">
                                                                <span className={`text-xs px-2 py-0.5 rounded ${detail.sentiment === 'Positive' ? 'bg-green-100 text-green-800' :
                                                                    detail.sentiment === 'Negative' ? 'bg-red-100 text-red-800' :
                                                                        'bg-gray-100 text-gray-800'
                                                                    }`}>
                                                                    {detail.sentiment}
                                                                </span>
                                                                <span className="text-xs text-gray-500 ml-2">
                                                                    Score: {detail.score.toFixed(2)}
                                                                </span>
                                                            </div>
                                                        </div>
                                                        <button
                                                            className="ml-2 px-2 py-1 text-xs bg-blue-50 text-blue-600 rounded hover:bg-blue-100"
                                                            onClick={() => handleExplainText(detail.text)}
                                                        >
                                                            Explain
                                                        </button>
                                                    </div>
                                                </div>
                                            ))}
                                    </div>
                                </div>
                            )}

                            {sentimentAnalysis.topPositiveFactors && sentimentAnalysis.topPositiveFactors.length > 0 && (
                                <div className="mb-2">
                                    <h5 className="text-sm font-medium text-success mb-1">Positive Factors:</h5>
                                    <ul className="text-sm list-disc pl-5">
                                        {sentimentAnalysis.topPositiveFactors.map((factor: string, index: number) => (
                                            <li key={index} className="mb-1 text-xs">
                                                {factor}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}

                            {sentimentAnalysis.topNegativeFactors && sentimentAnalysis.topNegativeFactors.length > 0 && (
                                <div className="mb-2">
                                    <h5 className="text-sm font-medium text-danger mb-1">Negative Factors:</h5>
                                    <ul className="text-sm list-disc pl-5">
                                        {sentimentAnalysis.topNegativeFactors.map((factor: string, index: number) => (
                                            <li key={index} className="mb-1 text-xs">
                                                {factor}
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Research Analysis Module */}
                <div className="bg-gray-50 rounded-lg p-5">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-lg font-semibold flex items-center">
                            <FaFileAlt className="mr-2 text-primary" />
                            Financial Analysis
                        </h3>
                        <button
                            onClick={() => openModal('research')}
                            className="text-gray-500 hover:text-primary"
                            aria-label="View financial analysis explanation"
                        >
                            <FaQuestionCircle size={18} />
                        </button>
                    </div>

                    <p className="text-sm text-gray-600 mb-4">
                        Comprehensive financial analysis of {stockSymbol} including growth metrics, financial health, valuation metrics, and investment recommendations.
                    </p>

                    <button
                        className={`btn btn-primary w-full ${loading.research ? 'opacity-50 cursor-not-allowed' : ''}`}
                        onClick={generateResearchReport}
                        disabled={loading.research}
                    >
                        {loading.research ? 'Analyzing...' : 'Generate Research Report'}
                    </button>

                    {researchAnalysis && (
                        <div className="mt-4">
                            <div className="mb-3">
                                <h4 className="font-semibold">Recommendation:</h4>
                                <span className={`font-medium ${researchAnalysis.recommendation === 'Buy' ? 'text-success' :
                                    researchAnalysis.recommendation === 'Sell' ? 'text-danger' :
                                        'text-primary'}`}>
                                    {researchAnalysis.recommendation}
                                </span>
                            </div>

                            {researchAnalysis.isSimulated && (
                                <div className="alert alert-info mb-3">
                                    <small>Note: This is simulated data. Real API call failed.</small>
                                </div>
                            )}

                            <div className="mb-3">
                                <h4 className="font-semibold">Summary:</h4>
                                <p className="text-sm text-gray-700">{researchAnalysis.summary}</p>
                                <button
                                    className="text-xs text-primary hover:underline mt-1"
                                    onClick={() => handleExplainText(researchAnalysis.summary)}
                                >
                                    Explain this
                                </button>
                            </div>

                            <div className="mb-3">
                                <h4 className="font-semibold">Strengths:</h4>
                                <p className="text-sm text-gray-700 whitespace-pre-line">{researchAnalysis.strengths}</p>
                            </div>

                            <div className="mb-3">
                                <h4 className="font-semibold">Weaknesses:</h4>
                                <p className="text-sm text-gray-700 whitespace-pre-line">{researchAnalysis.weaknesses}</p>
                            </div>

                            <button
                                className={`btn btn-outline btn-info w-full mt-4 flex items-center justify-center ${loadingMistralExplanation ? 'opacity-50 cursor-not-allowed' : ''}`}
                                onClick={generateMistralExplanation}
                                disabled={loadingMistralExplanation || researchAnalysis.isSimulated}
                            >
                                <FaLightbulb className="mr-2" />
                                {loadingMistralExplanation ? 'Analyzing...' : 'Explanation Analysis'}
                            </button>

                            {researchAnalysis.isSimulated && (
                                <div className="text-xs text-gray-500 mt-1 text-center">
                                    Note: Simulated data does not support explanation analysis functionality
                                </div>
                            )}
                        </div>
                    )}
                </div>

                {/* Strategy Analysis Module */}
                <div className="bg-gray-50 rounded-lg p-5">
                    <div className="flex items-center justify-between mb-3">
                        <h3 className="text-lg font-semibold flex items-center">
                            <FaChartBar className="mr-2 text-primary" />
                            Strategy Analysis
                        </h3>
                        <button
                            onClick={() => openModal('strategy')}
                            className="text-gray-500 hover:text-primary"
                            aria-label="View strategy analysis explanation"
                        >
                            <FaQuestionCircle size={18} />
                        </button>
                    </div>

                    <p className="text-sm text-gray-600 mb-4">
                        Quantitative trading strategies and backtesting results for {stockSymbol} based on historical data and market trends.
                    </p>

                    {/* Using StrategyAnalysis component */}
                    <StrategyAnalysis ticker={stockSymbol} />
                </div>
            </div>

            {/* Modals */}
            <ExplanationModal
                isOpen={activeModal === 'sentiment'}
                onClose={() => setActiveModal(null)}
                title="Sentiment Analysis"
                content={
                    <>
                        <p>Sentiment analysis uses natural language processing to gauge the emotional tone in news and social media content about a stock.</p>
                        <p className="mt-2">FinBert analyzes recent news articles and social media posts to determine if the overall sentiment is positive, negative, or neutral.</p>
                        <p className="mt-2">This can provide early signals of potential price movements, as market sentiment often precedes price changes.</p>
                    </>
                }
            />

            <ExplanationModal
                isOpen={activeModal === 'research'}
                onClose={() => setActiveModal(null)}
                title="Research Analysis"
                content={
                    <>
                        <p>Financial analysis provides an overview of a company's financial health based on key metrics and financial statements.</p>
                        <p className="mt-2">The analysis examines:</p>
                        <ul className="list-disc pl-5 mt-1">
                            <li>Revenue and earnings growth trends</li>
                            <li>Profitability metrics (margins, ROE, ROA)</li>
                            <li>Liquidity and solvency ratios</li>
                        </ul>
                        <p className="mt-2">This comprehensive assessment helps determine if a stock is potentially undervalued or overvalued.</p>
                    </>
                }
            />

            <ExplanationModal
                isOpen={activeModal === 'strategy'}
                onClose={() => setActiveModal(null)}
                title="Trading Strategy Analysis"
                content={
                    <>
                        <p>Our trading strategy analysis uses various strategies to evaluate technical indicators and suggest potential trading approaches.</p>
                        <p className="mt-2">The system analyzes patterns in historical price and volume data to identify potential entry and exit points.</p>
                    </>
                }
            />

            {/* TextExplainabilityModal */}
            {showExplainModal && (
                <TextExplainabilityModal
                    isOpen={showExplainModal}
                    onClose={handleCloseExplainModal}
                    text={selectedTextForExplain}
                />
            )}

            {/* Mistral explanation analysis modal */}
            <MistralExplanationModal
                isOpen={showMistralExplanationModal}
                onClose={handleCloseMistralExplanationModal}
                explanation={mistralExplanation}
                loading={loadingMistralExplanation}
                progress={analysisProgress}
            />
        </div>
    );
};

export default AnalysisSection; 