'use client';

import React, { useEffect, useState } from 'react';
import { FaTimes, FaSpinner, FaInfoCircle, FaBolt } from 'react-icons/fa';
import { explainSingleText } from '@/services/huggingfaceService';

interface ExplainabilityData {
    tokens: string[];
    importanceValues: number[];
    attentionMatrix: number[][] | null;
    importanceError: string | null;
}

interface ProcessingTimes {
    total: number;
    sentiment: number;
    attention: number;
    importance: number | null;
}

interface MethodInfo {
    method: string;
    description: string;
}

interface TextExplainabilityModalProps {
    isOpen: boolean;
    onClose: () => void;
    text: string;
}

const TextExplainabilityModal: React.FC<TextExplainabilityModalProps> = ({
    isOpen,
    onClose,
    text,
}) => {
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [explainabilityData, setExplainabilityData] = useState<ExplainabilityData | null>(null);
    const [sentiment, setSentiment] = useState<string>('');
    const [sentimentScore, setSentimentScore] = useState<number>(0);
    const [processingTimes, setProcessingTimes] = useState<ProcessingTimes | null>(null);
    const [methodInfo, setMethodInfo] = useState<MethodInfo | null>(null);
    const [selectedToken, setSelectedToken] = useState<number | null>(null);
    const [isImageEnlarged, setIsImageEnlarged] = useState(false);

    // Get explainability data
    useEffect(() => {
        const fetchExplainability = async () => {
            if (!isOpen || !text) return;

            setLoading(true);
            setError(null);
            setSelectedToken(null);

            try {
                // Ensure the text format is correct
                let cleanText = text;
                if (cleanText.startsWith('undefined:')) {
                    cleanText = cleanText.substring('undefined:'.length).trim();
                }

                // Limit text length
                const MAX_TEXT_LENGTH = 1000;
                if (cleanText.length > MAX_TEXT_LENGTH) {
                    cleanText = cleanText.substring(0, MAX_TEXT_LENGTH);
                }

                const data = await explainSingleText(cleanText);

                if (data.explainability) {
                    // Use type assertion to ensure type matching
                    setExplainabilityData(data.explainability as any);
                    setSentiment(data.sentiment);
                    setSentimentScore(data.sentimentScore);
                    setProcessingTimes(data.processingTimes as any || null);
                    setMethodInfo(data.methodInfo as any || null);
                } else {
                    setError('Failed to retrieve explainability data');
                }
            } catch (err: any) {
                // Display more useful error messages
                if (err.message && err.message.includes('backend service has started')) {
                    setError('Backend service connection failed, please ensure Flask backend is running');
                } else if (err.response && err.response.data && err.response.data.message) {
                    setError(err.response.data.message);
                } else {
                    setError(err.message || 'Error during analysis, unable to retrieve SHAP and self-attention data');
                }
                console.error('Failed to get explainability data:', err);
            } finally {
                setLoading(false);
            }
        };

        fetchExplainability();
    }, [isOpen, text]);

    // Reset selected token (when modal opens or text changes)
    useEffect(() => {
        setSelectedToken(null);
    }, [isOpen, text]);

    // Render feature importance tokens and corresponding values
    const renderImportanceTokens = () => {
        if (!explainabilityData?.tokens || !explainabilityData?.importanceValues) {
            return <p>No feature importance data available</p>;
        }

        const tokens = explainabilityData.tokens;
        const values = explainabilityData.importanceValues;

        // If tokens and values lengths don't match, it might be a data issue
        if (tokens.length === 0 || values.length === 0 || tokens.length !== values.length) {
            return <p>Feature importance data format is incorrect</p>;
        }

        return (
            <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">Key Words Affecting Sentiment Analysis</h3>
                <div className="grid grid-cols-1 gap-2">
                    {tokens.map((token, index) => {
                        const value = values[index] || 0;
                        const isPositive = value > 0;
                        const absValue = Math.abs(value);
                        const width = `${Math.min(absValue * 100, 100)}%`;

                        return (
                            <div key={index} className="flex items-center mb-1">
                                <div className="w-24 overflow-hidden text-ellipsis whitespace-nowrap">
                                    {token}
                                </div>
                                <div className="flex-1 h-6 bg-gray-200 rounded-md overflow-hidden">
                                    <div
                                        className={`h-full ${isPositive ? 'bg-green-500' : 'bg-red-500'}`}
                                        style={{ width }}
                                    ></div>
                                </div>
                                <div className="w-16 text-right ml-2">
                                    {value.toFixed(2)}
                                </div>
                            </div>
                        );
                    })}
                </div>
                <p className="text-sm text-gray-600 mt-2">
                    Positive values contribute to positive sentiment, negative values contribute to negative sentiment
                </p>
                {methodInfo?.method === "IntegratedGradients" && (
                    <div className="flex items-center mt-2 text-blue-600 text-sm">
                        <FaBolt className="mr-1" />
                        Calculated using Integrated Gradients technique
                    </div>
                )}
            </div>
        );
    };

    // Render Self-Attention visualization
    const renderAttentionHeatmap = () => {
        if (!explainabilityData?.tokens || !explainabilityData?.attentionMatrix) {
            return <p>No Self-Attention data available</p>;
        }

        // Get tokens and attention matrix
        const tokens = explainabilityData.tokens;

        // Filter punctuation and special tokens
        const punctuationRegex = /^[.,;:!?'"()[\]{}…—\-<>#+=%&$@*/\\]+$/;
        const specialTokens = ['[PAD]', '[CLS]', '[SEP]', '[UNK]'];

        // Create filtered tokens and corresponding index mapping
        const filteredTokensInfo = tokens.map((token, index) => {
            const isSpecialToken = specialTokens.includes(token);
            const isPunctuation = punctuationRegex.test(token);
            const isSpace = token.trim() === '';
            return {
                token,
                index,
                isValid: !isSpecialToken && !isPunctuation && !isSpace
            };
        });

        // Keep only valid tokens
        const filteredTokens = filteredTokensInfo.filter(item => item.isValid);

        // Find the 5 tokens with highest attention to the currently selected token
        const findImportantAttentions = (tokenIndex: number) => {
            if (!explainabilityData?.attentionMatrix) return [];

            const attentions = explainabilityData.attentionMatrix[tokenIndex];

            // Create pairs of filtered tokens and attention values
            const tokenAttentions = filteredTokens.map(item => ({
                token: item.token,
                index: item.index,
                attention: attentions[item.index]
            }));

            // Sort by attention value and return top 5
            return tokenAttentions
                .sort((a, b) => b.attention - a.attention)
                .slice(0, 5);
        };

        return (
            <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3">Self-Attention Visualization</h3>
                <p className="text-sm text-gray-600 mb-3">Select a word to see its attention distribution:</p>

                {/* Display token buttons for user selection */}
                <div className="flex flex-wrap gap-2 mb-4">
                    {filteredTokens.map((item) => {
                        return (
                            <button
                                key={`token-${item.index}`}
                                onClick={() => setSelectedToken(item.index)}
                                className={`px-2 py-1 rounded text-sm ${selectedToken === item.index
                                    ? 'bg-blue-600 text-white'
                                    : 'bg-gray-200 hover:bg-gray-300'
                                    }`}
                            >
                                {item.token}
                            </button>
                        );
                    })}
                </div>

                {/* Show attention distribution for selected token */}
                {selectedToken !== null && (
                    <div className="mt-4 p-4 bg-gray-50 rounded-lg">
                        <p className="font-medium mb-2">Word "{tokens[selectedToken]}" mainly attends to:</p>

                        <div className="space-y-2">
                            {findImportantAttentions(selectedToken).map((item, idx) => (
                                <div key={`attention-${idx}`} className="flex items-center">
                                    <div className="w-24 font-medium">{item.token}</div>
                                    <div className="flex-1">
                                        <div
                                            className="bg-blue-500 h-5 rounded"
                                            style={{ width: `${item.attention * 100}%` }}
                                        ></div>
                                    </div>
                                    <div className="w-16 text-right text-sm">
                                        {(item.attention * 100).toFixed(1)}%
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        );
    };

    // Render processing method information
    const renderMethodInfo = () => {
        if (!methodInfo) return null;

        const getMethodDisplayName = () => {
            switch (methodInfo.method) {
                case 'IntegratedGradients':
                    return 'Integrated Gradients';
                case 'LIME':
                    return 'Local Interpretable Model-agnostic Explanations (LIME)';
                case 'SHAP':
                    return 'SHAP Value Analysis';
                default:
                    return methodInfo.method;
            }
        };

        return (
            <div className="mb-4 p-3 bg-blue-50 rounded-md">
                <h3 className="text-lg font-semibold mb-2">Analysis Method Information</h3>
                <div className="grid grid-cols-1 gap-2">
                    <div className="flex justify-between">
                        <span className="text-sm">Method Used:</span>
                        <span className="text-sm font-medium">{getMethodDisplayName()}</span>
                    </div>

                    {methodInfo.description && (
                        <div className="text-sm text-gray-600 mt-1">
                            {methodInfo.description}
                        </div>
                    )}
                </div>
            </div>
        );
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 overflow-y-auto">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-auto m-4">
                <div className="flex items-center justify-between p-4 border-b border-gray-200">
                    <h3 className="text-xl font-bold">Text Explainability Analysis</h3>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-gray-700 transition-colors"
                        aria-label="Close"
                    >
                        <FaTimes size={20} />
                    </button>
                </div>

                <div className="p-6">
                    <div className="mb-4">
                        <h3 className="text-lg font-semibold">Analysis Text</h3>
                        <p className="p-3 bg-gray-100 rounded-md">{text}</p>
                    </div>

                    <div className="mb-4">
                        <h3 className="text-lg font-semibold">Sentiment Analysis Result</h3>
                        <div className="flex items-center mt-2">
                            <div
                                className={`px-3 py-1 rounded-md font-medium ${sentiment === 'Positive' ? 'bg-green-100 text-green-800' :
                                    sentiment === 'Negative' ? 'bg-red-100 text-red-800' :
                                        'bg-gray-100 text-gray-800'
                                    }`}
                            >
                                {sentiment === 'Positive' ? 'Positive' :
                                    sentiment === 'Negative' ? 'Negative' : 'Neutral'}
                            </div>
                            <div className="ml-4">
                                Sentiment Score: {sentimentScore.toFixed(2)}
                                <span className="text-sm text-gray-500 ml-1">(Between -1 and 1, positive values indicate positive sentiment)</span>
                            </div>
                        </div>
                    </div>

                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-12">
                            <FaSpinner className="animate-spin text-blue-500 text-4xl mb-4" />
                            <p>Analyzing text, please wait...</p>
                            <p className="text-sm text-gray-500 mt-2">Analysis processing may take 30-60 seconds, please be patient</p>
                            <div className="mt-4 w-full max-w-md bg-gray-200 rounded-full h-2.5">
                                <div className="bg-blue-500 h-2.5 rounded-full animate-pulse w-3/4"></div>
                            </div>
                        </div>
                    ) : error ? (
                        <div className="bg-red-100 border border-red-200 text-red-700 px-4 py-3 rounded-md">
                            <div className="flex">
                                <div className="flex-shrink-0">
                                    <FaInfoCircle className="h-5 w-5 text-red-500" />
                                </div>
                                <div className="ml-3">
                                    <p className="font-medium">Error during analysis</p>
                                    <p className="text-sm">{error}</p>
                                    <details className="mt-2 text-xs">
                                        <summary>Debug Info</summary>
                                        <pre className="mt-2 whitespace-pre-wrap">
                                            Backend URL: http://localhost:5000/api/explain-single-text
                                            Text: {text?.substring(0, 100)}...
                                            Text Length: {text?.length || 0}

                                            If you see "Backend service connection failed":
                                            1. Please confirm Flask service is running on port 5000
                                            2. Check if there are CORS errors in the console
                                            3. Try directly accessing http://localhost:5000/health
                                        </pre>
                                    </details>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <>
                            {/* Feature importance tokens */}
                            {renderImportanceTokens()}

                            {/* Self-Attention heatmap */}
                            {renderAttentionHeatmap && renderAttentionHeatmap()}

                            {/* Method information */}
                            {renderMethodInfo()}

                            {/* Error message display */}
                            {explainabilityData?.importanceError && (
                                <div className="bg-yellow-100 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-md mt-4">
                                    <div className="flex">
                                        <div className="flex-shrink-0">
                                            <FaInfoCircle className="h-5 w-5 text-yellow-500" />
                                        </div>
                                        <div className="ml-3">
                                            <p className="font-medium">Some analysis results unavailable</p>
                                            <p className="text-sm">Feature importance calculation error: {explainabilityData.importanceError}</p>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Processing time information */}
                            {processingTimes && (
                                <div className="mt-4 text-xs text-gray-500">
                                    <p>Processing time: {processingTimes.total?.toFixed(2)} seconds</p>
                                    <p>Sentiment analysis: {processingTimes.sentiment?.toFixed(2)} seconds</p>
                                    <p>Feature importance calculation: {processingTimes.importance?.toFixed(2)} seconds</p>
                                </div>
                            )}
                        </>
                    )}
                </div>

                <div className="p-4 border-t border-gray-200 text-right">
                    <button
                        onClick={onClose}
                        className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-800 rounded-md transition-colors"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

export default TextExplainabilityModal; 