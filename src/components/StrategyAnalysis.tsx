import React, { useState, useEffect, ReactNode } from 'react';
import {
    Box,
    Button,
    Card,
    CardContent,
    Typography,
    CircularProgress,
    Divider,
    Paper,
    Grid,
    Tooltip,
    IconButton,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';
import CloseIcon from '@mui/icons-material/Close';
import axios from 'axios';
import { getAvailableStrategies, analyzeStrategy, StrategyAnalysisResult } from '../services/strategyService';
import BacktestChart from './BacktestChart';

interface StrategyAnalysisProps {
    ticker: string;
}

const StrategyAnalysis: React.FC<StrategyAnalysisProps> = ({ ticker }): JSX.Element => {
    const [loading, setLoading] = useState({
        strategies: false,
        strategy: false
    });
    const [selectedStrategy, setSelectedStrategy] = useState<string>('');
    const [availableStrategies, setAvailableStrategies] = useState<string[]>([]);
    const [strategyAnalysis, setStrategyAnalysis] = useState<StrategyAnalysisResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [showStrategyDetails, setShowStrategyDetails] = useState(false);
    const [selectedSignal, setSelectedSignal] = useState<any>(null);
    const [showSignalDetails, setShowSignalDetails] = useState(false);

    useEffect(() => {
        const fetchStrategies = async () => {
            setLoading(prev => ({ ...prev, strategies: true }));
            try {
                // We're setting traditional strategies directly instead of fetching them
                setAvailableStrategies([
                    'moving_average',
                    'rsi',
                    'bollinger_bands',
                    'trend_following',
                    'backtrader_macro'
                ]);
            } catch (error) {
                console.error('Failed to fetch strategies:', error);
                setError('Failed to fetch strategies, please try again later');
            } finally {
                setLoading(prev => ({ ...prev, strategies: false }));
            }
        };

        fetchStrategies();
    }, []);

    // Auto-select first strategy if none selected
    useEffect(() => {
        if (availableStrategies.length > 0 && !selectedStrategy) {
            setSelectedStrategy(availableStrategies[0]);
        }
    }, [availableStrategies, selectedStrategy]);

    const runStrategyAnalysis = async () => {
        if (!ticker || !selectedStrategy) return;

        setLoading(prev => ({ ...prev, strategy: true }));
        setError(null);
        setStrategyAnalysis(null);

        try {
            // Try the API endpoint first
            const response = await axios.get(`/api/strategy?action=analyze&symbol=${ticker}&strategy=${selectedStrategy}`);
            if (response.data) {
                setStrategyAnalysis(response.data);
                setShowStrategyDetails(true);

                // Save strategy analysis results to localStorage
                localStorage.setItem('strategyAnalysis', JSON.stringify(response.data));

                // Trigger custom event to notify other components that data has been updated
                const event = new CustomEvent('strategyAnalysisComplete', { detail: response.data });
                window.dispatchEvent(event);
            }
        } catch (apiError) {
            console.error('API strategy analysis failed:', apiError);

            // Fallback to direct service call
            try {
                const analysisResult = await analyzeStrategy(ticker, selectedStrategy);
                setStrategyAnalysis(analysisResult);
                setShowStrategyDetails(true);

                // Save strategy analysis results to localStorage
                localStorage.setItem('strategyAnalysis', JSON.stringify(analysisResult));

                // Trigger custom event to notify other components that data has been updated
                const event = new CustomEvent('strategyAnalysisComplete', { detail: analysisResult });
                window.dispatchEvent(event);
            } catch (serviceError) {
                console.error('Service strategy analysis failed:', serviceError);
                setError('Strategy analysis failed, please try again later');
            }
        } finally {
            setLoading(prev => ({ ...prev, strategy: false }));
        }
    };

    const toggleStrategyDetails = () => {
        setShowStrategyDetails(!showStrategyDetails);
    };

    // Helper function to render performance metrics
    const renderPerformanceMetrics = (metrics: any): JSX.Element => {
        return (
            <Grid container spacing={2} className="mt-4">
                <Grid item xs={12} sm={6} md={4}>
                    <Paper className="p-4 bg-blue-50">
                        <Typography variant="subtitle2" className="text-gray-600">Annualized Return</Typography>
                        <Typography variant="h5" className="text-blue-600 font-bold">
                            {metrics.annualized_return?.toFixed(2)}%
                        </Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                    <Paper className="p-4 bg-blue-50">
                        <Typography variant="subtitle2" className="text-gray-600">Volatility</Typography>
                        <Typography variant="h5" className="text-blue-600 font-bold">
                            {metrics.volatility?.toFixed(2)}%
                        </Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                    <Paper className="p-4 bg-blue-50">
                        <Typography variant="subtitle2" className="text-gray-600">Sharpe Ratio</Typography>
                        <Typography variant="h5" className="text-blue-600 font-bold">
                            {metrics.sharpe_ratio?.toFixed(2)}
                        </Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                    <Paper className="p-4 bg-blue-50">
                        <Typography variant="subtitle2" className="text-gray-600">Max Drawdown</Typography>
                        <Typography variant="h5" className="text-blue-600 font-bold">
                            {metrics.max_drawdown?.toFixed(2)}%
                        </Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                    <Paper className="p-4 bg-blue-50">
                        <Typography variant="subtitle2" className="text-gray-600">Win Rate</Typography>
                        <Typography variant="h5" className="text-blue-600 font-bold">
                            {metrics.win_rate?.toFixed(2)}%
                        </Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} sm={6} md={4}>
                    <Paper className="p-4 bg-blue-50">
                        <Typography variant="subtitle2" className="text-gray-600">Total Return</Typography>
                        <Typography variant="h5" className="text-blue-600 font-bold">
                            {metrics.total_return?.toFixed(2)}%
                        </Typography>
                    </Paper>
                </Grid>
            </Grid>
        );
    };

    // Helper function to render backtest chart
    const renderBacktestChart = (backtest: any): JSX.Element => {
        if (!backtest || !backtest.dates || !backtest.strategy_returns) {
            return <Typography className="text-red-500">Backtest data not available</Typography>;
        }

        return (
            <div className="mt-4">
                <BacktestChart
                    dates={backtest.dates}
                    strategyReturns={backtest.strategy_returns}
                    activePeriods={backtest.active_periods}
                    totalDays={backtest.total_days}
                    activeDays={backtest.active_days}
                    height={450} // Increased height
                    title={`${ticker} Strategy Returns`}
                />
                <Typography variant="body2" className="mt-2 text-gray-600">
                    This chart shows the strategy returns based on {backtest.dates.length} trading days of historical data.
                </Typography>
            </div>
        );
    };

    // Add a new helper function to render macroeconomic indicators
    const renderMacroIndicators = (modelInfo: any): ReactNode => {
        // Strict validation: must have macro indicators array with length > 0, and the strategy type must support macro indicators
        if (!modelInfo ||
            !modelInfo.macro_indicators ||
            modelInfo.macro_indicators.length === 0 ||
            !strategyAnalysis ||
            strategyAnalysis.strategy.type !== 'backtrader_macro') {
            return null;
        }

        // Get current macro environment and data
        const currentEnvironment = modelInfo.current_macro_environment || 'NEUTRAL';
        const macroData = modelInfo.macro_data || null;

        // Environment description mapping
        const environmentDescriptions: Record<string, string> = {
            'EXPANSION': 'Strong economic growth with moderate inflation and improving employment',
            'OVERHEATING': 'Strong growth but high inflation and rising interest rates',
            'SLOWDOWN': 'Declining growth with persistent inflation and rising unemployment',
            'RECESSION': 'Negative growth with high unemployment and declining inflation',
            'RECOVERY': 'Economy beginning to grow after a recession with improving indicators',
            'NEUTRAL': 'Balanced economic conditions with no clear directional trend'
        };

        // Environment color mapping
        const environmentColors: Record<string, string> = {
            'EXPANSION': 'bg-green-100 text-green-800',
            'OVERHEATING': 'bg-orange-100 text-orange-800',
            'SLOWDOWN': 'bg-yellow-100 text-yellow-800',
            'RECESSION': 'bg-red-100 text-red-800',
            'RECOVERY': 'bg-blue-100 text-blue-800',
            'NEUTRAL': 'bg-gray-100 text-gray-800'
        };

        return (
            <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
                <h4 className="text-lg font-semibold mb-2">Macroeconomic Analysis</h4>

                {/* Current economic environment status */}
                <div className="mb-4">
                    <span className="text-sm text-gray-700 mr-2">Current Economic Environment:</span>
                    <span className={`px-3 py-1 rounded-full font-medium ${environmentColors[currentEnvironment] || 'bg-gray-100'}`}>
                        {currentEnvironment}
                    </span>
                    <p className="mt-1 text-sm text-gray-600">
                        {environmentDescriptions[currentEnvironment] || 'Economic conditions assessment based on real-time data'}
                    </p>
                </div>

                {/* Macroeconomic indicators */}
                <div className="mb-4">
                    <h5 className="font-medium mb-2">Real-Time Economic Indicators:</h5>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {macroData ? (
                            <>
                                <div className="flex items-center">
                                    <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                                    <span className="text-gray-800">CPI (Inflation): <strong>{macroData.inflation}</strong></span>
                                </div>
                                <div className="flex items-center">
                                    <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                                    <span className="text-gray-800">GDP Growth: <strong>{macroData.gdpGrowth}</strong></span>
                                </div>
                                <div className="flex items-center">
                                    <div className="w-2 h-2 bg-purple-500 rounded-full mr-2"></div>
                                    <span className="text-gray-800">Interest Rate: <strong>{macroData.interestRate}</strong></span>
                                </div>
                                <div className="flex items-center">
                                    <div className="w-2 h-2 bg-red-500 rounded-full mr-2"></div>
                                    <span className="text-gray-800">Unemployment: <strong>{macroData.unemployment}</strong></span>
                                </div>
                                <div className="col-span-2 text-xs text-gray-500 mt-1">
                                    Last updated: {new Date(macroData.lastUpdated).toLocaleString()}
                                </div>
                            </>
                        ) : (
                            <>
                                {modelInfo.macro_indicators.map((indicator: string, index: number) => (
                                    <div key={index} className="flex items-center">
                                        <div className="w-2 h-2 bg-blue-500 rounded-full mr-2"></div>
                                        <span className="text-gray-800">{indicator}</span>
                                    </div>
                                ))}
                            </>
                        )}
                    </div>
                </div>

                {/* Strategy adjustment explanation */}
                <div className="mt-3 p-3 bg-blue-50 rounded-md">
                    <p className="text-sm text-gray-700">
                        <strong>Strategy Adjustments:</strong> Based on the current {currentEnvironment.toLowerCase()} environment, this strategy has automatically adjusted its parameters for optimal performance:
                    </p>
                    <ul className="mt-2 text-sm ml-4 list-disc text-gray-600">
                        {currentEnvironment === 'EXPANSION' && (
                            <>
                                <li>Increased RSI overbought threshold to 75 (standard: 70)</li>
                                <li>Raised RSI oversold threshold to 35 (standard: 30)</li>
                                <li>Enhanced trend following sensitivity</li>
                            </>
                        )}
                        {currentEnvironment === 'OVERHEATING' && (
                            <>
                                <li>Decreased RSI overbought threshold to 65 (standard: 70)</li>
                                <li>Reduced reliance on volatility channels (Bollinger Bands)</li>
                                <li>Increased sensitivity to price reversals</li>
                            </>
                        )}
                        {currentEnvironment === 'SLOWDOWN' && (
                            <>
                                <li>Reduced requirement for moving average confirmations</li>
                                <li>Increased focus on momentum indicators</li>
                                <li>Enhanced sensitivity to support/resistance levels</li>
                            </>
                        )}
                        {currentEnvironment === 'RECESSION' && (
                            <>
                                <li>Decreased RSI oversold threshold to 25 (standard: 30)</li>
                                <li>Reduced reliance on volatility channels</li>
                                <li>Increased caution on buy signals</li>
                            </>
                        )}
                        {currentEnvironment === 'RECOVERY' && (
                            <>
                                <li>Raised RSI oversold threshold to 35 (standard: 30)</li>
                                <li>Earlier entry signals for potential growth opportunities</li>
                                <li>Reduced sell signal sensitivity</li>
                            </>
                        )}
                        {currentEnvironment === 'NEUTRAL' && (
                            <>
                                <li>Using standard parameter settings</li>
                                <li>Balanced weighting of technical indicators</li>
                                <li>Normal risk management practices</li>
                            </>
                        )}
                    </ul>
                </div>

                <p className="mt-3 text-sm text-gray-600">
                    This strategy integrates real-time economic data from Alpha Vantage to adapt trading decisions to the current macroeconomic environment, enhancing resilience across different economic cycles.
                </p>
            </div>
        );
    };

    // Get recent trade signals (most recent first)
    const getRecentTradeSignals = (signals: any[]) => {
        if (!signals || signals.length === 0) return [];

        // Sort by date in descending order (most recent first)
        return [...signals].sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
    };

    // Function to handle signal click
    const handleSignalClick = (signal: any) => {
        setSelectedSignal(signal);
        setShowSignalDetails(true);
    };

    // Function to close the signal details dialog
    const handleCloseSignalDetails = () => {
        setShowSignalDetails(false);
    };

    return (
        <Card>
            <CardContent>
                <Typography variant="h5" component="div" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
                    <Box component="span" sx={{ color: 'primary.main', mr: 1 }}>
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M3 3V21H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                            <path d="M7 17L11 13L15 17L21 11" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                    </Box>
                    <strong>Strategy Analysis</strong>
                </Typography>
                <div className="mb-8">
                    <p className="mb-4 text-gray-600">
                        Select a trading strategy to analyze the performance and potential signals for{' '}
                        <span className="font-medium">{ticker}</span>.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                        <div>
                            <label htmlFor="strategy" className="block text-sm font-medium text-blue-600 mb-1">
                                Select Strategy
                            </label>
                            <select
                                id="strategy"
                                value={selectedStrategy}
                                onChange={(e) => setSelectedStrategy(e.target.value)}
                                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary focus:border-primary sm:text-sm rounded-md"
                                disabled={loading.strategies || loading.strategy}
                            >
                                {availableStrategies.map((strategy) => (
                                    <option key={strategy} value={strategy}>
                                        {getStrategyDisplayName(strategy)}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>

                    <div className="flex flex-wrap gap-2 mb-4">
                        <button
                            onClick={runStrategyAnalysis}
                            disabled={!selectedStrategy || loading.strategy}
                            className={`px-4 py-2 rounded ${!selectedStrategy || loading.strategy
                                ? 'bg-gray-300 cursor-not-allowed'
                                : 'bg-primary text-white hover:bg-primary-dark'
                                } transition-colors`}
                        >
                            {loading.strategy ? 'Analyzing...' : 'Run Strategy Analysis'}
                        </button>

                        {strategyAnalysis && (
                            <button
                                onClick={toggleStrategyDetails}
                                className="px-4 py-2 rounded bg-blue-500 text-white hover:bg-blue-600 transition-colors"
                            >
                                {showStrategyDetails ? 'Hide Strategy Details' : 'Show Strategy Details'}
                            </button>
                        )}
                    </div>

                    {error && <div className="mt-4 text-red-500">{error}</div>}

                    {strategyAnalysis && showStrategyDetails && (
                        <div className="mt-6 p-6 bg-white rounded-lg shadow-md">
                            <h3 className="text-lg font-bold mb-3">{strategyAnalysis?.strategy?.name}</h3>

                            <div className="mb-6">
                                <h4 className="text-lg font-semibold mb-2">Strategy Description</h4>
                                <p className="text-gray-700">{strategyAnalysis?.strategy?.description}</p>

                                {strategyAnalysis?.explanation && (
                                    <div className="mt-3 p-3 bg-blue-50 rounded-md">
                                        <p className="text-gray-800">{strategyAnalysis.explanation}</p>
                                    </div>
                                )}

                                {/* Add macro indicators display */}
                                {strategyAnalysis?.model_info && renderMacroIndicators(strategyAnalysis.model_info)}
                            </div>

                            <Divider className="my-4" />

                            <div className="mb-6">
                                <h4 className="text-lg font-semibold mb-3">Performance Metrics</h4>
                                {renderPerformanceMetrics(strategyAnalysis?.performance_metrics)}
                            </div>

                            {/* Moved chart to directly after performance metrics */}
                            <div className="mb-6">
                                <h4 className="text-lg font-semibold mb-3">Backtest Results</h4>
                                {renderBacktestChart(strategyAnalysis?.backtest_results)}
                            </div>

                            <Divider className="my-4" />

                            {strategyAnalysis?.trade_signals && strategyAnalysis.trade_signals.length > 0 && (
                                <div className="mb-4">
                                    <h4 className="text-lg font-semibold mb-3">Recent Trade Signals</h4>
                                    <div className="overflow-x-auto">
                                        <table className="min-w-full">
                                            <thead>
                                                <tr className="bg-gray-100">
                                                    <th className="px-4 py-2 text-left">Date</th>
                                                    <th className="px-4 py-2 text-left">Action</th>
                                                    <th className="px-4 py-2 text-left">Price</th>
                                                    <th className="px-4 py-2 text-left">Info</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {getRecentTradeSignals(strategyAnalysis.trade_signals).slice(0, 5).map((signal, index) => (
                                                    <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : ''}>
                                                        <td className="px-4 py-2">{signal.date}</td>
                                                        <td className={`px-4 py-2 font-medium ${signal.action === 'BUY' ? 'text-green-600' : 'text-red-600'}`}>
                                                            {signal.action}
                                                        </td>
                                                        <td className="px-4 py-2">${signal.price.toFixed(2)}</td>
                                                        <td className="px-4 py-2">
                                                            <IconButton
                                                                size="small"
                                                                onClick={() => handleSignalClick(signal)}
                                                                aria-label="signal details"
                                                            >
                                                                <InfoIcon fontSize="small" color="primary" />
                                                            </IconButton>
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                        {strategyAnalysis.trade_signals.length > 5 && (
                                            <div className="mt-2 text-right">
                                                <Typography variant="body2" color="primary">
                                                    Showing 5 of {strategyAnalysis.trade_signals.length} signals
                                                </Typography>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            <Divider className="my-4" />

                            <div className="bg-yellow-50 p-4 rounded-md">
                                <h4 className="text-lg font-semibold mb-2">Recommendation</h4>
                                <p className="text-gray-800">{strategyAnalysis?.recommendation}</p>
                                {strategyAnalysis?.note && (
                                    <p className="mt-2 text-gray-600 text-sm">{strategyAnalysis.note}</p>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </CardContent>

            {/* Signal Details Dialog */}
            <Dialog
                open={showSignalDetails}
                onClose={handleCloseSignalDetails}
                aria-labelledby="signal-details-dialog-title"
                maxWidth="sm"
                fullWidth
            >
                <DialogTitle id="signal-details-dialog-title">
                    <div className="flex justify-between items-center">
                        <Typography component="h2" variant="h6">
                            Trading Strategy Analysis
                        </Typography>
                        <IconButton
                            edge="end"
                            color="inherit"
                            onClick={handleCloseSignalDetails}
                            aria-label="close"
                        >
                            <CloseIcon />
                        </IconButton>
                    </div>
                </DialogTitle>
                <DialogContent dividers>
                    {selectedSignal && (
                        <div>
                            <Typography variant="h6" className="mb-4">
                                Signal Details
                            </Typography>
                            <Grid container spacing={2}>
                                <Grid item xs={4}>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Date:
                                    </Typography>
                                </Grid>
                                <Grid item xs={8}>
                                    <Typography variant="body1">{selectedSignal.date}</Typography>
                                </Grid>

                                <Grid item xs={4}>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Action:
                                    </Typography>
                                </Grid>
                                <Grid item xs={8}>
                                    <Typography
                                        variant="body1"
                                        className={selectedSignal.action === 'BUY' ? 'text-green-600 font-medium' : 'text-red-600 font-medium'}
                                    >
                                        {selectedSignal.action}
                                    </Typography>
                                </Grid>

                                <Grid item xs={4}>
                                    <Typography variant="subtitle2" color="textSecondary">
                                        Price:
                                    </Typography>
                                </Grid>
                                <Grid item xs={8}>
                                    <Typography variant="body1">${selectedSignal.price.toFixed(2)}</Typography>
                                </Grid>

                                <Grid item xs={12}>
                                    <Typography variant="subtitle2" color="textSecondary" className="mb-1">
                                        Reason:
                                    </Typography>
                                    <Paper variant="outlined" className="p-3">
                                        <Typography variant="body1">{selectedSignal.reason || 'No reason provided'}</Typography>
                                    </Paper>
                                </Grid>
                            </Grid>

                            <Typography variant="body2" color="textSecondary" className="mt-4 mb-2">
                                This signal was generated based on technical analysis of recent price movements
                                and volume patterns. The recommendation is derived from the latest trading day's signals.
                            </Typography>
                        </div>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={handleCloseSignalDetails} color="primary">
                        Close
                    </Button>
                </DialogActions>
            </Dialog>
        </Card>
    );
};

// Helper function - get strategy display name
function getStrategyDisplayName(strategyType: string): string {
    const strategyNames: Record<string, string> = {
        'moving_average': 'Moving Average Crossover',
        'rsi': 'Relative Strength Index (RSI)',
        'bollinger_bands': 'Bollinger Bands',
        'trend_following': 'Trend Following',
        'backtrader_macro': 'Backtrader Macro Strategy ✨'
    };

    return strategyNames[strategyType] || strategyType;
}

export default StrategyAnalysis;