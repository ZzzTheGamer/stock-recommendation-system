import axios from 'axios';

// API request timeout setting (milliseconds)
const API_TIMEOUT = 1800000; // 30 minutes timeout, ensuring TreeSHAP analysis has enough time to complete

// Task polling interval (milliseconds)
const TASK_POLL_INTERVAL = 3000; // Check task status every 3 seconds

// Result type definitions
interface SuccessResult<T> {
    success: true;
    [key: string]: any;
}

interface ErrorResult {
    success: false;
    error: string;
    error_type: string;
}

type ApiResult<T> = SuccessResult<T> | ErrorResult;

// Task status types
interface TaskProgress {
    progress: number;
    message: string;
}

interface TaskStatus {
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: TaskProgress;
    result?: any;
    error?: string;
}

// More detailed error handling
const handleApiError = (error: any, method: string): ErrorResult => {
    console.error(`[Explainability] ${method} analysis error:`, error);

    let errorMessage = '';
    let errorType = 'UNKNOWN_ERROR';

    if (axios.isAxiosError(error)) {
        if (error.response) {
            // Server responded, but status code is not 2xx
            const serverError = error.response.data;
            errorMessage = serverError.error || `Server error (${error.response.status})`;
            errorType = serverError.error_type || `HTTP_${error.response.status}`;
            console.error('[Detailed error]', serverError);
        } else if (error.request) {
            // Request was sent but no response received
            errorMessage = 'Request timeout or server did not respond';
            errorType = 'REQUEST_TIMEOUT';
        } else {
            // Error in request setup
            errorMessage = `Request configuration error: ${error.message}`;
            errorType = 'REQUEST_SETUP_ERROR';
        }
    } else {
        // Non-Axios error
        errorMessage = error instanceof Error ? error.message : String(error);
    }

    return {
        success: false,
        error: errorMessage,
        error_type: errorType
    };
};

// ===== Task status management functions =====

/**
 * Get current status of a task
 */
export const getTaskStatus = async (taskId: string): Promise<ApiResult<TaskStatus>> => {
    try {
        const response = await axios.get(`/api/tasks/${taskId}`, {
            timeout: 10000 // 10 seconds timeout
        });

        if (response.data.success && response.data.taskStatus) {
            return {
                success: true,
                ...response.data.taskStatus
            };
        } else {
            return {
                success: false,
                error: response.data.error || 'Failed to get task status',
                error_type: response.data.error_type || 'TASK_STATUS_ERROR'
            };
        }
    } catch (error) {
        return handleApiError(error, 'Get task status');
    }
};

/**
 * Wait for task completion, periodically returning progress
 * @param taskId Task ID
 * @param onProgress Progress callback function
 */
export const waitForTaskCompletion = async (
    taskId: string,
    onProgress?: (progress: TaskProgress) => void
): Promise<ApiResult<any>> => {
    try {
        let isCompleted = false;
        let result: ApiResult<any> = {
            success: false,
            error: 'Task not completed',
            error_type: 'TASK_INCOMPLETE'
        };

        // Poll task status until completed
        while (!isCompleted) {
            const taskStatus = await getTaskStatus(taskId);

            if (!taskStatus.success) {
                return taskStatus;
            }

            // Callback progress
            if (onProgress && taskStatus.progress) {
                onProgress(taskStatus.progress);
            }

            // Check if task is complete
            if (taskStatus.status === 'completed' && taskStatus.result) {
                isCompleted = true;
                result = {
                    success: true,
                    ...taskStatus.result
                };
            } else if (taskStatus.status === 'failed') {
                isCompleted = true;
                result = {
                    success: false,
                    error: taskStatus.error || 'Task execution failed',
                    error_type: 'TASK_EXECUTION_FAILED'
                };
            } else {
                // Wait for next poll
                await new Promise(resolve => setTimeout(resolve, TASK_POLL_INTERVAL));
            }
        }

        return result;
    } catch (error) {
        return handleApiError(error, 'Wait for task completion');
    }
};

// ===== SHAP related functions =====

/**
 * Explain Mistral model's financial recommendation using KernelSHAP
 */
export const explainMistralWithSHAP = async (
    financialData: any,
    recommendation: string
) => {
    try {
        console.log('[Explainability] Starting calculation of SHAP values for Mistral recommendation');

        // Extract key financial metrics
        const features = [
            'revenueGrowth',
            'netIncomeGrowth',
            'currentRatio',
            'debtToEquity',
            'returnOnEquity',
            'peRatio'
        ];

        // Generate backend request data
        const requestData = {
            features: features.map(feature => ({
                name: feature,
                value: parseFloat(financialData[feature] || '0')
            })),
            recommendation
        };

        // Call backend API to calculate SHAP values
        const response = await axios.post('/api/explain-mistral/shap', requestData);

        console.log('[Explainability] SHAP analysis completed:', response.data);

        return {
            success: true,
            shapValues: response.data
        };
    } catch (error) {
        return handleApiError(error, 'SHAP');
    }
};

/**
 * Explain Mistral model's financial recommendation using TreeSHAP
 */
export const explainMistralWithTreeSHAP = async (
    financialData: any,
    recommendation: string
) => {
    try {
        console.log('[Explainability] Starting TreeSHAP analysis for Mistral recommendation');

        // Ensure all metrics exist
        const features = [
            'revenueGrowth',
            'netIncomeGrowth',
            'currentRatio',
            'debtToEquity',
            'returnOnEquity',
            'peRatio'
        ];

        features.forEach(feature => {
            if (financialData[feature] === undefined) {
                financialData[feature] = 0;
            }
        });

        // Generate backend request data
        const requestData = {
            financialData,
            recommendation
        };

        // Call backend API for TreeSHAP analysis
        const response = await axios.post('/api/explain-mistral/treeshap', requestData, {
            timeout: API_TIMEOUT // Set uniform timeout
        });

        console.log('[Explainability] TreeSHAP analysis completed:', response.data);

        return {
            success: true,
            method: 'TreeSHAP',
            ...response.data
        };
    } catch (error) {
        return handleApiError(error, 'TreeSHAP');
    }
};

/**
 * Perform TreeSHAP analysis asynchronously, returning task ID and progress information
 */
export const explainMistralWithTreeSHAPAsync = async (
    financialData: any,
    recommendation: string,
    onProgress?: (progress: TaskProgress) => void
) => {
    try {
        console.log('[Explainability] Starting async TreeSHAP analysis task');

        // Ensure all metrics exist
        const features = [
            'revenueGrowth',
            'netIncomeGrowth',
            'currentRatio',
            'debtToEquity',
            'returnOnEquity',
            'peRatio'
        ];

        features.forEach(feature => {
            if (financialData[feature] === undefined) {
                financialData[feature] = 0;
            }
        });

        // Generate backend request data
        const requestData = {
            financialData,
            recommendation
        };

        // Create async task
        const createTaskResponse = await axios.post('/api/explain-mistral/treeshap-async', requestData, {
            timeout: 30000 // Shorter timeout for task creation
        });

        if (!createTaskResponse.data.success) {
            throw new Error(createTaskResponse.data.error || 'Failed to create async task');
        }

        const taskId = createTaskResponse.data.taskId;
        console.log(`[Explainability] Created async TreeSHAP analysis task: ${taskId}`);

        // Start waiting for task completion
        const result = await waitForTaskCompletion(taskId, onProgress);

        if (!result.success) {
            throw new Error(result.error || 'TreeSHAP analysis task failed');
        }

        console.log('[Explainability] Async TreeSHAP analysis completed:', result);

        return {
            method: 'TreeSHAP',
            ...result
        };
    } catch (error) {
        return handleApiError(error, 'Async TreeSHAP');
    }
};

/**
 * Explain Mistral model's financial recommendation using LIME method
 */
export const explainMistralWithLIME = async (
    financialData: any,
    recommendation: string
) => {
    try {
        console.log('[Explainability] Starting LIME method analysis for Mistral recommendation');

        // Extract key financial metrics
        const features = [
            'revenueGrowth',
            'netIncomeGrowth',
            'currentRatio',
            'debtToEquity',
            'returnOnEquity',
            'peRatio'
        ];

        // Ensure all metrics exist
        features.forEach(feature => {
            if (financialData[feature] === undefined) {
                financialData[feature] = 0;
            }
        });

        // Generate backend request data
        const requestData = {
            financialData,
            recommendation
        };

        // Call backend API for LIME analysis
        const response = await axios.post('/api/explain-mistral/lime', requestData, {
            timeout: API_TIMEOUT // Set uniform timeout
        });

        console.log('[Explainability] LIME analysis completed:', response.data);

        return {
            success: true,
            method: 'LIME',
            ...response.data
        };
    } catch (error) {
        return handleApiError(error, 'LIME');
    }
};

// ===== Text explanation related functions =====

/**
 * Explain text generated by Mistral model
 */
export const explainMistralText = async (
    text: string,
    section: 'summary' | 'recommendation'
): Promise<ApiResult<{ textAnalysis: any }>> => {
    try {
        console.log(`[Explainability] Starting analysis of Mistral ${section} text`);

        if (!text) {
            throw new Error('Text content is empty');
        }

        // Call backend API to analyze text
        const response = await axios.post('/api/explain-mistral/text', {
            text,
            section
        }, {
            timeout: API_TIMEOUT // Set uniform timeout
        });

        console.log('[Explainability] Text analysis completed');

        return {
            success: true,
            textAnalysis: response.data
        };
    } catch (error) {
        return handleApiError(error, `Text(${section})`);
    }
};

/**
 * Get complete explanation analysis in one call (async method)
 */
export const getFullMistralExplanationAsync = async (
    financialData: any,
    recommendation: string,
    summaryText: string,
    recommendationText: string,
    explainMethod: 'SHAP' | 'LIME' | 'TreeSHAP' = 'TreeSHAP',
    onProgress?: (progress: TaskProgress) => void
): Promise<ApiResult<{ explanation: any }>> => {
    try {
        console.log(`[Explainability] Starting complete explanation analysis using ${explainMethod} method (async mode)`);

        // Add progress reporting
        const progress = {
            start: Date.now(),
            status: 'Starting analysis',
            stage: 0,
        };

        // Process financial data, ensure all fields are numbers
        const processedFinancialData = {
            revenueGrowth: parseFloat(financialData.revenueGrowth || '0'),
            netIncomeGrowth: parseFloat(financialData.netIncomeGrowth || '0'),
            currentRatio: parseFloat(financialData.currentRatio || '0'),
            debtToEquity: parseFloat(financialData.debtToEquity || '0'),
            returnOnEquity: parseFloat(financialData.returnOnEquity || '0'),
            peRatio: parseFloat(financialData.peRatio || '0')
        };

        // Call different endpoints based on selected method
        let featureAnalysis: any;

        progress.status = `Performing ${explainMethod} analysis`;
        progress.stage = 1;
        console.log(`[Explainability] ${progress.status}...`);

        // Set feature analysis progress conversion function
        const featureProgressAdapter = (featureProgress: TaskProgress) => {
            if (onProgress) {
                // Feature analysis takes 70% of total progress
                const overallProgress = Math.floor(featureProgress.progress * 0.7);
                onProgress({
                    progress: overallProgress,
                    message: featureProgress.message
                });
            }
        };

        // Use TreeSHAP for async analysis
        if (explainMethod === 'TreeSHAP') {
            featureAnalysis = await explainMistralWithTreeSHAPAsync(
                processedFinancialData,
                recommendation,
                featureProgressAdapter
            );
        } else if (explainMethod === 'LIME') {
            // LIME temporarily uses synchronous method
            featureAnalysis = await explainMistralWithLIME(processedFinancialData, recommendation);
        } else {
            // Default SHAP temporarily uses synchronous method
            featureAnalysis = await explainMistralWithSHAP(processedFinancialData, recommendation);
        }

        if (!featureAnalysis.success) {
            throw new Error(`${explainMethod} analysis failed: ${featureAnalysis.error}`);
        }

        // Calculate analysis time
        const analysisTime = Date.now() - progress.start;
        console.log(`[Explainability] ${explainMethod} analysis completed, time taken: ${analysisTime}ms`);

        // Analyze text (if provided)
        let textAnalysis: Record<string, any> = {};

        if (summaryText) {
            if (onProgress) {
                onProgress({
                    progress: 75,
                    message: 'Analyzing summary text...'
                });
            }

            progress.status = 'Analyzing summary text';
            progress.stage = 2;
            console.log(`[Explainability] ${progress.status}...`);

            const summaryAnalysis = await explainMistralText(summaryText, 'summary');
            if (summaryAnalysis.success && 'textAnalysis' in summaryAnalysis) {
                textAnalysis.summary = summaryAnalysis.textAnalysis;
            }
        }

        if (recommendationText) {
            if (onProgress) {
                onProgress({
                    progress: 85,
                    message: 'Analyzing investment recommendation text...'
                });
            }

            progress.status = 'Analyzing investment recommendation text';
            progress.stage = 3;
            console.log(`[Explainability] ${progress.status}...`);

            const recAnalysis = await explainMistralText(recommendationText, 'recommendation');
            if (recAnalysis.success && 'textAnalysis' in recAnalysis) {
                textAnalysis.recommendation = recAnalysis.textAnalysis;
            }
        }

        if (onProgress) {
            onProgress({
                progress: 95,
                message: 'Integrating analysis results...'
            });
        }

        // Combine all analysis results
        const fullExplanation = {
            method: explainMethod,
            ...featureAnalysis,
            textAnalysis,
            processingTime: Date.now() - progress.start
        };

        console.log(`[Explainability] Complete explanation analysis generation finished, total time: ${fullExplanation.processingTime}ms`);

        if (onProgress) {
            onProgress({
                progress: 100,
                message: 'Analysis completed'
            });
        }

        return {
            success: true,
            explanation: fullExplanation
        };
    } catch (error) {
        return handleApiError(error, 'Explanation analysis generation');
    }
};

/**
 * Get complete explanation analysis in one call
 */
export const getFullMistralExplanation = async (
    financialData: any,
    recommendation: string,
    summaryText: string,
    recommendationText: string,
    explainMethod: 'SHAP' | 'LIME' | 'TreeSHAP' = 'TreeSHAP'
): Promise<ApiResult<{ explanation: any }>> => {
    try {
        console.log(`[Explainability] Starting complete explanation analysis using ${explainMethod} method`);

        // Add progress reporting
        const progress = {
            start: Date.now(),
            status: 'Starting analysis',
            stage: 0,
        };

        // Process financial data, ensure all fields are numbers
        const processedFinancialData = {
            revenueGrowth: parseFloat(financialData.revenueGrowth || '0'),
            netIncomeGrowth: parseFloat(financialData.netIncomeGrowth || '0'),
            currentRatio: parseFloat(financialData.currentRatio || '0'),
            debtToEquity: parseFloat(financialData.debtToEquity || '0'),
            returnOnEquity: parseFloat(financialData.returnOnEquity || '0'),
            peRatio: parseFloat(financialData.peRatio || '0')
        };

        // Call different endpoints based on selected method
        let featureAnalysis: any;

        progress.status = `Performing ${explainMethod} analysis`;
        progress.stage = 1;
        console.log(`[Explainability] ${progress.status}...`);

        if (explainMethod === 'LIME') {
            featureAnalysis = await explainMistralWithLIME(processedFinancialData, recommendation);
        } else if (explainMethod === 'TreeSHAP') {
            featureAnalysis = await explainMistralWithTreeSHAP(processedFinancialData, recommendation);
        } else { // Default SHAP
            featureAnalysis = await explainMistralWithSHAP(processedFinancialData, recommendation);
        }

        if (!featureAnalysis.success) {
            throw new Error(`${explainMethod} analysis failed: ${featureAnalysis.error}`);
        }

        // Calculate analysis time
        const analysisTime = Date.now() - progress.start;
        console.log(`[Explainability] ${explainMethod} analysis completed, time taken: ${analysisTime}ms`);

        // Analyze text (if provided)
        let textAnalysis: Record<string, any> = {};

        if (summaryText) {
            progress.status = 'Analyzing summary text';
            progress.stage = 2;
            console.log(`[Explainability] ${progress.status}...`);

            const summaryAnalysis = await explainMistralText(summaryText, 'summary');
            if (summaryAnalysis.success && 'textAnalysis' in summaryAnalysis) {
                textAnalysis.summary = summaryAnalysis.textAnalysis;
            }
        }

        if (recommendationText) {
            progress.status = 'Analyzing investment recommendation text';
            progress.stage = 3;
            console.log(`[Explainability] ${progress.status}...`);

            const recAnalysis = await explainMistralText(recommendationText, 'recommendation');
            if (recAnalysis.success && 'textAnalysis' in recAnalysis) {
                textAnalysis.recommendation = recAnalysis.textAnalysis;
            }
        }

        // Combine all analysis results
        const fullExplanation = {
            method: explainMethod,
            ...featureAnalysis,
            textAnalysis,
            processingTime: Date.now() - progress.start
        };

        console.log(`[Explainability] Complete explanation analysis generation finished, total time: ${fullExplanation.processingTime}ms`);

        return {
            success: true,
            explanation: fullExplanation
        };
    } catch (error) {
        return handleApiError(error, 'Explanation analysis generation');
    }
}; 