'use client';

import React, { useEffect, useState, useRef } from 'react';
import { FaTimes, FaSpinner, FaInfoCircle } from 'react-icons/fa';
import { explainSingleText } from '@/services/huggingfaceService';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
import { Bar } from 'react-chartjs-2';

// 注册Chart.js组件
ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

interface ExplainabilityData {
    tokens: string[] | null;
    importanceValues: number[] | null;
    attentionMatrix: number[][] | null;
    importanceError: string | null;
}

// API响应类型定义
interface ExplainabilityResponse {
    sentiment: string;
    sentimentScore: number;
    explainability: ExplainabilityData;
    methodInfo?: {
        method: string;
        description: string;
    };
    processingTimes?: {
        total: number;
        sentiment: number;
        attention: number;
        importance: number;
    };
    ignored?: boolean;
    error?: boolean;
    message?: string;
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
    const chartRef = useRef<ChartJS<"bar"> | null>(null);

    useEffect(() => {
        let isMounted = true;
        let requestInProgress = false; // 追踪是否有请求正在进行中

        const fetchExplainability = async () => {
            // 添加更详细的调试信息来分析为什么API调用可能被阻止
            console.log('[调试] 准备获取可解释性数据, 状态检查:', {
                isOpen,
                textLength: text?.length || 0,
                textFirstChars: text?.substring(0, 20) || '空',
                requestInProgress
            });

            if (!isOpen) {
                console.log('[调试] 阻止API调用: 弹窗未打开 (isOpen=false)');
                return;
            }

            if (!text || text.trim() === '') {
                console.log('[调试] 阻止API调用: 文本为空');
                return;
            }

            if (requestInProgress) {
                console.log('[调试] 阻止API调用: 已有请求进行中');
                return;
            }

            // 标记请求开始
            requestInProgress = true;
            setLoading(true);
            setError(null);
            console.log('[调试] 开始获取可解释性数据... 文本:', text.substring(0, 50));

            try {
                // 确保text正确传递，并记录完整请求信息
                console.log('[调试] 正在调用API: /api/explain-single-text，文本长度:', text.length);

                const data = await explainSingleText(text) as ExplainabilityResponse;

                // 添加详细的响应结构日志
                console.log('[调试-详细] API响应结构:', {
                    sentiment: data.sentiment,
                    sentimentScore: data.sentimentScore,
                    methodInfo: data.methodInfo,
                    explainabilityPresent: !!data.explainability,
                    explainabilityStructure: data.explainability ? {
                        hasTokens: !!data.explainability.tokens,
                        hasImportanceValues: !!data.explainability.importanceValues,
                        hasAttentionMatrix: !!data.explainability.attentionMatrix,
                        importanceError: data.explainability.importanceError
                    } : 'missing',
                    processTimesPresent: !!data.processingTimes,
                    error: data.error,
                    message: data.message
                });

                console.log('[调试] API调用成功，接收到响应:', {
                    sentiment: data.sentiment,
                    sentimentScore: data.sentimentScore,
                    methodInfoPresent: !!data.methodInfo,
                    explainabilityPresent: !!data.explainability
                });

                // 如果组件已卸载，不进行状态更新
                if (!isMounted) {
                    console.log('[调试] 组件已卸载，不进行状态更新');
                    return;
                }

                // 如果响应被标记为忽略（来自旧请求），则不更新UI
                if (data.ignored) {
                    console.log('[调试] 忽略旧请求的响应，不更新UI');
                    return;
                }

                console.log('[调试] 检查explainability详细信息:', {
                    hasExplainability: !!data.explainability,
                    tokens: data.explainability?.tokens ? `${data.explainability.tokens.length}个` : 'null/undefined',
                    importanceValues: data.explainability?.importanceValues ? `${data.explainability.importanceValues.length}个` : 'null/undefined',
                });

                // 处理有效的响应数据
                if (data.explainability) {
                    console.log('[调试] 更新组件状态: explainabilityData, sentiment, sentimentScore');

                    // 添加更详细的日志记录
                    console.log('[调试详细] explainability完整结构:', JSON.stringify(data.explainability));
                    console.log('[调试] tokens数据:',
                        data.explainability.tokens ?
                            `存在，长度:${data.explainability.tokens.length}，类型:${Array.isArray(data.explainability.tokens) ? 'array' : typeof data.explainability.tokens}` :
                            '不存在或为null'
                    );
                    console.log('[调试] importanceValues数据:',
                        data.explainability.importanceValues ?
                            `存在，长度:${data.explainability.importanceValues.length}，类型:${Array.isArray(data.explainability.importanceValues) ? 'array' : typeof data.explainability.importanceValues}` :
                            '不存在或为null'
                    );

                    // 确保数据结构完整，即使部分数据缺失
                    const safeExplainabilityData: ExplainabilityData = {
                        tokens: Array.isArray(data.explainability.tokens) ? data.explainability.tokens : [],
                        importanceValues: Array.isArray(data.explainability.importanceValues) ? data.explainability.importanceValues : [],
                        attentionMatrix: data.explainability.attentionMatrix || null,
                        importanceError: data.explainability.importanceError || null
                    };

                    // 更新状态
                    setExplainabilityData(safeExplainabilityData);
                    setSentiment(data.sentiment || '未知');
                    setSentimentScore(typeof data.sentimentScore === 'number' ? data.sentimentScore : 0);
                } else {
                    console.log('[调试] 设置错误: 未能获取可解释性数据');
                    setError('未能获取可解释性数据');
                }
            } catch (err: any) {
                // 组件已卸载，不处理错误
                if (!isMounted) {
                    console.log('[调试] 组件已卸载，不处理错误');
                    return;
                }

                // 记录更详细的错误信息 - 完整输出错误对象
                console.error('[调试-完整错误]', JSON.stringify(err, null, 2));
                console.error('[调试] API调用失败，详细错误信息:', {
                    message: err.message,
                    status: err.response?.status,
                    statusText: err.response?.statusText,
                    data: err.response?.data,
                    axios: err.isAxiosError,
                    stack: err.stack?.substring(0, 300)
                });

                // 显示更有用的错误消息
                if (err.message && err.message.includes('后端服务已启动')) {
                    console.log('[调试] 设置错误: 后端服务连接失败');
                    setError('后端服务连接失败，请确保Flask后端已启动，并运行在GPU模式下');
                } else if (err.response && err.response.data && err.response.data.message) {
                    console.log('[调试] 设置错误: 来自响应数据');
                    setError(err.response.data.message);
                } else if (err.response && err.response.status === 500) {
                    console.log('[调试] 设置错误: 服务器内部错误');
                    setError(`服务器内部错误 (500): ${err.response.statusText || '未知错误'}`);
                } else if (err.message && err.message.includes('blocked by CORS policy')) {
                    console.log('[调试] 设置错误: CORS策略错误');
                    setError(`CORS错误: 浏览器阻止了跨域请求。请检查Flask后端的CORS配置。`);
                } else {
                    console.log('[调试] 设置错误: 通用错误消息');
                    setError(err.message || '分析过程中出错，无法获取SHAP和self-attention数据');
                }
            } finally {
                // 组件已卸载，不更新状态
                if (isMounted) {
                    console.log('[调试] 设置loading=false');
                    setLoading(false);
                } else {
                    console.log('[调试] 组件已卸载，不更新loading状态');
                }
                // 标记请求结束
                requestInProgress = false;
            }
        };

        // 设置一个更长的延迟以确保前一个请求完成
        console.log('[调试] 设置延迟200ms的计时器');
        const timer = setTimeout(() => {
            console.log('[调试] 计时器触发，准备调用fetchExplainability');
            fetchExplainability();
        }, 200);

        // 清理函数
        return () => {
            console.log('[调试] 清理函数执行: isMounted=false, clearTimeout');
            isMounted = false;
            clearTimeout(timer);
        };
    }, [isOpen, text]);

    // 渲染SHAP令牌和对应的值
    const renderShapTokens = () => {
        if (!explainabilityData?.tokens || !explainabilityData?.importanceValues ||
            explainabilityData.tokens.length === 0 || explainabilityData.importanceValues.length === 0) {
            return <p>没有可用的SHAP值数据</p>;
        }

        // 确保tokens和importanceValues长度匹配
        const minLength = Math.min(
            explainabilityData.tokens.length,
            explainabilityData.importanceValues.length
        );

        return (
            <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">影响情感判断的关键词</h3>
                <div className="grid grid-cols-1 gap-2">
                    {explainabilityData.tokens.slice(0, minLength).map((token, index) => {
                        // 安全访问importanceValues
                        const value = explainabilityData.importanceValues && index < minLength ?
                            explainabilityData.importanceValues[index] : 0;
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
                    正值表示对正面情感有贡献，负值表示对负面情感有贡献
                </p>
            </div>
        );
    };

    // 渲染Self-Attention热力图
    const renderAttentionHeatmap = () => {
        if (!explainabilityData?.attentionMatrix || !explainabilityData?.tokens) {
            return <p>没有可用的注意力矩阵数据</p>;
        }

        const matrix = explainabilityData.attentionMatrix;
        const tokens = explainabilityData.tokens;

        // 简化的交互式热力图
        return (
            <div className="mb-4 overflow-x-auto">
                <h3 className="text-lg font-semibold mb-2">Self-Attention热力图</h3>
                <div className="flex">
                    <div className="w-24"></div>
                    <div className="flex">
                        {tokens.map((token, i) => (
                            <div key={`header-${i}`} className="w-10 text-center text-xs rotate-45 origin-bottom-left">
                                {token}
                            </div>
                        ))}
                    </div>
                </div>
                {tokens.map((rowToken, i) => (
                    <div key={`row-${i}`} className="flex">
                        <div className="w-24 overflow-hidden text-ellipsis whitespace-nowrap text-xs">
                            {rowToken}
                        </div>
                        <div className="flex">
                            {matrix[i] && matrix[i].map((value, j) => {
                                return (
                                    <div
                                        key={`cell-${i}-${j}`}
                                        className="w-10 h-10 border border-gray-100 flex items-center justify-center text-xs text-white"
                                        style={{
                                            backgroundColor: `rgba(66, 133, 244, ${value})`
                                        }}
                                        title={`${rowToken} → ${tokens[j]}: ${value.toFixed(2)}`}
                                    >
                                        {value > 0.2 ? value.toFixed(1) : ''}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                ))}
                <p className="text-sm text-gray-600 mt-2">
                    颜色越深表示注意力权重越高
                </p>
            </div>
        );
    };

    // 使用Chart.js渲染特征重要性图
    const renderImportanceChart = () => {
        if (!explainabilityData?.tokens || !explainabilityData?.importanceValues) {
            return <p>没有可用的特征重要性数据</p>;
        }

        const tokens = explainabilityData.tokens;
        const importanceValues = explainabilityData.importanceValues;

        // 只选择前15个最重要的token进行展示
        const tokenImportance = tokens.map((token, index) => ({
            token,
            importance: importanceValues[index]
        }));

        // 按重要性绝对值排序
        tokenImportance.sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance));

        // 取前15个
        const topTokens = tokenImportance.slice(0, 15);

        // 为Chart.js准备数据
        const chartData = {
            labels: topTokens.map(item => item.token),
            datasets: [
                {
                    label: '特征重要性',
                    data: topTokens.map(item => item.importance),
                    backgroundColor: topTokens.map(item =>
                        item.importance > 0 ? 'rgba(75, 192, 75, 0.8)' : 'rgba(255, 99, 132, 0.8)'
                    ),
                    borderColor: topTokens.map(item =>
                        item.importance > 0 ? 'rgb(75, 192, 75)' : 'rgb(255, 99, 132)'
                    ),
                    borderWidth: 1,
                }
            ]
        };

        const chartOptions = {
            indexAxis: 'y' as const,
            responsive: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top' as const,
                },
                title: {
                    display: true,
                    text: '特征重要性 (SHAP值)',
                    font: {
                        size: 16
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function (context: any) {
                            const value = context.raw;
                            return `重要性: ${value.toFixed(4)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '重要性 (SHAP值)'
                    },
                    grid: {
                        display: true
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Token'
                    },
                    grid: {
                        display: false
                    }
                }
            }
        };

        return (
            <div className="mb-4">
                <h3 className="text-lg font-semibold mb-2">特征重要性可视化</h3>
                <div style={{ height: '400px' }}>
                    <Bar
                        ref={chartRef}
                        data={chartData}
                        options={chartOptions}
                    />
                </div>
                <p className="text-sm text-gray-600 mt-2">
                    绿色表示对正面情感有贡献，红色表示对负面情感有贡献。值越大表示影响越大。
                </p>
            </div>
        );
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 overflow-y-auto">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-auto m-4">
                <div className="flex items-center justify-between p-4 border-b border-gray-200">
                    <h3 className="text-xl font-bold">文本可解释性分析</h3>
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
                        <h3 className="text-lg font-semibold">分析文本</h3>
                        <p className="p-3 bg-gray-100 rounded-md">{text}</p>
                    </div>

                    <div className="mb-4">
                        <h3 className="text-lg font-semibold">情感分析结果</h3>
                        <div className="flex items-center mt-2">
                            <div
                                className={`px-3 py-1 rounded-md font-medium ${sentiment === 'Positive' ? 'bg-green-100 text-green-800' :
                                    sentiment === 'Negative' ? 'bg-red-100 text-red-800' :
                                        'bg-gray-100 text-gray-800'
                                    }`}
                            >
                                {sentiment === 'Positive' ? '正面' :
                                    sentiment === 'Negative' ? '负面' : '中性'}
                            </div>
                            <div className="ml-4">
                                情感分数: {sentimentScore.toFixed(2)}
                                <span className="text-sm text-gray-500 ml-1">(-1到1之间，正值表示正面情感)</span>
                            </div>
                        </div>
                    </div>

                    {loading ? (
                        <div className="flex flex-col items-center justify-center py-12">
                            <FaSpinner className="animate-spin text-blue-500 text-4xl mb-4" />
                            <p>正在分析文本，请稍候...</p>
                            <p className="text-sm text-gray-500 mt-2">分析处理可能需要30-60秒，请耐心等待</p>
                            <div className="mt-4 w-full max-w-md bg-gray-200 rounded-full h-2.5">
                                <div className="bg-blue-500 h-2.5 rounded-full animate-pulse w-3/4"></div>
                            </div>
                            <p className="text-xs text-gray-400 mt-2">正在进行集成梯度计算，这是一个复杂的数学过程</p>
                        </div>
                    ) : error ? (
                        <div className="bg-red-100 border border-red-200 text-red-700 px-4 py-3 rounded-md">
                            <div className="flex">
                                <div className="flex-shrink-0">
                                    <FaInfoCircle className="h-5 w-5 text-red-500" />
                                </div>
                                <div className="ml-3">
                                    <p className="font-medium">分析过程中出错</p>
                                    <p className="text-sm">{error}</p>
                                    <details className="mt-2 text-xs">
                                        <summary>调试信息</summary>
                                        <pre className="mt-2 whitespace-pre-wrap">
                                            后端URL: http://localhost:5000/api/explain-single-text
                                            文本: {text?.substring(0, 100)}...
                                            文本长度: {text?.length || 0}

                                            如果您看到"后端服务连接失败"：
                                            1. 请确认Flask服务运行在端口5000
                                            2. 检查控制台是否有CORS错误
                                            3. 尝试直接访问 http://localhost:5000/health
                                        </pre>
                                    </details>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <>
                            {/* 使用Chart.js渲染的特征重要性图表 */}
                            {renderImportanceChart()}

                            {/* SHAP令牌和对应的值 */}
                            {renderShapTokens()}

                            {/* Self-Attention热力图 */}
                            {renderAttentionHeatmap()}

                            {/* 错误信息显示 */}
                            {explainabilityData?.importanceError && (
                                <div className="bg-yellow-100 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-md mt-4">
                                    <div className="flex">
                                        <div className="flex-shrink-0">
                                            <FaInfoCircle className="h-5 w-5 text-yellow-500" />
                                        </div>
                                        <div className="ml-3">
                                            <p className="font-medium">部分分析结果不可用</p>
                                            {explainabilityData?.importanceError && (
                                                <p className="text-sm">特征重要性计算出错: {explainabilityData.importanceError}</p>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* Token 重要性可视化 */}
                            {!loading && !error && explainabilityData && explainabilityData.tokens && explainabilityData.importanceValues && (
                                <div className="mt-4">
                                    <h3 className="text-lg font-medium">Token重要性</h3>
                                    <div className="text-sm text-gray-500 mb-2">
                                        分析文本中每个token对情感判断的贡献度
                                    </div>
                                    <div className="border rounded p-3 bg-white">
                                        <div className="flex flex-wrap">
                                            {explainabilityData.tokens.map((token, index) => {
                                                // 只有当importanceValues数组存在且长度足够时才使用对应值
                                                const importance = explainabilityData.importanceValues &&
                                                    index < explainabilityData.importanceValues.length
                                                    ? explainabilityData.importanceValues[index]
                                                    : 0;

                                                // 计算背景色 - 正面绿色，负面红色
                                                const bgColor = importance > 0
                                                    ? `rgba(0, 128, 0, ${Math.min(Math.abs(importance) * 5, 0.9)})`
                                                    : `rgba(255, 0, 0, ${Math.min(Math.abs(importance) * 5, 0.9)})`;

                                                return (
                                                    <div
                                                        key={index}
                                                        className="px-1 py-0.5 m-0.5 rounded text-sm"
                                                        style={{
                                                            backgroundColor: bgColor,
                                                            color: Math.abs(importance) > 0.15 ? 'white' : 'black'
                                                        }}
                                                        title={`重要性: ${importance.toFixed(3)}`}
                                                    >
                                                        {token}
                                                    </div>
                                                );
                                            })}
                                        </div>
                                    </div>
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
                        关闭
                    </button>
                </div>
            </div>
        </div>
    );
};

export default TextExplainabilityModal; 