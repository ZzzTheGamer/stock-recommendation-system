import { NextResponse } from 'next/server';
import { analyzeStrategy, getAvailableStrategies } from '../../../services/strategyService';

export async function GET(request: Request) {
    try {
        const { searchParams } = new URL(request.url);
        const action = searchParams.get('action');
        const symbol = searchParams.get('symbol');
        const strategyType = searchParams.get('strategy');

        if (action === 'list') {
            // Return list of available strategies (excluding ML strategies)
            const strategies = [
                'moving_average',
                'rsi',
                'bollinger_bands',
                'trend_following',
                'backtrader_macro'
            ];
            return NextResponse.json({ strategies });
        } else if (action === 'analyze' && symbol && strategyType) {
            // Analyze a specific strategy for a symbol
            try {
                const result = await analyzeStrategy(symbol, strategyType);
                return NextResponse.json(result);
            } catch (error: any) {
                console.error(`[API] 策略分析失败: ${error.message}`);
                return NextResponse.json(
                    { error: error.message || '策略分析过程中发生错误' },
                    { status: 500 }
                );
            }
        } else {
            return NextResponse.json(
                { error: 'Invalid parameters. Required: action=[list|analyze], symbol (for analyze), strategy (for analyze)' },
                { status: 400 }
            );
        }
    } catch (error: any) {
        console.error('[API] Strategy analysis error:', error.message);
        return NextResponse.json(
            { error: error.message || 'An error occurred during strategy analysis' },
            { status: 500 }
        );
    }
}