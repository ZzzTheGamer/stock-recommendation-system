import { NextResponse } from 'next/server';

export async function POST(request: Request) {
    try {
        // Parse request body
        const body = await request.json();
        const { symbol, companyName, industry, financialData } = body;

        if (!symbol || !companyName || !industry || !financialData) {
            return NextResponse.json(
                { error: 'Missing required fields' },
                { status: 400 }
            );
        }

        console.log(`[API] Requesting financial analysis report for ${symbol}`);

        // Return simulated data
        return NextResponse.json({
            summary: `${companyName} (${symbol}) shows mixed financial performance with some areas of strength and concern.`,
            strengths: '1. Stable revenue growth\n2. Healthy profit margins\n3. Strong market position',
            weaknesses: '1. Increasing debt levels\n2. Cash flow challenges\n3. Competitive pressure',
            metricsAnalysis: 'Key financial metrics indicate a stable but challenging financial position.',
            recommendation: 'Hold',
            rating: 'Hold',
            apiCallSuccess: true,
            isSimulated: true,
            keyMetrics: financialData,
            analysisDate: new Date().toISOString()
        });
    } catch (error: any) {
        console.error('[API] Error generating analysis:', error.message);

        // Return error and simulated data
        return NextResponse.json({
            summary: 'Error generating analysis.',
            strengths: '',
            weaknesses: '',
            metricsAnalysis: '',
            recommendation: 'Hold',
            rating: 'Hold',
            apiCallSuccess: false,
            isSimulated: true,
            keyMetrics: {},
            analysisDate: new Date().toISOString()
        });
    }
} 