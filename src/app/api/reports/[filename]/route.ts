import { NextRequest, NextResponse } from 'next/server';

// Simulate PDF generation
export async function GET(
    request: NextRequest,
    { params }: { params: { filename: string } }
) {
    try {
        const filename = params.filename;

        // Now we just return a simple text as PDF
        const content = `
      Financial Analysis Report
      
      This is a placeholder for a real PDF report.
      In a production environment, this would be a properly formatted PDF document
      with financial analysis for the requested stock.
      
      Filename: ${filename}
      Generated: ${new Date().toISOString()}
    `;

        // Create a simple text response as PDF
        return new NextResponse(content, {
            headers: {
                'Content-Type': 'application/pdf',
                'Content-Disposition': `attachment; filename="${filename}"`,
            },
        });
    } catch (error) {
        console.error('Error generating PDF:', error);
        return NextResponse.json(
            { error: 'Failed to generate PDF report' },
            { status: 500 }
        );
    }
} 