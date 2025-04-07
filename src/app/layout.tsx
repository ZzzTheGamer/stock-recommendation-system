import React from 'react';
import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
    title: 'Stock Recommendation System',
    description: 'A comprehensive stock analysis and recommendation system',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <body className="bg-gray-50 min-h-screen">
                <main className="container mx-auto px-4 py-8">
                    {children}
                </main>
            </body>
        </html>
    );
} 