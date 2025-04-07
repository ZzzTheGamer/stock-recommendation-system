'use client';

import React, { useEffect, useRef } from 'react';
import { FaTimes } from 'react-icons/fa';

interface ExplanationModalProps {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    content: React.ReactNode;
}

const ExplanationModal: React.FC<ExplanationModalProps> = ({
    isOpen,
    onClose,
    title,
    content,
}) => {
    const modalRef = useRef<HTMLDivElement>(null);

    // Handle click outside to close modal
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
                onClose();
            }
        };

        if (isOpen) {
            document.addEventListener('mousedown', handleClickOutside);
            // Prevent background scrolling
            document.body.style.overflow = 'hidden';
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
            // Restore background scrolling
            document.body.style.overflow = 'auto';
        };
    }, [isOpen, onClose]);

    // Handle ESC key to close modal
    useEffect(() => {
        const handleEscKey = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                onClose();
            }
        };

        if (isOpen) {
            document.addEventListener('keydown', handleEscKey);
        }

        return () => {
            document.removeEventListener('keydown', handleEscKey);
        };
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
            <div
                ref={modalRef}
                className="bg-white rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-auto"
            >
                <div className="flex items-center justify-between p-4 border-b border-gray-200">
                    <h3 className="text-xl font-bold">{title}</h3>
                    <button
                        onClick={onClose}
                        className="text-gray-500 hover:text-gray-700 transition-colors"
                        aria-label="Close"
                    >
                        <FaTimes size={20} />
                    </button>
                </div>

                <div className="p-6">
                    {content}
                </div>

                <div className="p-4 border-t border-gray-200 text-right">
                    <button
                        onClick={onClose}
                        className="btn btn-secondary"
                    >
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ExplanationModal; 