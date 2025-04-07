import React from 'react';
import { FaExclamationTriangle } from 'react-icons/fa';

interface ErrorStateProps {
    message: string;
}

const ErrorState: React.FC<ErrorStateProps> = ({ message }) => {
    return (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
            <div className="flex items-center">
                <FaExclamationTriangle className="mr-2" />
                <strong className="font-bold">Error!</strong>
                <span className="block sm:inline ml-2">{message}</span>
            </div>
        </div>
    );
};

export default ErrorState; 