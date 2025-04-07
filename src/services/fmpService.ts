import axios from 'axios';

const API_KEY = process.env.NEXT_PUBLIC_FMP_API_KEY || '';
const BASE_URL = 'https://financialmodelingprep.com/api/v3';

/**
 * Get company profile
 * @param symbol Stock symbol
 */
export const getCompanyProfile = async (symbol: string) => {
    try {
        const response = await axios.get(`${BASE_URL}/profile/${symbol}`, {
            params: {
                apikey: API_KEY,
            },
        });
        return response.data[0];
    } catch (error) {
        console.error('Error fetching company profile:', error);
        return null;
    }
};

/**
 * Get company financial statements - Income Statement
 * @param symbol Stock symbol
 * @param period Period (quarter or annual)
 * @param limit Limit on number of results returned
 */
export const getIncomeStatement = async (symbol: string, period: string = 'annual', limit: number = 5) => {
    try {
        const response = await axios.get(`${BASE_URL}/income-statement/${symbol}`, {
            params: {
                period,
                limit,
                apikey: API_KEY,
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching income statement:', error);
        return [];
    }
};

/**
 * Get company financial statements - Balance Sheet
 * @param symbol Stock symbol
 * @param period Period (quarter or annual)
 * @param limit Limit on number of results returned
 */
export const getBalanceSheet = async (symbol: string, period: string = 'annual', limit: number = 5) => {
    try {
        const response = await axios.get(`${BASE_URL}/balance-sheet-statement/${symbol}`, {
            params: {
                period,
                limit,
                apikey: API_KEY,
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching balance sheet:', error);
        return [];
    }
};

/**
 * Get company financial statements - Cash Flow Statement
 * @param symbol Stock symbol
 * @param period Period (quarter or annual)
 * @param limit Limit on number of results returned
 */
export const getCashFlowStatement = async (symbol: string, period: string = 'annual', limit: number = 5) => {
    try {
        const response = await axios.get(`${BASE_URL}/cash-flow-statement/${symbol}`, {
            params: {
                period,
                limit,
                apikey: API_KEY,
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching cash flow statement:', error);
        return [];
    }
};

/**
 * Get company financial ratios
 * @param symbol Stock symbol
 * @param period Period (quarter or annual)
 * @param limit Limit on number of results returned
 */
export const getFinancialRatios = async (symbol: string, period: string = 'annual', limit: number = 5) => {
    try {
        const response = await axios.get(`${BASE_URL}/ratios/${symbol}`, {
            params: {
                period,
                limit,
                apikey: API_KEY,
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching financial ratios:', error);
        return [];
    }
};

/**
 * Get company key metrics
 * @param symbol Stock symbol
 * @param period Period (quarter or annual)
 * @param limit Limit on number of results returned
 */
export const getKeyMetrics = async (symbol: string, period: string = 'annual', limit: number = 5) => {
    try {
        const response = await axios.get(`${BASE_URL}/key-metrics/${symbol}`, {
            params: {
                period,
                limit,
                apikey: API_KEY,
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching key metrics:', error);
        return [];
    }
};

/**
 * Get company financial growth data
 * @param symbol Stock symbol
 * @param period Period (quarter or annual)
 * @param limit Limit on number of results returned
 */
export const getFinancialGrowth = async (symbol: string, period: string = 'annual', limit: number = 5) => {
    try {
        const response = await axios.get(`${BASE_URL}/financial-growth/${symbol}`, {
            params: {
                period,
                limit,
                apikey: API_KEY,
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching financial growth:', error);
        return [];
    }
};

/**
 * Get company stock price target
 * @param symbol Stock symbol
 */
export const getPriceTarget = async (symbol: string) => {
    try {
        const response = await axios.get(`${BASE_URL}/price-target/${symbol}`, {
            params: {
                apikey: API_KEY,
            },
        });
        return response.data[0];
    } catch (error) {
        console.error('Error fetching price target:', error);
        return null;
    }
};

/**
 * Get company analyst ratings
 * @param symbol Stock symbol
 * @param limit Limit on number of results returned
 */
export const getAnalystRatings = async (symbol: string, limit: number = 10) => {
    try {
        const response = await axios.get(`${BASE_URL}/analyst-stock-recommendations/${symbol}`, {
            params: {
                limit,
                apikey: API_KEY,
            },
        });
        return response.data;
    } catch (error) {
        console.error('Error fetching analyst ratings:', error);
        return [];
    }
};

/**
 * Get company historical stock price data
 * @param symbol Stock symbol
 * @param from Start date (YYYY-MM-DD)
 * @param to End date (YYYY-MM-DD)
 */
export const getHistoricalPrices = async (symbol: string, from: string, to: string) => {
    try {
        const response = await axios.get(`${BASE_URL}/historical-price-full/${symbol}`, {
            params: {
                from,
                to,
                apikey: API_KEY,
            },
        });
        return response.data.historical;
    } catch (error) {
        console.error('Error fetching historical prices:', error);
        return [];
    }
}; 