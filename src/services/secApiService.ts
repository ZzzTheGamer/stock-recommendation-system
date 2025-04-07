import axios from 'axios';

const API_KEY = process.env.NEXT_PUBLIC_SEC_API_KEY || '';
const BASE_URL = 'https://api.sec-api.io';

/**
 * 获取公司的SEC文件
 * @param symbol 股票代码
 * @param formType 表格类型，如10-K, 10-Q等
 * @param limit 返回结果数量限制
 */
export const getSecFilings = async (symbol: string, formType: string = '10-K', limit: number = 10) => {
    try {
        console.log(`[SEC API] 获取 ${symbol} 的 ${formType} 文件...`);

        // 构造查询字符串
        let queryString = `ticker:${symbol}`;
        if (formType.includes(',')) {
            // 如果有多个表格类型，使用 OR 连接
            const formTypes = formType.split(',').map(f => `formType:"${f.trim()}"`).join(' OR ');
            queryString += ` AND (${formTypes})`;
        } else {
            queryString += ` AND formType:"${formType}"`;
        }

        const response = await axios.post(
            BASE_URL,
            {
                query: queryString,
                from: 0,
                size: limit,
                sort: [{ filedAt: { order: "desc" } }]
            },
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': API_KEY
                }
            }
        );

        console.log(`[SEC API] 成功获取 ${symbol} 的 ${formType} 文件`);
        return response.data.filings || [];
    } catch (error) {
        console.error('Error fetching SEC filings:', error);
        // 如果API调用失败，返回空数组
        return [];
    }
};

/**
 * 获取公司的最新年报(10-K)
 * @param symbol 股票代码
 */
export const getLatestAnnualReport = async (symbol: string) => {
    try {
        const filings = await getSecFilings(symbol, '10-K', 1);
        return filings.length > 0 ? filings[0] : null;
    } catch (error) {
        console.error('Error fetching latest annual report:', error);
        return null;
    }
};

/**
 * 获取公司的最新季报(10-Q)
 * @param symbol 股票代码
 */
export const getLatestQuarterlyReport = async (symbol: string) => {
    try {
        const filings = await getSecFilings(symbol, '10-Q', 1);
        return filings.length > 0 ? filings[0] : null;
    } catch (error) {
        console.error('Error fetching latest quarterly report:', error);
        return null;
    }
};

/**
 * 获取公司的XBRL数据
 * @param accessionNumber SEC文件的accession number
 */
export const getXbrlData = async (accessionNumber: string) => {
    try {
        // 注意：这个端点可能需要根据SEC API文档进行调整
        const response = await axios.post(
            BASE_URL,
            {
                query: `accessionNo:"${accessionNumber}"`,
                from: 0,
                size: 1
            },
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': API_KEY
                }
            }
        );
        return response.data;
    } catch (error) {
        console.error('Error fetching XBRL data:', error);
        return null;
    }
};

/**
 * 搜索SEC文件
 * @param query 搜索查询
 */
export const searchSecFilings = async (query: string) => {
    try {
        const response = await axios.post(
            BASE_URL,
            {
                query,
                from: 0,
                size: 10,
                sort: [{ filedAt: { order: "desc" } }]
            },
            {
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': API_KEY
                }
            }
        );
        return response.data.filings || [];
    } catch (error) {
        console.error('Error searching SEC filings:', error);
        return [];
    }
}; 